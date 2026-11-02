package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"ai-photo-tool/analyzer"
	"ai-photo-tool/filters"
)

// Config 大模型接入配置。默认走 DeepSeek 官方接口；
// 换成任何兼容 OpenAI /chat/completions 的网关只需要改 BaseURL。
type Config struct {
	APIKey      string
	BaseURL     string
	Model       string // 默认 deepseek-flash（V4.1-Flash，原生支持图片输入）
	VisionModel string // 可选：只给图片请求换一个模型；不填就用 Model
	Thinking    bool   // 是否开启思考模式，默认关（默认思考 = high，等 30s+ 才出标签没必要）
	Timeout     time.Duration
}

// ConfigFromEnv 从环境变量读配置，全部可缺省
func ConfigFromEnv() Config {
	cfg := Config{
		APIKey:      firstEnv("DEEPSEEK_API_KEY", "AI_API_KEY"),
		BaseURL:     firstEnv("DEEPSEEK_BASE_URL", "AI_BASE_URL"),
		Model:       firstEnv("DEEPSEEK_MODEL", "AI_MODEL"),
		VisionModel: os.Getenv("AI_VISION_MODEL"),
		Thinking:    os.Getenv("AI_THINKING") == "1" || strings.EqualFold(os.Getenv("AI_THINKING"), "true"),
		Timeout:     45 * time.Second,
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = "https://api.deepseek.com"
	}
	if cfg.Model == "" {
		// deepseek-flash = V4.1-Flash，官方当前唯一支持图像理解的在售模型；
		// deepseek-v4-pro 反而看不了图，别拿它当视觉模型
		cfg.Model = "deepseek-flash"
	}
	return cfg
}

func firstEnv(keys ...string) string {
	for _, k := range keys {
		if v := os.Getenv(k); v != "" {
			return v
		}
	}
	return ""
}

// Client 极简 chat/completions 客户端，够用就行，不引第三方 SDK
type Client struct {
	cfg  Config
	http *http.Client
}

// NewClient 创建客户端；没有 API Key 时 Enabled() 为 false，调用方应走离线规则
func NewClient(cfg Config) *Client {
	timeout := cfg.Timeout
	if timeout <= 0 {
		timeout = 45 * time.Second
	}
	return &Client{
		cfg:  cfg,
		http: &http.Client{Timeout: timeout},
	}
}

// Enabled 是否配置了密钥
func (c *Client) Enabled() bool {
	return c != nil && c.cfg.APIKey != ""
}

// VisionEnabled 当前是否会把图片一起发给模型。
// 配了 AI_VISION_MODEL 就相信调用方；否则按模型名判断（deepseek-flash 能看图，deepseek-v4-pro 不能）。
func (c *Client) VisionEnabled() bool {
	if c == nil {
		return false
	}
	if c.cfg.VisionModel != "" {
		return true
	}
	return supportsVision(c.cfg.Model)
}

// supportsVision 粗略判断模型是否具备图片输入能力。
// 官方当前在售：deepseek-flash ✅、deepseek-v4-pro ❌；旧名 deepseek-v4-flash* 已由 Flash 承接。
func supportsVision(model string) bool {
	m := strings.ToLower(model)
	if m == "" {
		return false
	}
	return strings.Contains(m, "flash") || strings.Contains(m, "vision") || strings.Contains(m, "-vl")
}

// Model 暴露当前文本模型名，用于启动日志
func (c *Client) Model() string {
	if c == nil {
		return ""
	}
	return c.cfg.Model
}

// VisionModel 暴露实际承担看图的模型名，用于启动日志
func (c *Client) VisionModel() string {
	if c == nil || !c.VisionEnabled() {
		return ""
	}
	if c.cfg.VisionModel != "" {
		return c.cfg.VisionModel
	}
	return c.cfg.Model
}

func (c *Client) modelFor(imageDataURL string) string {
	if imageDataURL != "" && c.cfg.VisionModel != "" {
		return c.cfg.VisionModel
	}
	return c.cfg.Model
}

// 以下是 OpenAI 兼容协议的最小结构体
type chatMessage struct {
	Role    string `json:"role"`
	Content any    `json:"content"`
}

type imageURLPart struct {
	URL string `json:"url"`
}

type contentPart struct {
	Type     string        `json:"type"`
	Text     string        `json:"text,omitempty"`
	ImageURL *imageURLPart `json:"image_url,omitempty"`
}

type responseFormat struct {
	Type string `json:"type"`
}

// thinkingConfig DeepSeek 的思考模式开关（默认 enabled + effort high，短任务没必要）
type thinkingConfig struct {
	Type string `json:"type"` // enabled / disabled
}

type chatRequest struct {
	Model          string          `json:"model"`
	Messages       []chatMessage   `json:"messages"`
	Temperature    float64         `json:"temperature"`
	MaxTokens      int             `json:"max_tokens,omitempty"`
	ResponseFormat *responseFormat `json:"response_format,omitempty"`
	Thinking       *thinkingConfig `json:"thinking,omitempty"`
}

type chatResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
}

// 模型返回的原始结构，params 用 map 接收，方便逐项夹范围
type rawSuggestion struct {
	Summary string `json:"summary"`
	Tags    []struct {
		Label  string             `json:"label"`
		Reason string             `json:"reason"`
		Prompt string             `json:"prompt"`
		Params map[string]float64 `json:"params"`
	} `json:"tags"`
}

const systemPrompt = `你是一名资深商业修图师，同时也是 AI 修图提示词工程师。
用户会给你一张照片（可能带图片）以及它的量化特征：曝光、白平衡、饱和度、对比度、锐度、主色、肤色/天空/绿植占比。
请先看图判断主体和场景（人像/风光/食物/夜景/街拍…），再结合数值找出最需要修的地方，
给出 4-6 个优化标签，每个标签配一句可直接用于 AI 修图的提示词，
以及一组能通过曝光/色温/饱和度/对比度/高光阴影等参数实现的本地调整值。
要求：
- 标签文案是中文短词，2-6 个字，像"提亮暗部""通透蓝天"这种，不要编号
- 标签必须针对这张图的具体问题和主体，不要给通用套话；人像优先肤色，风光优先天空和层次
- 图片和量化特征冲突时以图片为准（比如数值说偏暖但画面是雪景，就别硬拉冷）
- prompt 是给生成式模型的中文指令，60 字以内，写清目标色调、光线、质感、氛围
- 只输出 JSON，不要 markdown 代码块，不要额外解释`

// suggest 调一次大模型生成建议
func (c *Client) suggest(ctx context.Context, feats analyzer.Features, imageDataURL string) (*Suggestion, error) {
	featJSON, err := json.MarshalIndent(feats, "", "  ")
	if err != nil {
		return nil, err
	}

	userText := fmt.Sprintf(`图片特征：
%s

请输出 JSON，格式：
{"summary": "一句话概括画面状态和主要问题，40 字以内",
 "tags": [{"label": "标签", "reason": "为什么这么建议，20 字以内",
           "prompt": "可直接用于 AI 修图的提示词", "params": {"brightness": 10}}]}

params 只允许使用这些键，数值必须在范围内：
temperature(-100~100 暖为正), tint(-100~100 绿为正), saturation(0~200, 100 为原始),
contrast(0~200, 100 为原始), brightness(-100~100), highlights(-100~100 提亮为正),
shadows(-100~100 提亮为正), vignette(0~100), grain(0~100),
highlightTintHue(0~360), highlightTintAmount(-100~100),
shadowTintHue(0~360), shadowTintAmount(-100~100)。
没把握的键直接省略，不要输出 0 值凑数。`, string(featJSON))

	// 图片只能放在 user 消息里；content 必须是块数组，不能是纯字符串
	textPart := contentPart{Type: "text", Text: userText}
	var userContent any
	if imageDataURL != "" && c.VisionEnabled() {
		userContent = []contentPart{
			textPart,
			{Type: "image_url", ImageURL: &imageURLPart{URL: imageDataURL}},
		}
	} else {
		userContent = userText
	}

	body := chatRequest{
		Model:       c.modelFor(imageDataURL),
		Temperature: 0.4,
		MaxTokens:   2048, // JSON 被截断就白跑了
		Messages: []chatMessage{
			{Role: "system", Content: systemPrompt},
			{Role: "user", Content: userContent},
		},
		ResponseFormat: &responseFormat{Type: "json_object"},
	}
	if c.cfg.Thinking {
		body.Thinking = &thinkingConfig{Type: "enabled"}
	} else {
		body.Thinking = &thinkingConfig{Type: "disabled"}
	}

	content, err := c.chat(ctx, body)
	if err != nil && (strings.Contains(err.Error(), "response_format") || strings.Contains(err.Error(), "HTTP 400")) {
		// 部分网关/模型组合不支持 JSON 模式，去掉后重试一次
		body.ResponseFormat = nil
		content, err = c.chat(ctx, body)
	}
	if err != nil {
		return nil, err
	}

	raw, err := parseSuggestion(content)
	if err != nil && body.ResponseFormat != nil {
		// JSON Output 有把非 JSON 内容洗成空格的坑，关掉重试一次
		body.ResponseFormat = nil
		if retried, retryErr := c.chat(ctx, body); retryErr == nil {
			raw, err = parseSuggestion(retried)
		}
	}
	if err != nil {
		return nil, err
	}

	sug := &Suggestion{Summary: strings.TrimSpace(raw.Summary)}
	for _, t := range raw.Tags {
		label := strings.TrimSpace(t.Label)
		prompt := strings.TrimSpace(t.Prompt)
		if label == "" || prompt == "" {
			continue
		}
		flt := baseFilter()
		for k, v := range t.Params {
			applyParam(&flt, k, v)
		}
		flt.Name = label
		flt.Description = strings.TrimSpace(t.Reason)
		sug.Tags = append(sug.Tags, Tag{
			Label:  label,
			Reason: strings.TrimSpace(t.Reason),
			Prompt: prompt,
			Filter: SanitizeFilter(flt),
		})
		if len(sug.Tags) == 6 {
			break
		}
	}
	if len(sug.Tags) == 0 {
		return nil, fmt.Errorf("模型没有返回有效标签")
	}
	return sug, nil
}

// parseSuggestion 解析模型返回的 JSON，兼容代码块包裹和空内容
func parseSuggestion(content string) (rawSuggestion, error) {
	var raw rawSuggestion
	trimmed := strings.TrimSpace(extractJSON(content))
	if trimmed == "" || trimmed == "{}" {
		return raw, fmt.Errorf("模型返回为空（JSON 模式下非 JSON 内容会被清空）")
	}
	if err := json.Unmarshal([]byte(trimmed), &raw); err != nil {
		return raw, fmt.Errorf("解析模型返回失败: %w", err)
	}
	if len(raw.Tags) == 0 {
		return raw, fmt.Errorf("模型没有返回有效标签")
	}
	return raw, nil
}

// chat 发一次请求，返回 message.content
func (c *Client) chat(ctx context.Context, body chatRequest) (string, error) {
	payload, err := json.Marshal(body)
	if err != nil {
		return "", err
	}

	url := strings.TrimSuffix(c.cfg.BaseURL, "/") + "/chat/completions"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.cfg.APIKey)

	resp, err := c.http.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return "", err
	}
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(data)))
	}

	var parsed chatResponse
	if err := json.Unmarshal(data, &parsed); err != nil {
		return "", err
	}
	if parsed.Error != nil && parsed.Error.Message != "" {
		return "", fmt.Errorf("模型返回错误: %s", parsed.Error.Message)
	}
	if len(parsed.Choices) == 0 {
		return "", fmt.Errorf("模型返回为空")
	}
	return parsed.Choices[0].Message.Content, nil
}

// extractJSON 容错：模型偶尔会包一层 ```json 代码块
func extractJSON(s string) string {
	start := strings.Index(s, "{")
	end := strings.LastIndex(s, "}")
	if start >= 0 && end > start {
		return s[start : end+1]
	}
	return s
}

// applyParam 把模型给的键值映射到具体的滤镜字段
func applyParam(f *filters.Filter, key string, v float64) {
	switch key {
	case "temperature":
		f.Temperature = v
	case "tint":
		f.Tint = v
	case "saturation":
		f.Saturation = v
	case "contrast":
		f.Contrast = v
	case "brightness":
		f.Brightness = v
	case "highlights":
		f.Highlights = v
	case "shadows":
		f.Shadows = v
	case "vignette":
		f.Vignette = v
	case "grain":
		f.Grain = v
	case "highlightTintHue":
		f.HighlightTintHue = v
	case "highlightTintAmount":
		f.HighlightTintAmount = v
	case "shadowTintHue":
		f.ShadowTintHue = v
	case "shadowTintAmount":
		f.ShadowTintAmount = v
	case "intensity":
		f.Intensity = v
	}
}
