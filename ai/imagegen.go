package ai

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// ImageGenConfig 图生图接入配置。
// 只依赖 OpenAI 兼容的 /images/edits（或 /images/generations），
// 换网关/换模型只改环境变量，不动代码。
type ImageGenConfig struct {
	APIKey   string
	BaseURL  string
	Model    string // 例如 gpt-image-1 / doubao-seedream-4-0 / wanx-v1
	Endpoint string // 默认 /images/edits；只有文生图的模型填 /images/generations
	Size     string // 默认 1024x1024
	Timeout  time.Duration
}

// ImageGenFromEnv 从环境变量读图生图配置，全部可缺省（缺 Model 即视为未启用）
func ImageGenFromEnv() ImageGenConfig {
	cfg := ImageGenConfig{
		APIKey:   firstEnv("AI_IMAGE_API_KEY", "DEEPSEEK_API_KEY", "AI_API_KEY"),
		BaseURL:  firstEnv("AI_IMAGE_BASE_URL", "DEEPSEEK_BASE_URL", "AI_BASE_URL"),
		Model:    os.Getenv("AI_IMAGE_MODEL"),
		Endpoint: os.Getenv("AI_IMAGE_ENDPOINT"),
		Size:     os.Getenv("AI_IMAGE_SIZE"),
		Timeout:  120 * time.Second, // 出图比出文字慢得多
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = "https://api.openai.com/v1"
	}
	if cfg.Endpoint == "" {
		// DALL·E 系列只支持纯文生图；其他按图生图处理
		if strings.HasPrefix(strings.ToLower(cfg.Model), "dall-e") {
			cfg.Endpoint = "/images/generations"
		} else {
			cfg.Endpoint = "/images/edits"
		}
	}
	if !strings.HasPrefix(cfg.Endpoint, "/") {
		cfg.Endpoint = "/" + cfg.Endpoint
	}
	if cfg.Size == "" {
		cfg.Size = "1024x1024"
	}
	return cfg
}

// ImageGen 极简图生图客户端
type ImageGen struct {
	cfg  ImageGenConfig
	http *http.Client
}

// NewImageGen 构造客户端
func NewImageGen(cfg ImageGenConfig) *ImageGen {
	return &ImageGen{
		cfg:  cfg,
		http: &http.Client{Timeout: cfg.Timeout},
	}
}

// Enabled 是否配置了图像模型（没有就不该把按钮露给用户点）
func (g *ImageGen) Enabled() bool {
	return g != nil && g.cfg.Model != "" && g.cfg.APIKey != ""
}

// Model 当前图像模型名，用于日志与前端提示
func (g *ImageGen) Model() string {
	if g == nil {
		return ""
	}
	return g.cfg.Model
}

// IsEdit 是否带参考图（edits）而不是纯文生图
func (g *ImageGen) IsEdit() bool {
	return g != nil && strings.Contains(g.cfg.Endpoint, "edits")
}

// imageResult 兼容 OpenAI（data[].b64_json / data[].url）和部分网关的 output.results[].url
type imageResult struct {
	Data []struct {
		B64JSON string `json:"b64_json"`
		URL     string `json:"url"`
	} `json:"data"`
	Output struct {
		Results []struct {
			URL string `json:"url"`
		} `json:"results"`
	} `json:"output"`
	Error *struct {
		Message string `json:"message"`
		Code    any    `json:"code"`
	} `json:"error"`
}

// Generate 用提示词（可选参考图）生成一张图，返回图片字节和 mime。
// imageBytes 为空时退化成纯文生图。
func (g *ImageGen) Generate(ctx context.Context, imageBytes []byte, filename, prompt string) ([]byte, string, error) {
	if !g.Enabled() {
		return nil, "", fmt.Errorf("未配置图像模型：请设置 AI_IMAGE_MODEL（以及 AI_IMAGE_API_KEY）")
	}
	if strings.TrimSpace(prompt) == "" {
		return nil, "", fmt.Errorf("提示词为空")
	}

	url := strings.TrimSuffix(g.cfg.BaseURL, "/") + g.cfg.Endpoint

	var (
		req *http.Request
		err error
	)
	if g.IsEdit() && len(imageBytes) > 0 {
		req, err = g.newEditRequest(ctx, url, imageBytes, filename, prompt)
	} else {
		req, err = g.newJSONRequest(ctx, url, prompt)
	}
	if err != nil {
		return nil, "", err
	}

	resp, err := g.http.Do(req)
	if err != nil {
		return nil, "", fmt.Errorf("请求图像接口失败: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(io.LimitReader(resp.Body, 32<<20))
	if err != nil {
		return nil, "", err
	}

	var parsed imageResult
	jsonErr := json.Unmarshal(body, &parsed)
	if resp.StatusCode != http.StatusOK {
		// 网关常见坑：模型没开渠道 / 分组不支持，给出能直接照做的提示
		msg := strings.TrimSpace(string(body))
		if jsonErr == nil && parsed.Error != nil && parsed.Error.Message != "" {
			msg = parsed.Error.Message
		}
		return nil, "", fmt.Errorf("%s（HTTP %d）", friendlyImageErr(msg), resp.StatusCode)
	}
	if jsonErr != nil {
		return nil, "", fmt.Errorf("图像接口返回的不是 JSON: %s", truncate(string(body), 200))
	}
	if parsed.Error != nil && parsed.Error.Message != "" {
		return nil, "", fmt.Errorf("%s", friendlyImageErr(parsed.Error.Message))
	}

	for _, d := range parsed.Data {
		if d.B64JSON != "" {
			raw, err := base64.StdEncoding.DecodeString(d.B64JSON)
			if err != nil {
				return nil, "", fmt.Errorf("解码 b64_json 失败: %w", err)
			}
			return raw, sniffImageType(raw), nil
		}
		if d.URL != "" {
			return g.fetchImage(ctx, d.URL)
		}
	}
	for _, r := range parsed.Output.Results {
		if r.URL != "" {
			return g.fetchImage(ctx, r.URL)
		}
	}
	return nil, "", fmt.Errorf("图像接口没有返回图片: %s", truncate(string(body), 200))
}

func (g *ImageGen) newEditRequest(ctx context.Context, url string, imageBytes []byte, filename, prompt string) (*http.Request, error) {
	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)

	fields := map[string]string{
		"model":           g.cfg.Model,
		"prompt":          prompt,
		"n":               "1",
		"size":            g.cfg.Size,
		"response_format": "b64_json",
	}
	for k, v := range fields {
		if err := mw.WriteField(k, v); err != nil {
			return nil, err
		}
	}

	name := filepath.Base(filename)
	if name == "" || name == "." || !hasImageExt(name) {
		name = "source" + extForType(sniffImageType(imageBytes))
	}
	part, err := mw.CreateFormFile("image", name)
	if err != nil {
		return nil, err
	}
	if _, err := part.Write(imageBytes); err != nil {
		return nil, err
	}
	if err := mw.Close(); err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, &buf)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", mw.FormDataContentType())
	req.Header.Set("Authorization", "Bearer "+g.cfg.APIKey)
	return req, nil
}

func (g *ImageGen) newJSONRequest(ctx context.Context, url, prompt string) (*http.Request, error) {
	payload, err := json.Marshal(map[string]any{
		"model":           g.cfg.Model,
		"prompt":          prompt,
		"n":               1,
		"size":            g.cfg.Size,
		"response_format": "b64_json",
	})
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+g.cfg.APIKey)
	return req, nil
}

func (g *ImageGen) fetchImage(ctx context.Context, url string) ([]byte, string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, "", err
	}
	resp, err := g.http.Do(req)
	if err != nil {
		return nil, "", fmt.Errorf("下载生成图失败: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, "", fmt.Errorf("下载生成图失败: HTTP %d", resp.StatusCode)
	}
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 32<<20))
	if err != nil {
		return nil, "", err
	}
	if len(raw) == 0 {
		return nil, "", fmt.Errorf("下载生成图失败: 内容为空")
	}
	return raw, sniffImageType(raw), nil
}

// friendlyImageErr 把网关的原始报错翻成人话
func friendlyImageErr(msg string) string {
	lower := strings.ToLower(msg)
	switch {
	case strings.Contains(lower, "model_not_found"), strings.Contains(lower, "no available channel"):
		return "当前网关没有这个图像模型的渠道，换一个 AI_IMAGE_MODEL 或联系管理员开通：" + msg
	case strings.Contains(lower, "invalid_api_key"), strings.Contains(lower, "unauthorized"):
		return "图像接口鉴权失败，检查 AI_IMAGE_API_KEY：" + msg
	case strings.Contains(lower, "moderation"), strings.Contains(lower, "safety"):
		return "提示词被安全策略拦截，换个说法再试：" + msg
	}
	return msg
}

func sniffImageType(b []byte) string {
	switch {
	case len(b) > 8 && bytes.HasPrefix(b, []byte("\x89PNG")):
		return "image/png"
	case len(b) > 3 && bytes.HasPrefix(b, []byte("\xff\xd8\xff")):
		return "image/jpeg"
	case len(b) > 12 && bytes.HasPrefix(b, []byte("RIFF")) && bytes.Equal(b[8:12], []byte("WEBP")):
		return "image/webp"
	}
	return "image/png"
}

func extForType(mime string) string {
	switch mime {
	case "image/jpeg":
		return ".jpg"
	case "image/webp":
		return ".webp"
	}
	return ".png"
}

func hasImageExt(name string) bool {
	switch strings.ToLower(filepath.Ext(name)) {
	case ".png", ".jpg", ".jpeg", ".webp":
		return true
	}
	return false
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}
