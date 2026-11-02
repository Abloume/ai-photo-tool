package ai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"ai-photo-tool/analyzer"
)

// 用一个假的兼容接口验证：请求格式、代码块剥离、参数映射与范围夹取
func TestGenerateParsesModelJSON(t *testing.T) {
	var gotModel, gotThinking string
	var gotMaxTokens int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Errorf("请求路径不对: %s", r.URL.Path)
		}
		var body struct {
			Model     string `json:"model"`
			MaxTokens int    `json:"max_tokens"`
			Thinking  *struct {
				Type string `json:"type"`
			} `json:"thinking"`
		}
		json.NewDecoder(r.Body).Decode(&body)
		gotModel = body.Model
		gotMaxTokens = body.MaxTokens
		if body.Thinking != nil {
			gotThinking = body.Thinking.Type
		}

		content := "```json\n" + `{"summary":"偏暗人像","tags":[
			{"label":"提亮暗部","reason":"阴影太黑","prompt":"整体提亮曝光","params":{"brightness":20,"saturation":999}},
			{"label":"","prompt":"label 为空应被丢弃"}]}` + "\n```"
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []any{map[string]any{"message": map[string]string{"content": content}}},
		})
	}))
	defer srv.Close()

	c := NewClient(Config{APIKey: "test-key", BaseURL: srv.URL, Model: "deepseek-chat", Timeout: 5 * time.Second})
	sug := Generate(context.Background(), c, analyzer.Features{}, "")

	if sugon := sug.Source; sugon != "deepseek" {
		t.Fatalf("source 应为 deepseek，实际 %s", sugon)
	}
	if gotModel != "deepseek-chat" {
		t.Errorf("请求模型名不对: %s", gotModel)
	}
	// 短 JSON 任务不该开思考模式（默认 high 会拖到几十秒）
	if gotThinking != "disabled" {
		t.Errorf("thinking 应显式关闭，实际 %q", gotThinking)
	}
	if gotMaxTokens <= 0 {
		t.Errorf("应设置 max_tokens 防止 JSON 被截断，实际 %d", gotMaxTokens)
	}
	if len(sug.Tags) != 1 {
		t.Fatalf("应只保留 1 个有效标签，实际 %d", len(sug.Tags))
	}
	tag := sug.Tags[0]
	if tag.Filter.Brightness != 20 {
		t.Errorf("brightness 应为 20，实际 %v", tag.Filter.Brightness)
	}
	if tag.Filter.Saturation != 200 {
		t.Errorf("越界的 saturation 应夹到 200，实际 %v", tag.Filter.Saturation)
	}
	if tag.Filter.Intensity != 100 {
		t.Errorf("intensity 默认应为 100，实际 %v", tag.Filter.Intensity)
	}
}

// 官方在售模型里只有 deepseek-flash（V4.1-Flash）能看图，v4-pro 不能
func TestSupportsVision(t *testing.T) {
	cases := map[string]bool{
		"deepseek-flash":               true,
		"deepseek-v4-flash":            true,
		"deepseek-v4-flash-vision-exp": true,
		"deepseek-v4-pro":              false,
		"deepseek-chat":                false,
		"":                             false,
	}
	for model, want := range cases {
		if got := (&Client{cfg: Config{Model: model}}).VisionEnabled(); got != want {
			t.Errorf("%q VisionEnabled 应为 %v，实际 %v", model, want, got)
		}
	}
	// 显式指定视觉模型时，即使主模型不支持也发图
	if !(&Client{cfg: Config{Model: "deepseek-v4-pro", VisionModel: "deepseek-flash"}}).VisionEnabled() {
		t.Error("配置了 VisionModel 时应启用视觉")
	}
}

// 没配密钥时必须走离线规则，且至少给够能点的标签
func TestGenerateFallsBackOffline(t *testing.T) {
	feats := analyzer.Features{
		Stats:          analyzer.ImageStats{IsDark: true, IsWarm: true},
		Aspect:         "3:2",
		ShadowClip:     0.2,
		Sharpness:      12,
		DominantColors: []string{"#112233"},
	}
	sug := Generate(context.Background(), NewClient(Config{}), feats, "")

	if sug.Source != "offline" {
		t.Fatalf("未配置密钥时应为 offline，实际 %s", sug.Source)
	}
	if len(sug.Tags) < 3 {
		t.Fatalf("离线建议至少 3 个标签，实际 %d", len(sug.Tags))
	}
	for _, tag := range sug.Tags {
		if tag.Label == "" || tag.Prompt == "" {
			t.Errorf("标签或提示词为空: %+v", tag)
		}
	}
}
