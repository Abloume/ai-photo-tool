package ai

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"image"
	"image/color"
	"image/png"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func tinyPNG(t *testing.T) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, 8, 8))
	img.Set(1, 1, color.RGBA{200, 30, 30, 255})
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatalf("生成测试 PNG 失败: %v", err)
	}
	return buf.Bytes()
}

// 图生图：multipart 字段要齐（model/prompt/size/image），b64_json 要能解出来
func TestImageGenEditMultipart(t *testing.T) {
	src := tinyPNG(t)
	var gotModel, gotPrompt, gotSize, gotName string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/images/edits" {
			t.Errorf("请求路径不对: %s", r.URL.Path)
		}
		if err := r.ParseMultipartForm(8 << 20); err != nil {
			t.Fatalf("不是合法 multipart: %v", err)
		}
		gotModel = r.FormValue("model")
		gotPrompt = r.FormValue("prompt")
		gotSize = r.FormValue("size")
		file, header, err := r.FormFile("image")
		if err != nil {
			t.Fatalf("缺少 image 字段: %v", err)
		}
		defer file.Close()
		gotName = header.Filename
		raw := make([]byte, 16)
		n, _ := file.Read(raw)
		if !bytes.HasPrefix(raw[:n], []byte("\x89PNG")) {
			t.Errorf("参考图不是 PNG: %v", raw[:n])
		}

		json.NewEncoder(w).Encode(map[string]any{
			"data": []any{map[string]string{
				"b64_json": base64.StdEncoding.EncodeToString(src),
			}},
		})
	}))
	defer srv.Close()

	g := NewImageGen(ImageGenConfig{
		APIKey: "test-key", BaseURL: srv.URL, Model: "gpt-image-1",
		Endpoint: "/images/edits", Size: "1024x1024", Timeout: 5 * time.Second,
	})
	out, mime, err := g.Generate(context.Background(), src, "photo.png", "让画面更暖")
	if err != nil {
		t.Fatalf("Generate 失败: %v", err)
	}
	if !bytes.Equal(out, src) {
		t.Error("返回的图片字节与 mock 不一致")
	}
	if mime != "image/png" {
		t.Errorf("mime 应为 image/png，实际 %s", mime)
	}
	if gotModel != "gpt-image-1" || gotPrompt != "让画面更暖" || gotSize != "1024x1024" {
		t.Errorf("表单字段不对: model=%q prompt=%q size=%q", gotModel, gotPrompt, gotSize)
	}
	if gotName != "photo.png" {
		t.Errorf("文件名应为 photo.png，实际 %q", gotName)
	}
}

// 返回 url 而不是 b64 时要能自己下载回来
func TestImageGenFetchesURL(t *testing.T) {
	jpg := []byte{0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10, 0x4a, 0x46}

	mux := http.NewServeMux()
	mux.HandleFunc("/images/edits", func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{
			"data": []any{map[string]string{"url": "http://" + r.Host + "/out.jpg"}},
		})
	})
	mux.HandleFunc("/out.jpg", func(w http.ResponseWriter, r *http.Request) {
		w.Write(jpg)
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	g := NewImageGen(ImageGenConfig{
		APIKey: "k", BaseURL: srv.URL, Model: "wanx-v1",
		Endpoint: "/images/edits", Size: "1024x1024", Timeout: 5 * time.Second,
	})
	out, mime, err := g.Generate(context.Background(), tinyPNG(t), "a.png", "换个天空")
	if err != nil {
		t.Fatalf("Generate 失败: %v", err)
	}
	if !bytes.Equal(out, jpg) {
		t.Error("下载到的图片字节不一致")
	}
	if mime != "image/jpeg" {
		t.Errorf("mime 应为 image/jpeg，实际 %s", mime)
	}
}

// 网关没有图像渠道时要返回能照做的提示，而不是一段生涩报错
func TestImageGenModelNotFound(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
		w.Write([]byte(`{"error":{"code":"model_not_found","message":"No available channel for model gpt-image-1 under group coding"}}`))
	}))
	defer srv.Close()

	g := NewImageGen(ImageGenConfig{
		APIKey: "k", BaseURL: srv.URL, Model: "gpt-image-1",
		Endpoint: "/images/edits", Size: "1024x1024", Timeout: 5 * time.Second,
	})
	_, _, err := g.Generate(context.Background(), tinyPNG(t), "a.png", "提亮")
	if err == nil {
		t.Fatal("应该报错")
	}
	if !strings.Contains(err.Error(), "AI_IMAGE_MODEL") {
		t.Errorf("错误信息应提示换 AI_IMAGE_MODEL，实际: %v", err)
	}
}

// 没配模型时 Enabled 为 false，且 Generate 直接给出配置提示
func TestImageGenNotConfigured(t *testing.T) {
	t.Setenv("AI_IMAGE_MODEL", "")
	t.Setenv("AI_IMAGE_API_KEY", "")
	t.Setenv("DEEPSEEK_API_KEY", "k")
	g := NewImageGen(ImageGenFromEnv())
	if g.Enabled() {
		t.Fatal("未配置 AI_IMAGE_MODEL 时不应启用")
	}
	if _, _, err := g.Generate(context.Background(), nil, "", "x"); err == nil ||
		!strings.Contains(err.Error(), "AI_IMAGE_MODEL") {
		t.Errorf("应提示配置 AI_IMAGE_MODEL，实际: %v", err)
	}
}

// 配置解析：DALL·E 默认走文生图，其他默认走图生图
func TestImageGenFromEnvEndpoints(t *testing.T) {
	t.Setenv("AI_IMAGE_API_KEY", "k")
	t.Setenv("AI_IMAGE_MODEL", "dall-e-3")
	t.Setenv("AI_IMAGE_BASE_URL", "https://example.com/v1")
	t.Setenv("AI_IMAGE_ENDPOINT", "")
	if cfg := ImageGenFromEnv(); cfg.Endpoint != "/images/generations" {
		t.Errorf("DALL·E 应默认 /images/generations，实际 %s", cfg.Endpoint)
	}

	t.Setenv("AI_IMAGE_MODEL", "gpt-image-1")
	if cfg := ImageGenFromEnv(); cfg.Endpoint != "/images/edits" {
		t.Errorf("其他模型应默认 /images/edits，实际 %s", cfg.Endpoint)
	}

	t.Setenv("AI_IMAGE_ENDPOINT", "images/custom")
	if cfg := ImageGenFromEnv(); cfg.Endpoint != "/images/custom" {
		t.Errorf("自定义 endpoint 应补前导斜杠，实际 %s", cfg.Endpoint)
	}
}
