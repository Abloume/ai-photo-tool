// Package ai 调用大模型（DeepSeek / 兼容 OpenAI 协议的网关）根据图片特征
// 生成修图优化标签和对应提示词；没配置密钥时自动退回离线规则。
package ai

import (
	"bytes"
	"context"
	"encoding/base64"
	"fmt"
	"image"
	"image/jpeg"
	"log"
	"os"
	"strings"

	"github.com/disintegration/imaging"

	"ai-photo-tool/analyzer"
	"ai-photo-tool/filters"
)

// Tag 一条优化建议：标签 + 为什么这么建议 + 可直接复用的提示词 + 本地可执行的滤镜参数
type Tag struct {
	Label  string         `json:"label"`
	Reason string         `json:"reason"`
	Prompt string         `json:"prompt"`
	Filter filters.Filter `json:"filter"`
}

// Suggestion AI 分析结果
type Suggestion struct {
	Summary string `json:"summary"`
	Tags    []Tag  `json:"tags"`
	Source  string `json:"source"` // deepseek / offline
	Model   string `json:"model"`  // 实际使用的模型，离线模式为空
}

// Generate 生成建议：优先走大模型，失败或未配置则用离线规则兜底。
// imageDataURL 只在配置了视觉模型时才会用到，纯文本模型传空字符串即可。
func Generate(ctx context.Context, c *Client, feats analyzer.Features, imageDataURL string) Suggestion {
	if c != nil && c.Enabled() {
		s, err := c.suggest(ctx, feats, imageDataURL)
		if err != nil {
			log.Printf("⚠️  AI 建议调用失败，退回离线规则: %v", err)
		} else if len(s.Tags) > 0 {
			s.Source = "deepseek"
			s.Model = c.modelFor(imageDataURL)
			return *s
		}
	}
	s := OfflineSuggest(feats)
	s.Source = "offline"
	return s
}

// OfflineSuggest 不依赖网络的规则版建议：从特征里判断需要修什么，
// 每个标签都带一句提示词和一组本地滤镜参数，保证「点标签就有反应」。
func OfflineSuggest(f analyzer.Features) Suggestion {
	st := f.Stats
	base := baseFilter()

	var tags []Tag
	add := func(label, reason, prompt string, mut func(*filters.Filter)) {
		flt := base
		mut(&flt)
		flt.Name = label
		tags = append(tags, Tag{
			Label:  label,
			Reason: reason,
			Prompt: prompt,
			Filter: SanitizeFilter(flt),
		})
	}

	if st.IsDark || f.ShadowClip > 0.08 {
		add("提亮暗部", "画面整体偏暗，阴影里的细节看不清",
			"整体提亮曝光，重点恢复暗部细节，适度压住高光避免发灰，保留自然的明暗层次。",
			func(x *filters.Filter) {
				x.Brightness += 12
				x.Shadows += 25
				x.Highlights -= 8
				x.Contrast += 5
			})
	}
	if st.IsBright || f.HighlightClip > 0.06 {
		add("压暗高光", "高光区域接近溢出，亮部层次丢失",
			"压低高光找回亮部细节，整体亮度略降，让天空和反光不过曝，画面更耐看。",
			func(x *filters.Filter) {
				x.Brightness -= 8
				x.Highlights -= 25
				x.Contrast += 6
			})
	}
	if st.IsWarm {
		add("校正偏暖", "白平衡偏暖，画面发黄",
			"把色温往中性偏冷方向拉一点，去掉泛黄的色偏，还原干净准确的白平衡。",
			func(x *filters.Filter) {
				x.Temperature -= 14
				x.Tint -= 3
				x.Saturation += 3
			})
	}
	if st.IsCool {
		add("校正偏冷", "白平衡偏冷，画面发蓝",
			"把色温往暖方向提一点，减少阴影里的蓝色调，让肤色和整体观感更自然。",
			func(x *filters.Filter) {
				x.Temperature += 14
				x.Tint += 3
				x.Saturation += 3
			})
	}
	// 高饱和的图不该再"提升通透"加饱和，否则和"收敛色彩"打架
	if st.IsLowSat || (!st.IsHighSat && f.Contrast < 30) {
		add("提升通透", "画面发灰、层次偏平",
			"提高对比度和自然饱和度，压一点灰雾感，让画面通透干净、层次分明。",
			func(x *filters.Filter) {
				x.Contrast += 16
				x.Saturation += 18
				x.Highlights -= 5
				x.Shadows += 5
			})
	}
	if st.IsHighSat || st.Saturation > 0.55 {
		add("收敛色彩", "色彩过于浓艳，容易失真",
			"适当降低饱和度，收紧过艳的红色和绿色，保留真实质感，观感更高级。",
			func(x *filters.Filter) {
				x.Saturation -= 16
				x.Contrast += 4
			})
	}
	if f.Sharpness < 25 || f.Contrast < 34 {
		add("增强质感", "细节偏软，画面不够锐利",
			"增强局部对比和清晰度感，让纹理和边缘更扎实，整体更具照片质感。",
			func(x *filters.Filter) {
				x.Contrast += 12
				x.Saturation += 6
				x.Highlights += 6
				x.Shadows -= 4
			})
	}
	if f.SkinRatio > 0.12 {
		add("自然肤色", "画面中人物肤色占比不低",
			"轻微提亮并柔化肤色，压制偏红偏黄的色偏，保留皮肤质感，呈现自然通透的人像效果。",
			func(x *filters.Filter) {
				x.Brightness += 4
				x.Tint += 4
				x.Temperature += 3
				x.Saturation -= 4
				x.Highlights += 5
			})
	}
	if f.SkyRatio > 0.15 {
		add("通透蓝天", "天空/水面占比高，蓝色容易发灰",
			"加深蓝色并提升通透度，让天空和水面更干净，云层层次分明。",
			func(x *filters.Filter) {
				x.Saturation += 14
				x.Contrast += 10
				x.Temperature -= 6
				x.Highlights -= 6
			})
	}

	// 兜底：曝光白平衡都正常的图也得有得点
	if len(tags) < 3 {
		add("自然通透", "画面基础不错，只需要轻微提气",
			"轻微提升对比与饱和度，让画面更通透干净，保持真实自然不做作。",
			func(x *filters.Filter) {
				x.Contrast += 8
				x.Saturation += 8
				x.Shadows += 4
			})
		add("胶片质感", "适合加一点胶片氛围",
			"加入轻微胶片颗粒和暖调，高光泛一点点奶油色，营造复古而有温度的胶片氛围。",
			func(x *filters.Filter) {
				x.Temperature += 8
				x.Grain += 18
				x.Contrast += 6
				x.HighlightTintHue = 45
				x.HighlightTintAmount = 10
				x.ShadowTintHue = 210
				x.ShadowTintAmount = 8
			})
		add("电影感", "适合压暗四周、强化氛围",
			"压暗四角并加强明暗对比，阴影偏冷、高光偏暖，做出电影画面般的氛围感。",
			func(x *filters.Filter) {
				x.Contrast += 12
				x.Vignette += 18
				x.Shadows -= 6
				x.Highlights -= 4
				x.HighlightTintHue = 40
				x.HighlightTintAmount = 12
				x.ShadowTintHue = 215
				x.ShadowTintAmount = 12
			})
	}

	if len(tags) > 6 {
		tags = tags[:6]
	}

	return Suggestion{Summary: offlineSummary(f), Tags: tags}
}

// offlineSummary 用一句话说清这张图的状态，替代大模型的画面描述
func offlineSummary(f analyzer.Features) string {
	st := f.Stats
	parts := []string{fmt.Sprintf("%s 画幅", f.Aspect)}

	switch {
	case st.IsDark:
		parts = append(parts, "整体偏暗")
	case st.IsBright:
		parts = append(parts, "整体偏亮")
	default:
		parts = append(parts, "曝光基本正常")
	}

	switch {
	case st.IsWarm:
		parts = append(parts, "色调偏暖")
	case st.IsCool:
		parts = append(parts, "色调偏冷")
	default:
		parts = append(parts, "白平衡接近中性")
	}

	switch {
	case f.SkinRatio > 0.15:
		parts = append(parts, "人物/肤色占比大")
	case f.SkyRatio > 0.15:
		parts = append(parts, "天空或水面占比高")
	case f.GreenRatio > 0.2:
		parts = append(parts, "绿植较多")
	}

	if f.Sharpness < 25 {
		parts = append(parts, "细节略软")
	}

	return strings.Join(parts, "，") + "。"
}

// baseFilter 所有建议都从"不改动"的中性参数出发，再叠加具体调整
func baseFilter() filters.Filter {
	return filters.Filter{
		Category:    "ai",
		Description: "AI 建议生成",
		Saturation:  100,
		Contrast:    100,
		Intensity:   100,
	}
}

// SanitizeFilter 把参数夹到滤镜支持的范围，防止模型返回离谱数值（如亮度 800）
func SanitizeFilter(f filters.Filter) filters.Filter {
	clamp := func(v, min, max float64) float64 {
		if v < min {
			return min
		}
		if v > max {
			return max
		}
		return v
	}
	f.Temperature = clamp(f.Temperature, -100, 100)
	f.Tint = clamp(f.Tint, -100, 100)
	f.Saturation = clamp(f.Saturation, 0, 200)
	f.Contrast = clamp(f.Contrast, 0, 200)
	f.Brightness = clamp(f.Brightness, -100, 100)
	f.Highlights = clamp(f.Highlights, -100, 100)
	f.Shadows = clamp(f.Shadows, -100, 100)
	f.Vignette = clamp(f.Vignette, 0, 100)
	f.Grain = clamp(f.Grain, 0, 100)
	f.Intensity = clamp(f.Intensity, 0, 100)
	f.ShadowTintHue = clamp(f.ShadowTintHue, 0, 360)
	f.ShadowTintAmount = clamp(f.ShadowTintAmount, -100, 100)
	f.HighlightTintHue = clamp(f.HighlightTintHue, 0, 360)
	f.HighlightTintAmount = clamp(f.HighlightTintAmount, -100, 100)
	if f.Category == "" {
		f.Category = "ai"
	}
	return f
}

// ImageDataURL 把图片缩到 maxSide 以内再编码成 data URL，
// 避免整个原图 base64 塞进请求体（视觉模型也不需要那么大分辨率）。
func ImageDataURL(img image.Image, maxSide int) (string, error) {
	src := img
	b := img.Bounds()
	if b.Dx() > maxSide || b.Dy() > maxSide {
		src = imaging.Fit(img, maxSide, maxSide, imaging.Lanczos)
	}
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, src, &jpeg.Options{Quality: 80}); err != nil {
		return "", err
	}
	return "data:image/jpeg;base64," + base64.StdEncoding.EncodeToString(buf.Bytes()), nil
}

// LoadDotEnv 启动时读一下项目根目录的 .env，方便本地放 DEEPSEEK_API_KEY。
// 已存在的环境变量优先，不被文件覆盖。
func LoadDotEnv(path string) {
	data, err := os.ReadFile(path)
	if err != nil {
		return
	}
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		key, value, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		key = strings.TrimSpace(key)
		value = strings.Trim(strings.TrimSpace(value), `"'`)
		if key == "" {
			continue
		}
		if _, exists := os.LookupEnv(key); exists {
			continue
		}
		os.Setenv(key, value)
	}
}
