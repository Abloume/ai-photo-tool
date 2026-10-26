// Package analyzer 分析图片色彩特征并自动推荐滤镜
//
// 【JS→Go 差异点】
// - JS 前端分析图片用 canvas.getImageData()，Go 后端直接遍历 image.Image
// - JS 里大量数组操作靠 map/filter/reduce，Go 用 for 循环更直接
// - 色彩空间转换：RGB→XYZ→Lab 公式和 JS 一样，但 Go 的 math 包精度更高
package analyzer

import (
	"image"
	"math"

	"ai-photo-tool/filters"
)

// ImageStats 图片色彩统计特征
type ImageStats struct {
	AvgR        float64 `json:"avgR"`        // 平均红 0-255
	AvgG        float64 `json:"avgG"`        // 平均绿
	AvgB        float64 `json:"avgB"`        // 平均蓝
	AvgLum      float64 `json:"avgLum"`      // 平均亮度 0-255
	Saturation  float64 `json:"saturation"`  // 平均饱和度 0-1
	ColorTemp   float64 `json:"colorTemp"`   // 相关色温 K（1000-12000）
	IsDark      bool    `json:"isDark"`      // 是否偏暗
	IsBright    bool    `json:"isBright"`    // 是否偏亮
	IsWarm      bool    `json:"isWarm"`      // 是否偏暖
	IsCool      bool    `json:"isCool"`      // 是否偏冷
	IsHighSat   bool    `json:"isHighSat"`   // 是否高饱和
	IsLowSat    bool    `json:"isLowSat"`    // 是否低饱和
}

// Analyze 分析图片，返回色彩统计特征
func Analyze(img image.Image) ImageStats {
	bounds := img.Bounds()
	w := bounds.Dx()
	h := bounds.Dy()
	_ = w // 保留宽高变量，后续扩展用
	_ = h

	var sumR, sumG, sumB float64
	var sumSat float64

	// 采样步长：大图不用遍历每个像素，步长采样就够统计用
	// 【JS→Go 差异】JS 处理大图会卡主线程，Go 这里可以轻松做并行
	step := 2
	if w*h > 4000000 { // 400万像素以上加大步长
		step = 4
	}

	count := 0
	for y := bounds.Min.Y; y < bounds.Max.Y; y += step {
		for x := bounds.Min.X; x < bounds.Max.X; x += step {
			r, g, b, _ := img.At(x, y).RGBA()
			r8 := float64(r >> 8)
			g8 := float64(g >> 8)
			b8 := float64(b >> 8)

			sumR += r8
			sumG += g8
			sumB += b8

			// 计算该像素饱和度
			maxC := math.Max(math.Max(r8, g8), b8)
			minC := math.Min(math.Min(r8, g8), b8)
			if maxC > 0 {
				sumSat += (maxC - minC) / maxC
			}
			count++
		}
	}

	avgR := sumR / float64(count)
	avgG := sumG / float64(count)
	avgB := sumB / float64(count)
	avgLum := 0.299*avgR + 0.587*avgG + 0.114*avgB
	avgSat := sumSat / float64(count)

	// 计算相关色温（近似算法）
	// 基于 McCamy 公式的简化版，从 RGB 粗略估算 CCT
	cct := estimateCCT(avgR, avgG, avgB)

	stats := ImageStats{
		AvgR:       avgR,
		AvgG:       avgG,
		AvgB:       avgB,
		AvgLum:     avgLum,
		Saturation: avgSat,
		ColorTemp:  cct,
	}

	// 判断分类标签
	stats.IsDark = avgLum < 100
	stats.IsBright = avgLum > 160
	stats.IsWarm = cct < 5000
	stats.IsCool = cct > 6500
	stats.IsHighSat = avgSat > 0.45
	stats.IsLowSat = avgSat < 0.25

	return stats
}

// estimateCCT 用简化公式从 RGB 估算相关色温
// 参考：McCamy 公式的简化实现，精度足够做推荐用
func estimateCCT(r, g, b float64) float64 {
	// 先转成 0-1
	rn := r / 255.0
	gn := g / 255.0
	bn := b / 255.0

	// RGB → XYZ（sRGB D65）
	x := 0.4124*rn + 0.3576*gn + 0.1805*bn
	y := 0.2126*rn + 0.7152*gn + 0.0722*bn
	z := 0.0193*rn + 0.1192*gn + 0.9505*bn

	// XYZ → xy 色度坐标
	sum := x + y + z
	if sum == 0 {
		return 6500
	}
	xi := x / sum
	yi := y / sum

	// McCamy 公式：CCT = 449*(xi-0.3320)^3 + 3525*(xi-0.3320)^2 + 6823.3*(xi-0.3320) + 5520.33
	// 实际更常用 uv 坐标版，这里用简化的 xy 版够用
	n := (xi - 0.3366) / (yi - 0.1735)
	cct := 449.0*math.Pow(n, 3) + 3525.0*math.Pow(n, 2) + 6823.3*n + 5520.33

	// 限制合理范围
	if cct < 1000 {
		cct = 1000
	}
	if cct > 12000 {
		cct = 12000
	}
	return cct
}

// Recommend 根据图片特征推荐 Top N 滤镜
// 打分逻辑：每个滤镜和图片特征的匹配度打分，取最高分
func Recommend(stats ImageStats, allFilters []filters.Filter, topN int) []filters.Filter {
	type scored struct {
		f     filters.Filter
		score float64
	}

	var results []scored

	for _, f := range allFilters {
		score := scoreMatch(stats, f)
		results = append(results, scored{f, score})
	}

	// 简单冒泡排序（滤镜只有20个，不用堆排序）
	for i := 0; i < len(results); i++ {
		for j := i + 1; j < len(results); j++ {
			if results[j].score > results[i].score {
				results[i], results[j] = results[j], results[i]
			}
		}
	}

	// 取前 topN
	if topN > len(results) {
		topN = len(results)
	}
	top := make([]filters.Filter, topN)
	for i := 0; i < topN; i++ {
		top[i] = results[i].f
	}
	return top
}

// scoreMatch 计算单个滤镜与图片的匹配分
// 规则设计：
// - 暗图推荐暗调/电影感滤镜
// - 亮图推荐日系/清新滤镜
// - 暖图推荐暖调胶片，冷图推荐冷调日系
// - 高饱和图适当降饱和滤镜更耐看，低饱和图推荐加饱和的
func scoreMatch(stats ImageStats, f filters.Filter) float64 {
	score := 50.0 // 基础分

	// 1. 明暗匹配
	if stats.IsDark {
		// 暗图适合对比度高、暗角强的
		score += f.Contrast - 100          // 对比度越高加分
		score += f.Vignette * 0.5         // 暗角加分
		if f.Category == "cinematic" {
			score += 15
		}
	}
	if stats.IsBright {
		// 亮图适合日系、柔和的
		score += (100 - f.Contrast) * 0.5  // 低对比加分
		score += f.Brightness             // 提亮加分
		if f.Category == "japanese" {
			score += 15
		}
	}

	// 2. 冷暖匹配
	if stats.IsWarm {
		// 暖图：暖调胶片加分，冷调日系减分
		if f.Temperature > 0 {
			score += 10
		}
		if f.Category == "film" {
			score += 8
		}
	}
	if stats.IsCool {
		// 冷图：冷调日系加分
		if f.Temperature < 0 {
			score += 10
		}
		if f.Category == "japanese" {
			score += 8
		}
	}

	// 3. 饱和度匹配
	if stats.IsHighSat {
		// 已经很饱和了，推荐稍微降饱和的更耐看
		if f.Saturation < 100 {
			score += 8
		}
	}
	if stats.IsLowSat {
		// 偏灰的图，推荐加饱和的滤镜
		if f.Saturation > 100 {
			score += 8
		}
	}

	// 4. 胶片/颗粒偏好
	// （这里没有用户偏好维度，暂时给胶片类基础加分）
	if f.Category == "film" {
		score += 5
	}

	return score
}

// GenerateAutoFilter 根据图片分析结果，自动生成最佳优化参数
// 相当于"一键智能优化"：自动调曝光、白平衡、对比度、饱和度
func GenerateAutoFilter(stats ImageStats) filters.Filter {
	f := filters.Filter{
		Name:        "AI 自动优化",
		Category:    "auto",
		Description: "根据图片色彩分析自动调整",
		Temperature: 0,
		Tint:        0,
		Saturation: 100,
		Contrast:    100,
		Brightness:  0,
		Highlights:  0,
		Shadows:     0,
		Vignette:    0,
		Grain:       0,
	}

	// 1. 亮度调整：偏暗就提亮，偏亮就压高光
	if stats.IsDark {
		f.Brightness += 15    // 整体提亮
		f.Shadows += 20       // 暗部细节拉回来
	}
	if stats.IsBright {
		f.Highlights -= 15    // 高光压下来
		f.Brightness -= 5     // 整体稍微压暗
	}

	// 2. 白平衡修正：偏冷加暖，偏暖加冷（往中间拉）
	if stats.IsCool {
		f.Temperature += 10   // 往暖拉一点
	}
	if stats.IsWarm {
		f.Temperature -= 10   // 往冷拉一点
	}

	// 3. 饱和度调整：偏灰就加饱和，太艳就降一点
	if stats.IsLowSat {
		f.Saturation += 25    // 偏灰的图加饱和
	}
	if stats.IsHighSat {
		f.Saturation -= 10    // 太艳的稍微压一压，更耐看
	}

	// 4. 对比度：整体偏灰就加一点对比
	// （用亮度范围粗略判断：平均亮度在中间但饱和度低，说明画面发灰）
	if stats.IsLowSat && stats.AvgLum > 80 && stats.AvgLum < 180 {
		f.Contrast += 15
	}

	return f
}
