package analyzer

import (
	"fmt"
	"image"
	"math"
	"sort"
)

// Features 面向 AI 建议的图片特征描述。
// Analyze 只给出色彩均值，够做滤镜打分；但要让 AI（或离线规则）说出"哪里该修"，
// 还需要明暗分布、对比度、锐度、主色、肤色/绿植/天空占比这些语义线索。
type Features struct {
	Stats ImageStats `json:"stats"` // 复用 Analyze 的均值/冷暖/饱和判断

	Width  int    `json:"width"`
	Height int    `json:"height"`
	Aspect string `json:"aspect"` // 例如 3:2 / 1:1 / 9:16

	Contrast       float64  `json:"contrast"`       // 亮度标准差 0-128，越大越"硬"
	ShadowClip     float64  `json:"shadowClip"`     // 死黑像素占比 0-1
	HighlightClip  float64  `json:"highlightClip"`  // 高光溢出像素占比 0-1
	Sharpness      float64  `json:"sharpness"`      // 锐度 0-100，越小越糊
	DominantColors []string `json:"dominantColors"` // 主色 hex，最多 3 个
	SkinRatio      float64  `json:"skinRatio"`      // 肤色像素占比 0-1
	GreenRatio     float64  `json:"greenRatio"`     // 绿植像素占比 0-1
	SkyRatio       float64  `json:"skyRatio"`       // 天空/水面（蓝）占比 0-1
}

// Describe 一次性提取分析 + 语义特征。采样步长和 Analyze 保持一致，
// 大图不会因为多算几个指标就变慢一个量级。
func Describe(img image.Image) Features {
	stats := Analyze(img)

	b := img.Bounds()
	w, h := b.Dx(), b.Dy()

	step := 2
	if w*h > 2000000 {
		step = 3
	}
	if w*h > 8000000 {
		step = 5
	}

	var (
		sumLum, sumLumSq float64
		sampled          int
		shadowPix        int
		highlightPix     int
		skinPix          int
		greenPix         int
		skyPix           int
		gradSum          float64
		gradCount        int
	)

	// 主色统计：每通道压到 4bit（16 级），桶数量最多 4096
	colorCount := make(map[uint16]int)
	colorSumR := make(map[uint16]float64)
	colorSumG := make(map[uint16]float64)
	colorSumB := make(map[uint16]float64)

	for y := b.Min.Y; y < b.Max.Y; y += step {
		for x := b.Min.X; x < b.Max.X; x += step {
			r, g, bl, _ := img.At(x, y).RGBA()
			r8 := float64(r >> 8)
			g8 := float64(g >> 8)
			b8 := float64(bl >> 8)

			lum := 0.299*r8 + 0.587*g8 + 0.114*b8
			sumLum += lum
			sumLumSq += lum * lum
			sampled++

			if lum < 20 {
				shadowPix++
			}
			if lum > 240 {
				highlightPix++
			}
			if isSkin(r8, g8, b8) {
				skinPix++
			}
			if isGreenish(r8, g8, b8, lum) {
				greenPix++
			}
			if isSkyish(r8, g8, b8, lum) {
				skyPix++
			}

			key := uint16(r8/16)<<8 | uint16(g8/16)<<4 | uint16(b8/16)
			colorCount[key]++
			colorSumR[key] += r8
			colorSumG[key] += g8
			colorSumB[key] += b8

			// 水平方向相邻采样点的亮度差，近似梯度幅值做锐度估计
			if nx := x + step; nx < b.Max.X {
				nr, ng, nb, _ := img.At(nx, y).RGBA()
				nlum := 0.299*float64(nr>>8) + 0.587*float64(ng>>8) + 0.114*float64(nb>>8)
				gradSum += math.Abs(nlum - lum)
				gradCount++
			}
		}
	}

	if sampled == 0 {
		sampled = 1
	}
	meanLum := sumLum / float64(sampled)
	variance := sumLumSq/float64(sampled) - meanLum*meanLum
	if variance < 0 {
		variance = 0
	}

	// 平均梯度 0-255 尺度；实测清晰照片约 8-20，明显虚焦约 2-5
	meanGrad := 0.0
	if gradCount > 0 {
		meanGrad = gradSum / float64(gradCount)
	}
	sharpness := math.Min(100, meanGrad*6)

	feats := Features{
		Stats:          stats,
		Width:          w,
		Height:         h,
		Aspect:         aspectLabel(w, h),
		Contrast:       math.Sqrt(variance),
		ShadowClip:     float64(shadowPix) / float64(sampled),
		HighlightClip:  float64(highlightPix) / float64(sampled),
		Sharpness:      sharpness,
		DominantColors: topColors(colorCount, colorSumR, colorSumG, colorSumB, 3),
		SkinRatio:      float64(skinPix) / float64(sampled),
		GreenRatio:     float64(greenPix) / float64(sampled),
		SkyRatio:       float64(skyPix) / float64(sampled),
	}
	return feats
}

// isSkin 用 YCbCr 经验区间判肤色，对肤色/暖色墙面会有少量误判，够做建议用
func isSkin(r, g, b float64) bool {
	y := 0.299*r + 0.587*g + 0.114*b
	cb := 128 - 0.168736*r - 0.331264*g + 0.5*b
	cr := 128 + 0.5*r - 0.418688*g - 0.081312*b
	return y > 60 && cb >= 77 && cb <= 127 && cr >= 133 && cr <= 173
}

func isGreenish(r, g, b, lum float64) bool {
	return lum > 30 && g > r+12 && g > b+8
}

func isSkyish(r, g, b, lum float64) bool {
	return lum > 90 && b > r+15 && b > g+5
}

// aspectLabel 把宽高化简成 16:9 / 3:2 这种常见比例，除不尽就给小数比
func aspectLabel(w, h int) string {
	if w <= 0 || h <= 0 {
		return "未知"
	}
	if w == h {
		return "1:1"
	}
	g := gcd(w, h)
	rw, rh := w/g, h/g
	if rw <= 21 && rh <= 21 {
		return fmt.Sprintf("%d:%d", rw, rh)
	}
	ratio := float64(w) / float64(h)
	return fmt.Sprintf("%.2f:1", ratio)
}

func gcd(a, b int) int {
	for b != 0 {
		a, b = b, a%b
	}
	if a == 0 {
		return 1
	}
	return a
}

// topColors 取出现最多的几个色桶，用桶内均值算平均色
func topColors(count map[uint16]int, sumR, sumG, sumB map[uint16]float64, n int) []string {
	type bucket struct {
		key uint16
		cnt int
	}
	list := make([]bucket, 0, len(count))
	for k, c := range count {
		list = append(list, bucket{k, c})
	}
	sort.Slice(list, func(i, j int) bool { return list[i].cnt > list[j].cnt })

	if n > len(list) {
		n = len(list)
	}
	out := make([]string, 0, n)
	for _, b := range list[:n] {
		c := float64(b.cnt)
		if c == 0 {
			continue
		}
		out = append(out, fmt.Sprintf("#%02x%02x%02x",
			int(sumR[b.key]/c), int(sumG[b.key]/c), int(sumB[b.key]/c)))
	}
	return out
}
