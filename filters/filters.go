// Package filters 提供参数化滤镜定义与像素级应用
//
// 【JS→Go 差异点】
// - JS 里 Canvas 处理图片是 ImageData.data 一维 Uint8ClampedArray，每4字节 RGBA
// - Go 里是 image.Image 接口，需要按 (x,y) 逐点调用 At(x,y) 获取 color.Color
// - 性能差异：Go 直接操作内存更快，但 GC 会短暂暂停；JS 有 JIT 优化但单线程阻塞
// - 类型差异：JS 数字统一 number(浮点)，Go 区分 uint8/float64，颜色值需要显式转换
package filters

import (
	"image"
	"image/color"
	_ "image/jpeg"
	"math"
	"sync"
)

// hashNoise 基于像素坐标的确定性伪随机数，返回 [-1, 1]。
//
// 【JS→Go 差异】
// - JS 里加噪点直接 Math.random()，但那样并行/重算结果都不一样，没法复现
// - Go 这里用整数哈希 (x,y) 算出固定噪声：同一像素永远得到同一个值
// - 好处：① 并行写 dst 不需要锁（每个 goroutine 自己算自己的）② 同图同参数输出一致
func hashNoise(x, y int) float64 {
	n := uint32(x)*73856093 ^ uint32(y)*19349663
	n = (n ^ (n >> 13)) * 1274126177
	n = n ^ (n >> 16)
	// n ∈ [0, 2^32)，映射到 [-1, 1)
	return float64(n)/2147483648.0 - 1.0
}

// Filter 滤镜参数定义
// 所有范围值设计为 -100 ~ +100 或 0 ~ 200，便于前端滑块映射
type Filter struct {
	Name        string  `json:"name"`        // 滤镜名称
	Category    string  `json:"category"`    // 分类：film / japanese / cinematic / vsco
	Description string  `json:"description"` // 适用场景描述

	// 基础调整
	Temperature float64 `json:"temperature"` // 色温：-100冷 ~ +100暖
	Tint        float64 `json:"tint"`        // 色调：-100品红 ~ +100绿
	Saturation  float64 `json:"saturation"`  // 饱和度：0=灰度，100=原始，200=极饱和
	Contrast    float64 `json:"contrast"`     // 对比度：0=平，100=原始，200=高对比
	Brightness  float64 `json:"brightness"`   // 亮度：-100暗 ~ +100亮

	// 明暗分区调整
	Highlights  float64 `json:"highlights"`  // 高光压亮：-100压暗 ~ +100提亮
	Shadows     float64 `json:"shadows"`     // 阴影提亮：-100压暗 ~ +100提亮

	// 创意调整
	Vignette    float64 `json:"vignette"`    // 暗角：0无 ~ 100极强
	Grain       float64 `json:"grain"`       // 胶片颗粒：0无 ~ 100强

	// 滤镜强度：0=原图，100=完整滤镜效果
	Intensity float64 `json:"intensity"` // 强度百分比 0-100

	// 色调分离（阴影/高光分别偏色）
	ShadowTintHue    float64 `json:"shadowTintHue"`    // 阴影色相角度 0-360
	ShadowTintAmount float64 `json:"shadowTintAmount"`  // 阴影偏色强度 -100 ~ +100
	HighlightTintHue    float64 `json:"highlightTintHue"`
	HighlightTintAmount float64 `json:"highlightTintAmount"`
}

// Apply 对输入图片应用滤镜，返回处理后的新图片
// 【JS→Go 差异】JS 里直接修改 ImageData 数组，Go 里必须新建 *image.RGBA
// 因为 image.Image 接口本身是只读的，写操作通过 *image.RGBA.Set 完成
func Apply(src image.Image, f Filter) *image.RGBA {
	bounds := src.Bounds()
	dst := image.NewRGBA(bounds)

	w := bounds.Dx()
	h := bounds.Dy()

	// 预计算混合强度（与像素无关，避免每行重复算）
	intensity := f.Intensity
	if intensity == 0 {
		intensity = 100
	}
	intensity /= 100.0
	if intensity < 0 {
		intensity = 0
	}
	if intensity > 1 {
		intensity = 1
	}

	// 【JS→Go 差异】
	// - JS 是单线程，只能一个 for 循环跑完所有像素，大图会卡主线程
	// - Go 开 goroutine：每一行 y 一个 goroutine，写到 dst 不同行互不冲突
	// - sync.WaitGroup 等所有行写完再返回；Go 调度器自动把 goroutine 排到多核
	var wg sync.WaitGroup
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		wg.Add(1)
		go func(y int) {
			defer wg.Done()
			for x := bounds.Min.X; x < bounds.Max.X; x++ {
				r, g, b, a := src.At(x, y).RGBA()
				// RGBA() 返回 0-65535 的值，需要转成 0-255
				origR := float64(r >> 8)
				origG := float64(g >> 8)
				origB := float64(b >> 8)
				a8 := float64(a >> 8)

				// 逐步应用各项调整
				r8, g8, b8 := origR, origG, origB
				r8, g8, b8 = adjustTemperature(r8, g8, b8, f.Temperature)
				r8, g8, b8 = adjustTint(r8, g8, b8, f.Tint)
				r8, g8, b8 = adjustSaturation(r8, g8, b8, f.Saturation)
				r8, g8, b8 = adjustContrast(r8, g8, b8, f.Contrast)
				r8, g8, b8 = adjustBrightness(r8, g8, b8, f.Brightness)
				r8, g8, b8 = adjustHighlightsShadows(r8, g8, b8, f.Highlights, f.Shadows)
				r8, g8, b8 = splitTone(r8, g8, b8, f)
				r8, g8, b8 = applyVignette(r8, g8, b8, x, y, w, h, f.Vignette)

				// 胶片颗粒：在亮度三通道上叠加确定性噪点，不改色相
				// Grain=0 无颗粒，100 最强（±约14 亮度）
				if f.Grain > 0 {
					n := hashNoise(x, y)
					amt := f.Grain / 100.0 * 14
					r8 += n * amt
					g8 += n * amt
					b8 += n * amt
				}

				// 按滤镜强度混合原图和调整结果
				r8 = origR*(1-intensity) + r8*intensity
				g8 = origG*(1-intensity) + g8*intensity
				b8 = origB*(1-intensity) + b8*intensity

				// clamping 到 0-255
				r8 = clamp255(r8)
				g8 = clamp255(g8)
				b8 = clamp255(b8)

				dst.Set(x, y, color.RGBA{
					R: uint8(r8),
					G: uint8(g8),
					B: uint8(b8),
					A: uint8(a8),
				})
			}
		}(y)
	}
	wg.Wait()

	return dst
}

// ---- 各项调整算法 ----

// adjustTemperature 色温调整：暖色加红减蓝，冷色减红加蓝
func adjustTemperature(r, g, b, t float64) (float64, float64, float64) {
	// t 范围 -100 ~ +100
	scale := t / 100.0
	r += scale * 30  // 暖色红通道加
	b -= scale * 30  // 暖色蓝通道减
	return r, g, b
}

// adjustTint 色调调整：偏绿加绿，偏品红减绿
func adjustTint(r, g, b, t float64) (float64, float64, float64) {
	scale := t / 100.0
	g += scale * 20
	return r, g, b
}

// adjustSaturation 饱和度调整：基于灰度值做插值
// 【JS→Go 差异】JS 常用 HSL 转换，这里用更简单的 luminance 加权法
func adjustSaturation(r, g, b, s float64) (float64, float64, float64) {
	// s 范围 0 ~ 200，100为原始
	lum := 0.299*r + 0.587*g + 0.114*b
	scale := s / 100.0
	r = lum + (r-lum)*scale
	g = lum + (g-lum)*scale
	b = lum + (b-lum)*scale
	return r, g, b
}

// adjustContrast 对比度调整：以 128 为中点拉伸
func adjustContrast(r, g, b, c float64) (float64, float64, float64) {
	// c 范围 0 ~ 200
	scale := c / 100.0
	r = 128 + (r-128)*scale
	g = 128 + (g-128)*scale
	b = 128 + (b-128)*scale
	return r, g, b
}

// adjustBrightness 亮度调整
func adjustBrightness(r, g, b, br float64) (float64, float64, float64) {
	offset := br / 100.0 * 50
	return r + offset, g + offset, b + offset
}

// adjustHighlightsShadows 高光/阴影分区调整
// 简单实现：亮度 > 180 的算高光，< 70 的算阴影
func adjustHighlightsShadows(r, g, b, hi, sh float64) (float64, float64, float64) {
	lum := 0.299*r + 0.587*g + 0.114*b

	// 高光权重：亮度越高权重越大
	hiWeight := math.Max(0, (lum-150)/105) // 150以上开始生效
	hiOffset := hi / 100.0 * 40 * hiWeight

	// 阴影权重：亮度越低权重越大
	shWeight := math.Max(0, (80-lum)/80) // 80以下开始生效
	shOffset := sh / 100.0 * 40 * shWeight

	offset := hiOffset + shOffset
	return r + offset, g + offset, b + offset
}

// splitTone 色调分离：阴影偏一个色，高光偏另一个色
func splitTone(r, g, b float64, f Filter) (float64, float64, float64) {
	lum := 0.299*r + 0.587*g + 0.114*b

	// 阴影偏色
	if f.ShadowTintAmount != 0 {
		shWeight := math.Max(0, (80-lum)/80)
		sr, sg, sb := hueToRGB(f.ShadowTintHue)
		amt := f.ShadowTintAmount / 100.0 * 25 * shWeight
		r += sr * amt / 255
		g += sg * amt / 255
		b += sb * amt / 255
	}

	// 高光偏色
	if f.HighlightTintAmount != 0 {
		hiWeight := math.Max(0, (lum-175)/80)
		hr, hg, hb := hueToRGB(f.HighlightTintHue)
		amt := f.HighlightTintAmount / 100.0 * 25 * hiWeight
		r += hr * amt / 255
		g += hg * amt / 255
		b += hb * amt / 255
	}

	return r, g, b
}

// applyVignette 暗角：四周压暗
func applyVignette(r, g, b float64, x, y, w, h int, strength float64) (float64, float64, float64) {
	if strength == 0 {
		return r, g, b
	}

	// 计算到中心的距离（归一化 0-1）
	cx := float64(w) / 2.0
	cy := float64(h) / 2.0
	dx := (float64(x) - cx) / cx
	dy := (float64(y) - cy) / cy
	dist := math.Sqrt(dx*dx + dy*dy)

	// 距离越远越暗
	factor := 1.0 - (strength/100.0)*math.Max(0, dist-0.5)*1.2
	return r * factor, g * factor, b * factor
}

// cloud 定义一朵云的位置和大小（归一化坐标 0-1）
// cloudBlob 一朵积云由多个小的云瓣叠加而成，形状更自然
//
// TODO(废弃)：天空替换曾尝试用这套云端点做"上部画云"，但纯启发式效果太差
// （阴天/逆光/建筑边缘全翻车），已从功能中移除，数据先留着。
// 后续若重启，应接专用 sky-segmentation ONNX 模型做 mask，不要再用规则判定。
type cloudBlob struct {
	cx, cy  float64 // 中心位置（归一化 0-1）
	rx, ry  float64 // 水平/垂直半径
	opacity float64 // 不透明度
}

// 预定义的云朵布局：每朵云由多层云瓣垂直叠加，形成有厚度的积云
// 云都集中在图片上部 5%-22% 的区域，避免压到地面景物
//
// 注意：当前没有任何代码引用 presetClouds，仅作废弃方案存档。
var presetClouds = []cloudBlob{
	// 第一朵云（左上方）— 上下3层，有厚度
	{cx: 0.12, cy: 0.06, rx: 0.05, ry: 0.015, opacity: 0.4},
	{cx: 0.16, cy: 0.05, rx: 0.07, ry: 0.02, opacity: 0.55},
	{cx: 0.2, cy: 0.065, rx: 0.05, ry: 0.015, opacity: 0.45},
	{cx: 0.16, cy: 0.075, rx: 0.09, ry: 0.012, opacity: 0.3},

	// 第二朵云（中上）— 4层，最大最饱满
	{cx: 0.42, cy: 0.03, rx: 0.05, ry: 0.015, opacity: 0.45},
	{cx: 0.48, cy: 0.02, rx: 0.08, ry: 0.022, opacity: 0.65},
	{cx: 0.54, cy: 0.03, rx: 0.07, ry: 0.018, opacity: 0.55},
	{cx: 0.48, cy: 0.045, rx: 0.1, ry: 0.015, opacity: 0.4},
	{cx: 0.4, cy: 0.038, rx: 0.05, ry: 0.012, opacity: 0.35},

	// 第三朵云（右上方）— 3层
	{cx: 0.78, cy: 0.07, rx: 0.04, ry: 0.012, opacity: 0.35},
	{cx: 0.83, cy: 0.06, rx: 0.06, ry: 0.018, opacity: 0.5},
	{cx: 0.87, cy: 0.075, rx: 0.04, ry: 0.012, opacity: 0.35},

	// 第四朵小云（中间）— 2层
	{cx: 0.3, cy: 0.11, rx: 0.04, ry: 0.012, opacity: 0.3},
	{cx: 0.34, cy: 0.12, rx: 0.05, ry: 0.015, opacity: 0.35},
}


// hueToRGB 色相角度转 RGB（饱和度100%，亮度50%）
func hueToRGB(hue float64) (float64, float64, float64) {
	h := math.Mod(hue, 360) / 60.0
	c := 255.0
	x := c * (1 - math.Abs(math.Mod(h, 2)-1))

	switch {
	case h < 1:
		return c, x, 0
	case h < 2:
		return x, c, 0
	case h < 3:
		return 0, c, x
	case h < 4:
		return 0, x, c
	case h < 5:
		return x, 0, c
	default:
		return c, 0, x
	}
}

func clamp255(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return v
}

// ---- 预设滤镜库 ----

// Presets 返回全部预设滤镜
func Presets() []Filter {
	return []Filter{
		// ===== 胶片复古风 =====
		{
			Name: "Kodak Portra 400", Category: "film", Description: "经典人像胶卷，柔和暖调，肤色自然",
			Temperature: 15, Tint: 5, Saturation: 90, Contrast: 90, Brightness: 5,
			Highlights: -10, Shadows: 20, Vignette: 15, Grain: 30,
			ShadowTintHue: 30, ShadowTintAmount: 10,
			HighlightTintHue: 45, HighlightTintAmount: 8,
		},
		{
			Name: "Kodak Ektar 100", Category: "film", Description: "高饱和风光胶卷，色彩浓郁鲜亮",
			Temperature: 5, Tint: 0, Saturation: 130, Contrast: 110, Brightness: 0,
			Highlights: -5, Shadows: 5, Vignette: 10, Grain: 20,
		},
		{
			Name: "Fuji Pro 400H", Category: "film", Description: "富士人像卷，清透冷调，淡雅肤色",
			Temperature: -10, Tint: -5, Saturation: 85, Contrast: 85, Brightness: 8,
			Highlights: 0, Shadows: 15, Vignette: 10, Grain: 25,
			ShadowTintHue: 200, ShadowTintAmount: 8,
		},
		{
			Name: "Fuji Velvia 50", Category: "film", Description: "富士反转片，极致饱和，风光利器",
			Temperature: 0, Tint: 0, Saturation: 150, Contrast: 120, Brightness: -3,
			Highlights: -15, Shadows: 0, Vignette: 20, Grain: 15,
		},
		{
			Name: "Cinestill 800T", Category: "film", Description: "电影胶片，夜晚暖光，红蓝对比强",
			Temperature: 20, Tint: -15, Saturation: 110, Contrast: 115, Brightness: -5,
			Highlights: -20, Shadows: 10, Vignette: 30, Grain: 40,
			ShadowTintHue: 210, ShadowTintAmount: 15,
			HighlightTintHue: 30, HighlightTintAmount: 12,
		},

		// ===== 日系小清新 =====
		{
			Name: "日系·青空", Category: "japanese", Description: "高明度低饱和，冷调清透",
			Temperature: -15, Tint: 5, Saturation: 70, Contrast: 80, Brightness: 18,
			Highlights: 10, Shadows: 25, Vignette: 0, Grain: 10,
			ShadowTintHue: 190, ShadowTintAmount: 10,
		},
		{
			Name: "日系·白桃", Category: "japanese", Description: "暖粉调，人像肤色柔和粉嫩",
			Temperature: 10, Tint: -10, Saturation: 75, Contrast: 75, Brightness: 15,
			Highlights: 5, Shadows: 20, Vignette: 0, Grain: 8,
			ShadowTintHue: 330, ShadowTintAmount: 12,
		},
		{
			Name: "日系·森林", Category: "japanese", Description: "低饱和绿调，自然治愈感",
			Temperature: -5, Tint: 15, Saturation: 65, Contrast: 80, Brightness: 12,
			Highlights: 0, Shadows: 15, Vignette: 5, Grain: 10,
			ShadowTintHue: 120, ShadowTintAmount: 10,
		},
		{
			Name: "日系·雾感", Category: "japanese", Description: "胶片灰雾感，朦胧柔焦",
			Temperature: -5, Tint: 0, Saturation: 60, Contrast: 65, Brightness: 20,
			Highlights: 15, Shadows: 30, Vignette: 0, Grain: 20,
		},
		{
			Name: "日系·樱花", Category: "japanese", Description: "粉白柔焦，春季人像专用",
			Temperature: 8, Tint: -15, Saturation: 72, Contrast: 70, Brightness: 22,
			Highlights: 10, Shadows: 25, Vignette: 0, Grain: 8,
			ShadowTintHue: 340, ShadowTintAmount: 15,
		},

		// ===== 电影感调色 =====
		{
			Name: "Teal & Orange", Category: "cinematic", Description: "经典橙青对比，好莱坞大片感",
			Temperature: 5, Tint: 0, Saturation: 110, Contrast: 125, Brightness: -3,
			Highlights: -10, Shadows: -5, Vignette: 25, Grain: 15,
			ShadowTintHue: 190, ShadowTintAmount: 25,
			HighlightTintHue: 30, HighlightTintAmount: 20,
		},
		{
			Name: "暗夜·黑调", Category: "cinematic", Description: "低-key暗调，高对比悬疑感",
			Temperature: -5, Tint: 0, Saturation: 90, Contrast: 135, Brightness: -15,
			Highlights: -25, Shadows: -15, Vignette: 40, Grain: 20,
			ShadowTintHue: 220, ShadowTintAmount: 15,
		},
		{
			Name: "暖金·黄昏", Category: "cinematic", Description: "黄金小时暖调，逆光人像",
			Temperature: 30, Tint: 5, Saturation: 105, Contrast: 110, Brightness: 5,
			Highlights: 5, Shadows: 10, Vignette: 20, Grain: 15,
			HighlightTintHue: 40, HighlightTintAmount: 20,
		},
		{
			Name: "冷蓝·雨夜", Category: "cinematic", Description: "冷蓝调，夜景城市孤独感",
			Temperature: -25, Tint: 0, Saturation: 85, Contrast: 120, Brightness: -8,
			Highlights: -10, Shadows: -5, Vignette: 35, Grain: 25,
			ShadowTintHue: 210, ShadowTintAmount: 20,
		},
		{
			Name: "复古·80s", Category: "cinematic", Description: "80年代港风，红绿撞色暖暗",
			Temperature: 15, Tint: -20, Saturation: 115, Contrast: 115, Brightness: -5,
			Highlights: -15, Shadows: 5, Vignette: 30, Grain: 35,
			ShadowTintHue: 350, ShadowTintAmount: 15,
			HighlightTintHue: 50, HighlightTintAmount: 10,
		},

		// ===== VSCO 经典系列 =====
		{
			Name: "VSCO A6", Category: "vsco", Description: "柔和低对比，复古胶片日常",
			Temperature: 8, Tint: 3, Saturation: 80, Contrast: 82, Brightness: 5,
			Highlights: -5, Shadows: 18, Vignette: 12, Grain: 20,
		},
		{
			Name: "VSCO C1", Category: "vsco", Description: "明快清新，旅行街拍热门",
			Temperature: -3, Tint: 2, Saturation: 108, Contrast: 105, Brightness: 5,
			Highlights: 5, Shadows: 10, Vignette: 8, Grain: 10,
		},
		{
			Name: "VSCO M5", Category: "vsco", Description: "静物美食专用，浓郁质感",
			Temperature: 10, Tint: -5, Saturation: 120, Contrast: 115, Brightness: -2,
			Highlights: -10, Shadows: 8, Vignette: 15, Grain: 12,
		},
		{
			Name: "VSCO HB1", Category: "vsco", Description: "黑白灰阶，高级黑白摄影",
			Temperature: 0, Tint: 0, Saturation: 0, Contrast: 125, Brightness: 0,
			Highlights: -15, Shadows: -10, Vignette: 20, Grain: 25,
		},
		{
			Name: "VSCO F2", Category: "vsco", Description: "家用胶片感，生活记录风",
			Temperature: 12, Tint: 5, Saturation: 88, Contrast: 88, Brightness: 8,
			Highlights: 0, Shadows: 15, Vignette: 10, Grain: 30,
		},
	}
}
