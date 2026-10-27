// Package matting 实现基于 onnx 的图像抠图（前景/背景分割）。
//
// 用 inference.Session 通用封装，不依赖第三方 rmbg 库。
// 当前支持 u2netp（快速版，320 输入）和 RMBG-2.0（1024 输入）。
//
// 【JS→Go 差异】
// - 浏览器里跑分割模型用 onnxruntime-web，输入是 Uint8ClampedArray
// - Go 里直接 float32 CHW tensor，省一次类型转换
package matting

import (
	"fmt"
	"image"
	"image/color"
	_ "image/jpeg"
	_ "image/png"

	"github.com/disintegration/imaging"

	"ai-photo-tool/inference"
)

// u2net 的归一化参数（ImageNet 标准）
var (
	u2netMean = [3]float32{0.485, 0.456, 0.406}
	u2netStd  = [3]float32{0.229, 0.224, 0.225}
)

// Matting 抠图器
type Matting struct {
	session    *inference.Session
	inputSize  int
	inputName  string
	outputName string
}

// NewU2netp 加载 u2netp 快速抠图模型（320x320 输入）
func NewU2netp(modelPath string) (*Matting, error) {
	const size = 320
	sess, err := inference.NewSession(
		modelPath,
		"input.1",
		"1959",
		[]int64{1, 3, size, size},
		[]int64{1, 1, size, size},
		4,
	)
	if err != nil {
		return nil, err
	}
	return &Matting{session: sess, inputSize: size, inputName: "input.1", outputName: "1959"}, nil
}

// NewRMBG2 加载 RMBG-2.0 高精度抠图模型（1024x1024 输入）
// 输入节点 pixel_values [1,3,1024,1024]，输出 alphas [1,1,1024,1024]
func NewRMBG2(modelPath string) (*Matting, error) {
	const size = 1024
	sess, err := inference.NewSession(
		modelPath,
		"pixel_values",
		"alphas",
		[]int64{1, 3, size, size},
		[]int64{1, 1, size, size},
		4,
	)
	if err != nil {
		return nil, err
	}
	return &Matting{session: sess, inputSize: size, inputName: "pixel_values", outputName: "alphas"}, nil
}

// RemoveBackground 输入原图，输出带透明背景的 RGBA 图
func (m *Matting) RemoveBackground(img image.Image) (*image.RGBA, error) {
	mask, err := m.predictMask(img)
	if err != nil {
		return nil, err
	}

	// mask resize 回原图大小
	w, h := img.Bounds().Dx(), img.Bounds().Dy()
	resized := imaging.Resize(mask, w, h, imaging.Linear)
	fullMask := image.NewGray(resized.Bounds())
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, _, _, _ := resized.At(x, y).RGBA()
			fullMask.SetGray(x, y, color.Gray{Y: uint8(r >> 8)})
		}
	}

	// alpha 合成：mask 越亮越保留前景
	// 加阈值：边缘压硬，减少半透明噪点
	dst := image.NewRGBA(img.Bounds())
	b := img.Bounds()
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			r, g, bl, _ := img.At(x, y).RGBA()
			a := float64(fullMask.GrayAt(x, y).Y) / 255.0
			// 阈值压缩
			if a < 0.15 {
				a = 0
			} else if a > 0.85 {
				a = 1
			} else {
				a = (a - 0.15) / 0.7
			}
			dst.SetRGBA(x, y, color.RGBA{
				R: uint8(r >> 8),
				G: uint8(g >> 8),
				B: uint8(bl >> 8),
				A: uint8(a * 255),
			})
		}
	}
	return dst, nil
}

// predictMask 预处理→推理→后处理，得到 inputSize×inputSize 的灰度 mask
func (m *Matting) predictMask(img image.Image) (*image.Gray, error) {
	s := m.inputSize
	resized := imaging.Resize(img, s, s, imaging.Linear)
	nrgba := imaging.Clone(resized)
	pix := nrgba.Pix
	stride := nrgba.Stride

	// CHW float32
	input := make([]float32, 3*s*s)
	for y := 0; y < s; y++ {
		row := pix[y*stride : y*stride+s*4]
		for x := 0; x < s; x++ {
			base := x * 4
			input[0*s*s+y*s+x] = (float32(row[base+0])/255.0 - u2netMean[0]) / u2netStd[0]
			input[1*s*s+y*s+x] = (float32(row[base+1])/255.0 - u2netMean[1]) / u2netStd[1]
			input[2*s*s+y*s+x] = (float32(row[base+2])/255.0 - u2netMean[2]) / u2netStd[2]
		}
	}

	out, err := m.session.Run(input)
	if err != nil {
		return nil, fmt.Errorf("抠图推理失败: %w", err)
	}

	// 输出已是 sigmoid 后的概率 [0,1]
	mask := image.NewGray(image.Rect(0, 0, s, s))
	for i, v := range out {
		if v < 0 {
			v = 0
		}
		if v > 1 {
			v = 1
		}
		mask.Pix[i] = uint8(v * 255)
	}
	return mask, nil
}

// RecommendMattingModel 根据图片特征推荐最合适的抠图模型
// 返回：推荐的模型类型（"fast" / "hd" / "v2"）和推荐理由
func RecommendMattingModel(img image.Image) (modelType string, reason string) {
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()

	// 分析边缘区域（四周各 10%）的像素特征
	// 截图/UI 通常有大面积纯色边框（黑/白），真实照片边缘是自然场景
	edgeBand := int(float64(w) * 0.1)
	if edgeBand < 10 {
		edgeBand = 10
	}

	sampleStep := 4
	totalEdgePixels := 0
	solidColorPixels := 0 // 接近纯黑或纯白的像素
	colorVariance := 0.0  // 颜色方差（越大越丰富）

	// 采样边缘像素
	sampleEdge := func(x, y int) {
		r, g, b, _ := img.At(x, y).RGBA()
		rf, gf, bf := float64(r>>8), float64(g>>8), float64(b>>8)
		totalEdgePixels++
		// 判断是否接近纯黑（<30）或纯白（>225）
		if (rf < 30 && gf < 30 && bf < 30) || (rf > 225 && gf > 225 && bf > 225) {
			solidColorPixels++
		}
		// 简单的颜色变化量
		colorVariance += rf + gf + bf
	}

	// 上下边
	for y := 0; y < edgeBand; y += sampleStep {
		for x := 0; x < w; x += sampleStep {
			sampleEdge(x, y)
			sampleEdge(x, h-1-y)
		}
	}
	// 左右边
	for x := 0; x < edgeBand; x += sampleStep {
		for y := edgeBand; y < h-edgeBand; y += sampleStep {
			sampleEdge(x, y)
			sampleEdge(w-1-x, y)
		}
	}

	if totalEdgePixels == 0 {
		return "hd", "图片太小，默认推荐高清抠图"
	}

	solidRatio := float64(solidColorPixels) / float64(totalEdgePixels)

	// 判断逻辑：
	// 1. 边缘纯色占比 > 40% → 可能是截图/UI，用 u2net hd（只保留主体）
	// 2. 边缘纯色占比 < 20% → 可能是真实场景照片，用 RMBG-2.0（主体抠得干净）
	// 3. 中间 → 默认 u2netp（快）

	if solidRatio > 0.4 {
		return "hd", "检测到大面积纯色边框，可能是截图/UI，推荐高清抠图（只保留主体）"
	} else if solidRatio < 0.2 {
		return "v2", "检测到自然场景背景，推荐 RMBG-2.0（主体抠得更干净）"
	} else {
		return "fast", "场景不明显，推荐快速抠图（速度优先）"
	}
}

// Close 释放资源
func (m *Matting) Close() error {
	if m == nil || m.session == nil {
		return nil
	}
	return m.session.Close()
}
