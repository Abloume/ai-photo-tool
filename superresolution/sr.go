// Package superresolution 实现 Real-ESRGAN 超分辨率放大。
//
// 用通用 inference.Session 封装，大图按 tile 分块推理（避免一次性喂太大图爆内存）。
// 预处理：RGB/255 → 归一化到 [-1,1] → CHW float32
// 后处理：输出 clamp 到 [0,1] → ×255 → uint8 RGB
//
// 【JS→Go 差异】
// - 浏览器跑超分用 onnxruntime-web + WebGL/WASM，大图要 OffscreenCanvas 分块
// - Go 直接 CPU 推理，tile 分块在内存里切，没有 canvas 那层开销
package superresolution

import (
	"fmt"
	"image"
	"image/color"

	"ai-photo-tool/inference"
)

const (
	// tileSize 分块大小：这个模型固定输入 224×224，输出 896×896（4 倍）
	tileSize = 224
	// scale 放大倍数
	scale = 4
)

// SuperResolution 超分器
type SuperResolution struct {
	session *inference.Session
}

// NewRealESRGAN 加载 RealESRGAN_x4plus onnx 模型。
// 输入节点 input.1（1×3×224×224），输出节点 1895（1×3×896×896）。
func NewRealESRGAN(modelPath string) (*SuperResolution, error) {
	sess, err := inference.NewSession(
		modelPath,
		"input.1", "1895",
		[]int64{1, 3, tileSize, tileSize},
		[]int64{1, 3, tileSize * scale, tileSize * scale},
		4,
	)
	if err != nil {
		return nil, err
	}
	return &SuperResolution{session: sess}, nil
}

// Upscale 输入低分辨率图，返回 4 倍放大的图
func (sr *SuperResolution) Upscale(img image.Image) (*image.RGBA, error) {
	b := img.Bounds()
	srcW, srcH := b.Dx(), b.Dy()
	dstW, dstH := srcW*scale, srcH*scale

	dst := image.NewRGBA(image.Rect(0, 0, dstW, dstH))

	// 按 tile 分块遍历
	for ty := 0; ty < srcH; ty += tileSize {
		for tx := 0; tx < srcW; tx += tileSize {
			// 计算 tile 边界（不超过原图）
			tw := tileSize
			if tx+tw > srcW {
				tw = srcW - tx
			}
			th := tileSize
			if ty+th > srcH {
				th = srcH - ty
			}

			// 构造 tile 的 RGBA（从原图裁出 tw×th 的小块）
			tileImg := image.NewRGBA(image.Rect(0, 0, tw, th))
			for y := 0; y < th; y++ {
				for x := 0; x < tw; x++ {
					r, g, bl, _ := img.At(tx+x, ty+y).RGBA()
					tileImg.SetRGBA(x, y, color.RGBA{
						R: uint8(r >> 8), G: uint8(g >> 8), B: uint8(bl >> 8), A: 255,
					})
				}
			}

			// 推理这个 tile
			out, err := sr.runTile(tileImg)
			if err != nil {
				return nil, fmt.Errorf("tile (%d,%d) 推理失败: %w", tx, ty, err)
			}

			// 把结果贴到 dst 对应位置
			outW, outH := tw*scale, th*scale
			for y := 0; y < outH; y++ {
				for x := 0; x < outW; x++ {
					idx := (y*outW + x) * 3
					dst.SetRGBA(tx*scale+x, ty*scale+y, color.RGBA{
						R: uint8(out[idx] * 255),
						G: uint8(out[idx+1] * 255),
						B: uint8(out[idx+2] * 255),
						A: 255,
					})
				}
			}
		}
	}

	return dst, nil
}

// runTile 对一个 tile 做预处理→推理→后处理，返回 CHW float32 [0,1]
func (sr *SuperResolution) runTile(tile *image.RGBA) ([]float32, error) {
	w, h := tile.Bounds().Dx(), tile.Bounds().Dy()

	// 预处理：RGB/255 → [-1,1] → CHW
	input := make([]float32, 3*w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			c := tile.RGBAAt(x, y)
			input[0*w*h+y*w+x] = (float32(c.R)/255.0 - 0.5) / 0.5
			input[1*w*h+y*w+x] = (float32(c.G)/255.0 - 0.5) / 0.5
			input[2*w*h+y*w+x] = (float32(c.B)/255.0 - 0.5) / 0.5
		}
	}

	out, err := sr.session.Run(input)
	if err != nil {
		return nil, err
	}

	// 后处理：clamp 到 [0,1]
	for i, v := range out {
		if v < 0 {
			out[i] = 0
		}
		if v > 1 {
			out[i] = 1
		}
	}
	return out, nil
}

// Close 释放资源
func (sr *SuperResolution) Close() error {
	if sr == nil || sr.session == nil {
		return nil
	}
	return sr.session.Close()
}
