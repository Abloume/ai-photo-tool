// Package autocompose 实现自动构图/裁剪。
//
// 简单版：按黄金比例裁剪，保留中心区域。
// 不依赖 AI 模型，纯规则算法。
package autocompose

import (
	"image"

	"github.com/disintegration/imaging"
)

// 黄金比例 1:1.618
const goldenRatio = 1.618

// AutoCrop 自动裁剪图片。
// 默认按黄金比例裁剪，保留中心区域。
// ratio: 裁剪比例（宽/高），0 表示自动按黄金比例
func AutoCrop(img image.Image, ratio float64) image.Image {
	b := img.Bounds()
	srcW := b.Dx()
	srcH := b.Dy()

	if ratio <= 0 {
		// 自动：按黄金比例，长边不变，短边按比例裁
		if srcW > srcH {
			ratio = goldenRatio // 宽 > 高，裁成更宽的比例
		} else {
			ratio = 1.0 / goldenRatio // 高 > 宽，裁成更窄的比例
		}
	}

	// 计算裁剪后的尺寸
	var cropW, cropH int
	if float64(srcW)/float64(srcH) > ratio {
		// 原图更宽，按高度算宽度
		cropH = srcH
		cropW = int(float64(srcH) * ratio)
	} else {
		// 原图更高，按宽度算高度
		cropW = srcW
		cropH = int(float64(srcW) / ratio)
	}

	// 居中裁剪
	x0 := (srcW - cropW) / 2
	y0 := (srcH - cropH) / 2

	// 确保不越界
	if x0 < 0 {
		x0 = 0
	}
	if y0 < 0 {
		y0 = 0
	}

	cropRect := image.Rect(x0, y0, x0+cropW, y0+cropH)
	return imaging.Clone(imaging.Crop(img, cropRect))
}

// SmartCrop 智能裁剪（简化版）：
// 保留中心 80% 区域，裁掉边缘 20%
// 适合"主体在中间"的照片
func SmartCrop(img image.Image) image.Image {
	b := img.Bounds()
	srcW := b.Dx()
	srcH := b.Dy()

	// 保留中心 80%
	marginW := int(float64(srcW) * 0.1)
	marginH := int(float64(srcH) * 0.1)

	cropRect := image.Rect(marginW, marginH, srcW-marginW, srcH-marginH)
	return imaging.Clone(imaging.Crop(img, cropRect))
}
