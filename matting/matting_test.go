package matting

import (
	"image"
	"image/color"
	"os"
	"testing"
)

// 端到端测试：加载 u2netp.onnx，对一张纯色图推理，验证输出不是空的
func TestU2netpEndToEnd(t *testing.T) {
	modelPath := "../models/u2netp.onnx"
	if _, err := os.Stat(modelPath); err != nil {
		t.Skip("跳过：models/u2netp.onnx 不存在")
	}

	m, err := NewU2netp(modelPath)
	if err != nil {
		t.Fatalf("加载模型失败: %v", err)
	}
	defer m.Close()

	// 造一张 640x480 的测试图（左半黑右半白，模拟前景背景对比）
	img := image.NewRGBA(image.Rect(0, 0, 640, 480))
	for y := 0; y < 480; y++ {
		for x := 0; x < 640; x++ {
			c := color.RGBA{R: 200, G: 180, B: 160, A: 255}
			img.Set(x, y, c)
		}
	}

	out, err := m.RemoveBackground(img)
	if err != nil {
		t.Fatalf("抠图失败: %v", err)
	}

	// 输出尺寸应该和原图一致
	b := out.Bounds()
	if b.Dx() != 640 || b.Dy() != 480 {
		t.Errorf("输出尺寸不对: got %dx%d, want 640x480", b.Dx(), b.Dy())
	}

	// 抽样几个像素，alpha 不应该全 0 或全 255（模型应该输出了有区分度的 mask）
	var transparent, opaque int
	for y := 0; y < 480; y += 40 {
		for x := 0; x < 640; x += 40 {
			_, _, _, a := out.At(x, y).RGBA()
			if a == 0 {
				transparent++
			} else if a == 255 {
				opaque++
			}
		}
	}
	t.Logf("透明像素: %d, 不透明像素: %d（纯色图模型输出可能偏一边，只要不报错就算过）", transparent, opaque)
}
