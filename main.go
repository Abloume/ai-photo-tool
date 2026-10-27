// Package main 是自动滤镜 Web 服务入口
//
// 【JS→Go 差异点】
// - Node.js (Express) 写路由是 app.get/post，Go 标准库 net/http 是 HandleFunc
// - Node 异步回调/async-await，Go 是同步 goroutine 模型，每个请求一个 goroutine
// - Node 处理文件读写是 fs 模块回调，Go 是 os 包同步函数
// - 静态文件：Express 用 express.static，Go 用 http.FileServer
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"log"
	"net/http"
	"os"
	"runtime"
	"strconv"
	"time"

	"ai-photo-tool/analyzer"
	"ai-photo-tool/autocompose"
	"ai-photo-tool/filters"
	"ai-photo-tool/matting"
	"ai-photo-tool/superresolution"

	"github.com/josuedeavila/rmbg"
	ort "github.com/yalue/onnxruntime_go"
)

// BackgroundRemover 统一抠图接口：v1（rmbg 库）和 v2（matting/RMBG-2.0）都实现它。
// 这样 handler 不用关心底层是哪个版本，切换版本只改启动时的选择逻辑。
type BackgroundRemover interface {
	RemoveBackground(image.Image) (image.Image, error)
}

// v2Adapter 把 matting.Matting（返回 *image.RGBA）适配成 BackgroundRemover（返回 image.Image）。
// 【JS→Go 差异】Go 没有结构类型推断，返回值类型不同就不满足接口，得手动包一层。
type v2Adapter struct {
	m *matting.Matting
}

func (a *v2Adapter) RemoveBackground(img image.Image) (image.Image, error) {
	return a.m.RemoveBackground(img)
}

// v1 抠图器（旧版 rmbg 库，保留不删）
var removerFast *rmbg.RemBG
var removerHD *rmbg.RemBG

// v2 抠图器（新版 matting 包，RMBG-2.0）
var removerV2 *matting.Matting

// 当前实际生效的抠图器（启动时选 v2，失败回退 v1）
var activeRemover BackgroundRemover

// 超分器（Real-ESRGAN）
var sr *superresolution.SuperResolution

// 全局滤镜列表（启动时加载一次）
var presetFilters = filters.Presets()

// 服务启动时间：air 重新编译重启后会变，前端靠它判断要不要刷新页面
var startupTime = time.Now().Unix()

// 命令行参数：端口和模型路径可配，不用改代码重新编译
// 【JS→Go 差异】Node 服务常用 process.env 读环境变量，Go 标准库用 flag 包
var (
	addrFlag      = flag.String("addr", ":8080", "服务监听地址，例如 :8080 或 127.0.0.1:9000")
	fastModelFlag = flag.String("fast-model", "models/u2netp.onnx", "快速抠图 ONNX 模型路径")
	hdModelFlag   = flag.String("hd-model", "models/u2net.onnx", "高清抠图 ONNX 模型路径")
)

// initORTLib 在第一次初始化抠图器之前，显式指定 onnxruntime 动态库路径。
// 不设置的话 onnxruntime_go 只会去系统路径找，macOS 下必须靠 DYLD_LIBRARY_PATH。
// 【JS→Go 差异】Node 的原生扩展(.node)放哪 require 哪；Go 的 cgo 动态库得在初始化前手动注册路径
func initORTLib() {
	var candidates []string
	switch runtime.GOOS {
	case "darwin":
		candidates = []string{"libs/libonnxruntime.dylib", "libs/libonnxruntime.1.22.0.dylib"}
	case "linux":
		candidates = []string{"libs/libonnxruntime.so", "libs/libonnxruntime.1.22.0.so"}
	case "windows":
		candidates = []string{"libs/onnxruntime.dll"}
	}
	for _, p := range candidates {
		if _, err := os.Stat(p); err == nil {
			ort.SetSharedLibraryPath(p)
			fmt.Printf("✅  ONNX 动态库: %s\n", p)
			return
		}
	}
	fmt.Println("ℹ️  未在 libs/ 找到 onnxruntime 动态库，回退系统路径（可用 DYLD_LIBRARY_PATH 指定）")
}

func initRemover(modelPath string) *rmbg.RemBG {
	r, err := rmbg.New(&rmbg.Config{
		ModelPath:         modelPath,
		IntraOpNumThreads: 4,
	})
	if err != nil {
		fmt.Printf("⚠️  抠图模型加载失败 %s: %v\n", modelPath, err)
		return nil
	}
	fmt.Printf("✅  抠图模型已加载: %s\n", modelPath)
	return r
}

func main() {
	flag.Parse()

	// 必须在任何 rmbg.New() 之前调用，否则 ORT 环境已按系统路径初始化完就晚了
	initORTLib()

	// 加载 v1 抠图模型（旧版 rmbg 库，保留不删，作为默认和 fallback）
	removerFast = initRemover(*fastModelFlag)
	removerHD = initRemover(*hdModelFlag)

	// 加载 v2 抠图模型（新版 RMBG-2.0）
	// 目前 v2 效果待调优，默认不启用；需 ?model=v2 才走
	removerV2Instance, v2Err := matting.NewRMBG2("models/rmbg-2.0.onnx")
	if v2Err == nil {
		removerV2 = removerV2Instance
		fmt.Println("ℹ️  RMBG-2.0 已加载（需 ?model=v2 启用，默认走 v1）")
	} else {
		fmt.Printf("ℹ️  RMBG-2.0 加载失败（%v）\n", v2Err)
	}

	// 默认 activeRemover 指向 v1（优先 hd，没有就 fast）
	if removerHD != nil {
		activeRemover = removerHD
	} else if removerFast != nil {
		activeRemover = removerFast
	}

	// 加载 Real-ESRGAN 超分模型（失败不阻塞启动，只是超分不可用）
	srInstance, err := superresolution.NewRealESRGAN("models/realesrgan-x4.onnx")
	if err != nil {
		fmt.Printf("⚠️  超分模型加载失败: %v\n", err)
		sr = nil
	} else {
		sr = srInstance
		fmt.Println("✅  Real-ESRGAN 超分已就绪（4 倍放大）")
	}

	if activeRemover == nil {
		fmt.Println("ℹ️  AI 抠图未启用（无可用模型）")
	} else {
		fmt.Println("✅  AI 抠图已就绪")
	}

	// 路由注册
	// 【JS→Go 差异】Express 里 app.Use 中间件链，Go 是每个 handler 独立注册
	http.HandleFunc("/api/filters", handleListFilters)
	http.HandleFunc("/api/analyze", handleAnalyze)
	http.HandleFunc("/api/apply", handleApply)
	http.HandleFunc("/api/auto-enhance", handleAutoEnhance)
	http.HandleFunc("/api/remove-bg", handleRemoveBg)
	http.HandleFunc("/api/recommend-matting", handleRecommendMatting)
	http.HandleFunc("/api/auto-crop", handleAutoCrop)
	http.HandleFunc("/api/change-bg", handleChangeBg)
	http.HandleFunc("/api/super-resolve", handleSuperResolve)
	http.HandleFunc("/api/reload-check", handleReloadCheck)
	http.Handle("/", http.FileServer(http.Dir("static")))

	fmt.Printf("自动滤镜服务已启动: http://localhost%s\n", *addrFlag)
	fmt.Printf("内置滤镜数量: %d 个\n", len(presetFilters))
	log.Fatal(http.ListenAndServe(*addrFlag, nil))
}

// handleListFilters 返回全部预设滤镜列表
func handleListFilters(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(presetFilters)
}

// handleReloadCheck 开发模式下给前端轮询用：
// 返回服务启动时间 + index.html 的修改时间，前端对比发现变了就自动刷新页面。
// 改 .go 触发 air 重启 → startup 变；改 index.html → htmlMtime 变。
func handleReloadCheck(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	mtime := int64(0)
	if fi, err := os.Stat("static/index.html"); err == nil {
		mtime = fi.ModTime().Unix()
	}
	json.NewEncoder(w).Encode(map[string]int64{
		"startup":   startupTime,
		"htmlMtime": mtime,
	})
}

// analyzeResponse 分析接口返回结构
type analyzeResponse struct {
	Stats     analyzer.ImageStats `json:"stats"`
	Recommend []filters.Filter    `json:"recommend"`
}

// handleAnalyze 上传图片，分析色彩特征并返回推荐滤镜
func handleAnalyze(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}

	// 解析上传的图片
	// 【JS→Go 差异】Node 用 multer 中间件，Go 标准库 r.ParseMultipartForm
	file, _, err := r.FormFile("image")
	if err != nil {
		http.Error(w, "读取图片失败: "+err.Error(), http.StatusBadRequest)
		return
	}
	defer file.Close()

	// 解码图片
	// 【JS→Go 差异】JS 用 createImageBitmap / canvas，Go 用 image.Decode 需要注册格式
	img, _, err := image.Decode(file)
	if err != nil {
		http.Error(w, "解码图片失败: "+err.Error(), http.StatusBadRequest)
		return
	}

	// 分析
	stats := analyzer.Analyze(img)
	recommend := analyzer.Recommend(stats, presetFilters, 3)

	resp := analyzeResponse{
		Stats:     stats,
		Recommend: recommend,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// handleApply 应用指定滤镜，返回处理后的图片
// query 参数：index=滤镜序号
func handleApply(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}

	// 获取滤镜索引
	idxStr := r.URL.Query().Get("index")
	idx, err := strconv.Atoi(idxStr)
	if err != nil || idx < 0 || idx >= len(presetFilters) {
		http.Error(w, "无效的滤镜序号", http.StatusBadRequest)
		return
	}
	f := presetFilters[idx]

	// 读取滤镜强度参数（可选，默认 100）
	intensityStr := r.URL.Query().Get("intensity")
	if intensityStr != "" {
		intensity, err := strconv.ParseFloat(intensityStr, 64)
		if err == nil {
			f.Intensity = intensity
		}
	}

	// 解析上传的图片
	file, _, err := r.FormFile("image")
	if err != nil {
		http.Error(w, "读取图片失败: "+err.Error(), http.StatusBadRequest)
		return
	}
	defer file.Close()

	img, format, err := image.Decode(file)
	if err != nil {
		http.Error(w, "解码图片失败: "+err.Error(), http.StatusBadRequest)
		return
	}

	// 应用滤镜
	result := filters.Apply(img, f)

	// 编码返回
	// 【JS→Go 差异】JS 用 canvas.toBlob，Go 手动编码成 bytes.Buffer
	w.Header().Set("Content-Type", "image/jpeg")
	var buf bytes.Buffer
	// JPEG 质量 85，平衡画质和体积
	jpeg.Encode(&buf, result, &jpeg.Options{Quality: 85})
	w.Write(buf.Bytes())

	_ = format // format 记录下来但不用，因为统一输出 jpeg
	_ = png.Encode // 防止 import 未使用报错（如果以后加 png 输出）
}

// handleAutoEnhance AI 一键优化：自动分析图片并调整到最佳状态
func handleAutoEnhance(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}

	// 解析上传的图片
	file, _, err := r.FormFile("image")
	if err != nil {
		http.Error(w, "读取图片失败: "+err.Error(), http.StatusBadRequest)
		return
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		http.Error(w, "解码图片失败: "+err.Error(), http.StatusBadRequest)
		return
	}

	// 先分析图片色彩特征
	stats := analyzer.Analyze(img)

	// 根据分析结果自动生成优化参数
	autoFilter := analyzer.GenerateAutoFilter(stats)

	// 应用优化
	result := filters.Apply(img, autoFilter)

	// 返回处理后的 JPEG
	w.Header().Set("Content-Type", "image/jpeg")
	var buf bytes.Buffer
	jpeg.Encode(&buf, result, &jpeg.Options{Quality: 85})
	w.Write(buf.Bytes())
}

// handleRemoveBg AI 抠图：上传图片，返回透明背景 PNG
// query 参数：
//   - fast  → v1 u2netp（快速版，默认）
//   - hd    → v1 u2net（高清版，更干净）
//   - v2    → v2 RMBG-2.0（新版，效果待调优）
func handleRemoveBg(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}

	// 选择抠图器
	var remover BackgroundRemover
	modelType := r.URL.Query().Get("model")
	switch modelType {
	case "hd":
		if removerHD != nil {
			remover = removerHD
		} else {
			remover = activeRemover
		}
	case "v2":
		if removerV2 != nil {
			remover = &v2Adapter{m: removerV2}
		} else {
			remover = activeRemover
		}
	default: // fast 或空
		if removerFast != nil {
			remover = removerFast
		} else {
			remover = activeRemover
		}
	}

	if remover == nil {
		http.Error(w, "AI 抠图未初始化，模型加载失败", http.StatusServiceUnavailable)
		return
	}

	// 解析上传的图片
	file, _, err := r.FormFile("image")
	if err != nil {
		http.Error(w, "读取图片失败: "+err.Error(), http.StatusBadRequest)
		return
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		http.Error(w, "解码图片失败: "+err.Error(), http.StatusBadRequest)
		return
	}

	// AI 抠图：自动生成 mask 并应用
	result, err := remover.RemoveBackground(img)
	if err != nil {
		http.Error(w, "抠图失败: "+err.Error(), http.StatusInternalServerError)
		return
	}

	// 确保输出是 RGBA 带透明通道
	// rmbg 库（v1）输出的是白色背景，不是透明，需要手动转成 alpha
	// 用 flood fill：从边缘开始，只把和边缘连通的白色区域转成透明
	// 这样人物身上的白色（不与边缘连通）不会被误判
	rgbaResult := image.NewRGBA(result.Bounds())
	b := result.Bounds()
	imgW, imgH := b.Dx(), b.Dy()

	// 第一步：先把所有像素复制到 RGBA
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			r, g, bl, a := result.At(x, y).RGBA()
			rgbaResult.SetRGBA(x, y, color.RGBA{
				R: uint8(r >> 8), G: uint8(g >> 8), B: uint8(bl >> 8), A: uint8(a >> 8),
			})
		}
	}

	// 第二步：flood fill 从边缘开始，识别白色背景
	// visited 记录哪些像素已经处理过
	visited := make([][]bool, imgH)
	for i := range visited {
		visited[i] = make([]bool, imgW)
	}

	// 判断是否是"白色背景"像素
	isWhiteBg := func(x, y int) bool {
		c := rgbaResult.RGBAAt(x, y)
		// 接近白色的像素才算背景
		return c.R > 235 && c.G > 235 && c.B > 235
	}

	// BFS flood fill 队列
	type point struct{ x, y int }
	queue := []point{}

	// 从四条边的所有白色像素开始
	for x := 0; x < imgW; x++ {
		if isWhiteBg(x, 0) {
			queue = append(queue, point{x, 0})
		}
		if isWhiteBg(x, imgH-1) {
			queue = append(queue, point{x, imgH - 1})
		}
	}
	for y := 0; y < imgH; y++ {
		if isWhiteBg(0, y) {
			queue = append(queue, point{0, y})
		}
		if isWhiteBg(imgW-1, y) {
			queue = append(queue, point{imgW - 1, y})
		}
	}

	// 四个方向
	dirs := []point{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}
	transparentCount := 0

	for len(queue) > 0 {
		p := queue[0]
		queue = queue[1:]
		x, y := p.x, p.y
		if x < 0 || x >= imgW || y < 0 || y >= imgH || visited[y][x] {
			continue
		}
		visited[y][x] = true
		if !isWhiteBg(x, y) {
			continue
		}
		// 标记为透明
		c := rgbaResult.RGBAAt(x, y)
		rgbaResult.SetRGBA(x, y, color.RGBA{R: c.R, G: c.G, B: c.B, A: 0})
		transparentCount++
		// 扩散到邻居
		for _, d := range dirs {
			queue = append(queue, point{x + d.x, y + d.y})
		}
	}

	fmt.Printf("🔍 抠图调试: 总像素=%d, flood fill 透明像素=%d, result类型=%T\n",
		imgW*imgH, transparentCount, result)

	// 返回 PNG（带透明通道）
	w.Header().Set("Content-Type", "image/png")
	var buf bytes.Buffer
	png.Encode(&buf, rgbaResult)
	w.Write(buf.Bytes())
}

// handleRecommendMatting AI 分析图片，推荐最合适的抠图模型
func handleRecommendMatting(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}

	file, _, err := r.FormFile("image")
	if err != nil {
		http.Error(w, "读取图片失败: "+err.Error(), http.StatusBadRequest)
		return
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		http.Error(w, "解码图片失败: "+err.Error(), http.StatusBadRequest)
		return
	}

	modelType, reason := matting.RecommendMattingModel(img)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"model":  modelType,
		"reason": reason,
	})
}

// handleAutoCrop 自动裁剪：按黄金比例/三分法自动构图
func handleAutoCrop(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}

	file, _, err := r.FormFile("image")
	if err != nil {
		http.Error(w, "读取图片失败: "+err.Error(), http.StatusBadRequest)
		return
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		http.Error(w, "解码图片失败: "+err.Error(), http.StatusBadRequest)
		return
	}

	// 按黄金比例自动裁剪
	result := autocompose.AutoCrop(img, 0)

	w.Header().Set("Content-Type", "image/jpeg")
	var buf bytes.Buffer
	jpeg.Encode(&buf, result, &jpeg.Options{Quality: 90})
	w.Write(buf.Bytes())
}

// handleChangeBg 换背景：把抠图后的透明 PNG 填充成指定颜色
// query 参数：color=rrggbb（十六进制颜色，如 ff0000 红色）
func handleChangeBg(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}

	colorHex := r.URL.Query().Get("color")
	if colorHex == "" {
		colorHex = "ffffff" // 默认白色
	}

	// 解析十六进制颜色
	var r8, g8, b8 uint8
	fmt.Sscanf(colorHex, "%02x%02x%02x", &r8, &g8, &b8)

	file, _, err := r.FormFile("image")
	if err != nil {
		http.Error(w, "读取图片失败: "+err.Error(), http.StatusBadRequest)
		return
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		http.Error(w, "解码图片失败: "+err.Error(), http.StatusBadRequest)
		return
	}

	// 遍历像素，把透明区域填充成指定颜色
	b := img.Bounds()
	dst := image.NewRGBA(b)
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			pr, pg, pb, pa := img.At(x, y).RGBA()
			alpha := pa >> 8 // 0-255
			if alpha < 128 {
				// 半透明/透明区域，填充背景色
				// （用 alpha 混合：背景色 * (1-alpha) + 原图 * alpha，过渡更自然）
				bgR := float64(r8) * (1 - float64(alpha)/255)
				bgG := float64(g8) * (1 - float64(alpha)/255)
				bgB := float64(b8) * (1 - float64(alpha)/255)
				fgR := float64(pr>>8) * (float64(alpha) / 255)
				fgG := float64(pg>>8) * (float64(alpha) / 255)
				fgB := float64(pb>>8) * (float64(alpha) / 255)
				dst.SetRGBA(x, y, color.RGBA{
					R: uint8(bgR + fgR),
					G: uint8(bgG + fgG),
					B: uint8(bgB + fgB),
					A: 255,
				})
			} else {
				// 前景区域，保留原图
				dst.SetRGBA(x, y, color.RGBA{
					R: uint8(pr >> 8),
					G: uint8(pg >> 8),
					B: uint8(pb >> 8),
					A: 255,
				})
			}
		}
	}

	w.Header().Set("Content-Type", "image/png")
	var buf bytes.Buffer
	png.Encode(&buf, dst)
	w.Write(buf.Bytes())
}

// handleSuperResolve Real-ESRGAN 超分：上传图片，返回 4 倍放大的图
func handleSuperResolve(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	if sr == nil {
		http.Error(w, "超分模型未初始化", http.StatusServiceUnavailable)
		return
	}

	file, _, err := r.FormFile("image")
	if err != nil {
		http.Error(w, "读取图片失败: "+err.Error(), http.StatusBadRequest)
		return
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		http.Error(w, "解码图片失败: "+err.Error(), http.StatusBadRequest)
		return
	}

	result, err := sr.Upscale(img)
	if err != nil {
		http.Error(w, "超分失败: "+err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "image/jpeg")
	var buf bytes.Buffer
	jpeg.Encode(&buf, result, &jpeg.Options{Quality: 90})
	w.Write(buf.Bytes())
}
