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
	"fmt"
	"image"
	"image/jpeg"
	"image/png"
	"log"
	"net/http"
	"strconv"

	"ai-photo-tool/analyzer"
	"ai-photo-tool/filters"

	"github.com/josuedeavila/rmbg"
)

// 全局抠图器：fast = 轻量版（快速），hd = 完整版（高清）
var removerFast *rmbg.RemBG
var removerHD *rmbg.RemBG

// 全局滤镜列表（启动时加载一次）
var presetFilters = filters.Presets()

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
	// 加载两个抠图模型：轻量版 + 完整版
	removerFast = initRemover("models/u2netp.onnx")
	// removerHD = initRemover("models/u2net.onnx") // 高清模型太大，暂时注释
	removerHD = nil

	if removerFast == nil && removerHD == nil {
		fmt.Println("ℹ️  AI 抠图未启用（无可用模型）")
	} else if removerHD != nil {
		fmt.Println("✅  AI 抠图已就绪（快速版 + 高清版）")
	} else {
		fmt.Println("✅  AI 抠图已就绪（仅快速版）")
	}

	// 路由注册
	// 【JS→Go 差异】Express 里 app.Use 中间件链，Go 是每个 handler 独立注册
	http.HandleFunc("/api/filters", handleListFilters)
	http.HandleFunc("/api/analyze", handleAnalyze)
	http.HandleFunc("/api/apply", handleApply)
	http.HandleFunc("/api/auto-enhance", handleAutoEnhance)
	http.HandleFunc("/api/remove-bg", handleRemoveBg)
	http.Handle("/", http.FileServer(http.Dir("static")))

	addr := ":8080"
	fmt.Printf("自动滤镜服务已启动: http://localhost%s\n", addr)
	fmt.Printf("内置滤镜数量: %d 个\n", len(presetFilters))
	log.Fatal(http.ListenAndServe(addr, nil))
}

// handleListFilters 返回全部预设滤镜列表
func handleListFilters(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(presetFilters)
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
// query 参数：model=fast（轻量快速，默认）/ hd（完整版高清）
func handleRemoveBg(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}

	// 选择模型：默认 fast，hd 不可用时自动回退
	modelType := r.URL.Query().Get("model")
	var remover *rmbg.RemBG
	if modelType == "hd" && removerHD != nil {
		remover = removerHD
	} else {
		remover = removerFast
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

	// 返回 PNG（带透明通道）
	w.Header().Set("Content-Type", "image/png")
	var buf bytes.Buffer
	png.Encode(&buf, result)
	w.Write(buf.Bytes())
}
