// Package inference 封装 onnxruntime_go 的通用推理流程。
//
// 【JS→Go 差异点】
// - JS/前端跑 AI 模型一般用 ONNX Runtime Web（WASM），线程模型是 Web Worker
// - Go 直接链接原生 onnxruntime 动态库，没有 WASM 那层序列化开销
// - session.Run 是同步阻塞调用，并发请求用 mutex 串行化（onnxruntime session 不是 goroutine 安全的）
//
// 这个包只负责：加载模型、喂 tensor、取回 tensor。
// 每个模型的预处理（图→float32）和后处理（float32→图）由具体模型自己写，
// 因为 u2net/RMBG/Real-ESRGAN 的归一化和输出格式都不一样。
package inference

import (
	"fmt"
	"os"
	"runtime"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

// 环境初始化只做一次（onnxruntime_go 要求）
var initOnce sync.Once

// autoSetupLibrary 在初始化环境前，自动在 libs/ 下找 onnxruntime 动态库。
// 不设的话 onnxruntime_go 只找系统路径的 onnxruntime.so，macOS 下必须手动指。
// 候选路径同时覆盖项目根目录和包目录（go test 时 cwd 在包目录）。
func autoSetupLibrary() {
	var names []string
	switch runtime.GOOS {
	case "darwin":
		names = []string{"libonnxruntime.dylib", "libonnxruntime.1.22.0.dylib"}
	case "linux":
		names = []string{"libonnxruntime.so", "libonnxruntime.1.22.0.so"}
	case "windows":
		names = []string{"onnxruntime.dll"}
	}
	dirs := []string{"libs", "../libs", "../../libs"}
	for _, dir := range dirs {
		for _, n := range names {
			p := dir + "/" + n
			if _, err := os.Stat(p); err == nil {
				ort.SetSharedLibraryPath(p)
				return
			}
		}
	}
	// 找不到就算了，交给 InitializeEnvironment 自己报错
}

// Session 封装一个 onnx 模型推理会话（单输入、单输出）
type Session struct {
	sess        *ort.DynamicAdvancedSession
	mu          sync.Mutex
	inputName   string
	outputName  string
	inputShape  []int64
	outputShape []int64
}

// NewSession 加载模型。
// inputName/outputName 是模型图里的输入/输出节点名（用 onnx 工具 Netron 看模型图能查到）。
// inputShape/outputShape 是 [N, C, H, W] 格式。
func NewSession(modelPath, inputName, outputName string, inputShape, outputShape []int64, threads int) (*Session, error) {
	initOnce.Do(func() {
		autoSetupLibrary()
		// onnxruntime 环境只能初始化一次；rmbg 库可能已经初始化过了，重复调会报错，忽略即可
		if err := ort.InitializeEnvironment(); err != nil {
			fmt.Printf("ℹ️  onnxruntime 环境已初始化（跳过重复初始化）: %v\n", err)
		}
	})

	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("创建 session options 失败: %w", err)
	}
	defer opts.Destroy()

	_ = opts.SetIntraOpNumThreads(threads)
	_ = opts.SetGraphOptimizationLevel(ort.GraphOptimizationLevelEnableAll)

	sess, err := ort.NewDynamicAdvancedSession(
		modelPath,
		[]string{inputName},
		[]string{outputName},
		opts,
	)
	if err != nil {
		return nil, fmt.Errorf("加载 onnx 模型 %s 失败: %w", modelPath, err)
	}

	return &Session{
		sess:        sess,
		inputName:   inputName,
		outputName:  outputName,
		inputShape:  inputShape,
		outputShape: outputShape,
	}, nil
}

// Run 喂入预处理好的输入数据，返回推理结果。
// input 长度必须等于 inputShape 的乘积；返回长度等于 outputShape 的乘积。
// 并发安全：内部加锁，session.Run 不是 goroutine 安全的。
func (s *Session) Run(input []float32) ([]float32, error) {
	return s.RunDynamic(input, s.inputShape, s.outputShape)
}

// RunDynamic 每次调用可指定不同的输入/输出 shape（用于动态 shape 模型）。
// input 长度必须等于 inputShape 乘积；返回长度等于 outputShape 乘积。
func (s *Session) RunDynamic(input []float32, inputShape, outputShape []int64) ([]float32, error) {
	inputTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(inputShape...))
	if err != nil {
		return nil, fmt.Errorf("创建输入 tensor 失败: %w", err)
	}
	defer inputTensor.Destroy()
	copy(inputTensor.GetData(), input)

	outputTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(outputShape...))
	if err != nil {
		return nil, fmt.Errorf("创建输出 tensor 失败: %w", err)
	}
	defer outputTensor.Destroy()

	s.mu.Lock()
	err = s.sess.Run(
		[]ort.Value{inputTensor},
		[]ort.Value{outputTensor},
	)
	s.mu.Unlock()
	if err != nil {
		return nil, fmt.Errorf("推理失败: %w", err)
	}

	out := make([]float32, len(outputTensor.GetData()))
	copy(out, outputTensor.GetData())
	return out, nil
}

// Close 释放模型资源
func (s *Session) Close() error {
	if s.sess != nil {
		return s.sess.Destroy()
	}
	return nil
}
