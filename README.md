# ai-photo-tool

本地运行的 AI 修图小工具：上传图片，自动推荐滤镜、一键智能优化、AI 抠图。
后端用 Go 写，前端是单页 HTML，跑在本地 `:8080`。

## 功能

- **20 个预设滤镜**：胶片 / 日系 / 电影感 / VSCO 四大类，参数化像素调整（色温、色调、饱和度、对比度、高光阴影、暗角、色调分离、胶片颗粒、强度）。
- **智能推荐**：分析图片色彩特征（亮度、饱和度、相关色温、冷暖明暗），按规则推荐 Top3 滤镜，显示在滤镜栏顶部。
- **一键优化**：根据分析结果自动拉曝光、白平衡、对比度。
- **AI 抠图**：基于 U²-Net ONNX 模型，快速版（u2netp）+ 高清版（u2net）双模型，输出透明 PNG。

## 启动

```bash
# macOS：onnxruntime 动态库在 libs/ 下，需要让运行时找到
DYLD_LIBRARY_PATH=libs go run .
# 然后浏览器打开 http://localhost:8080
```

可选参数：

```bash
DYLD_LIBRARY_PATH=libs go run . -addr :9000                  # 换端口
DYLD_LIBRARY_PATH=libs go run . -fast-model models/u2netp.onnx # 指定快速抠图模型
DYLD_LIBRARY_PATH=libs go run . -hd-model models/u2net.onnx     # 指定高清抠图模型
```

> 不带 `DYLD_LIBRARY_PATH=libs` 时，滤镜和优化功能仍可用，但抠图初始化会因找不到 onnxruntime 动态库而 panic。

## 模型与依赖

抠图需要两个 ONNX 模型放到 `models/` 目录：

| 文件 | 说明 |
|---|---|
| `models/u2netp.onnx` | 轻量快速版（约 4MB），启动即加载 |
| `models/u2net.onnx` | 高清完整版（约 170MB），首次调用时延迟加载 |

onnxruntime 动态库放在 `libs/`（macOS 下是 `libonnxruntime.dylib`）。
模型缺失不影响滤镜和优化功能，只是抠图不可用，启动会打印提示。

## 目录结构

```
main.go              # HTTP 服务入口、路由、抠图 handler
filters/filters.go   # 滤镜参数定义 + 像素级 Apply（按行并行）
analyzer/analyzer.go # 图片色彩统计 + 滤镜推荐打分
static/index.html    # 单页前端
models/              # ONNX 模型（不入库）
libs/                # onnxruntime 动态库
scripts/             # 早期 Python 抠图脚本（已被 Go 原生实现取代，留作参考）
```

## 开发说明

- 像素处理（滤镜 Apply、图片 Analyze）按行开 goroutine 并行，`sync.WaitGroup` 等待，无锁写各行结果。
- 胶片颗粒用基于 `(x,y)` 的确定性哈希噪声，并行安全且同图同参数可复现。
- 跑测试：`go test ./...`。
