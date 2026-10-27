# ai-photo-tool

本地运行的 AI 修图小工具：上传图片，自动推荐滤镜、一键智能优化、AI 抠图。
后端用 Go 写，前端是单页 HTML，跑在本地 `:8080`。

## 功能

- **20 个预设滤镜**：胶片 / 日系 / 电影感 / VSCO 四大类，参数化像素调整（色温、色调、饱和度、对比度、高光阴影、暗角、色调分离、胶片颗粒、强度）。
- **智能推荐**：分析图片色彩特征（亮度、饱和度、相关色温、冷暖明暗），按规则推荐 Top3 滤镜，显示在滤镜栏顶部。
- **一键优化**：根据分析结果自动拉曝光、白平衡、对比度。
- **AI 抠图**：基于 U²-Net ONNX 模型，快速版（u2netp）+ 高清版（u2net）双模型，输出透明 PNG。

## 启动

### 开发模式（热重载）

```bash
make dev
# 等价于 air：改 .go 自动编译重启后端
# 改 static/index.html 或后端重启后，浏览器自动刷新（前端每 2s 轮询 /api/reload-check）
# 浏览器打开 http://localhost:8080
```

> 首次用需先装 air：`go install github.com/air-verse/air@latest`

### 生产模式（单二进制）

```bash
make build          # 编译到 bin/ai-photo-tool
./bin/ai-photo-tool # 直接跑，不带 Go 环境也能起
```

### 常用命令

| 命令 | 作用 |
|---|---|
| `make dev` | 开发热重载（air） |
| `make build` | 编译生产二进制到 `bin/` |
| `make run` | 编译并直接运行 |
| `make test` | 跑测试 |
| `make clean` | 清理 `tmp/` 和 `bin/` |

换端口等参数直接传给二进制：

```bash
./bin/ai-photo-tool -addr :9000
```

> 程序启动时会自动在 `libs/` 下找 onnxruntime 动态库并注册，不用再手动设 `DYLD_LIBRARY_PATH`。找不到时滤镜/优化仍可用，仅抠图不可用。

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
Makefile             # 开发/生产入口（make dev / build / run / test）
.air.toml           # air 热重载配置
models/              # ONNX 模型（不入库）
libs/                # onnxruntime 动态库
bin/                 # 生产编译产物（不入库）
tmp/                 # air 开发模式编译产物（不入库）
scripts/             # 早期 Python 抠图脚本（已被 Go 原生实现取代，留作参考）
```

## 开发说明

- 像素处理（滤镜 Apply、图片 Analyze）按行开 goroutine 并行，`sync.WaitGroup` 等待，无锁写各行结果。
- 胶片颗粒用基于 `(x,y)` 的确定性哈希噪声，并行安全且同图同参数可复现。
- 跑测试：`go test ./...`。
