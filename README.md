# ai-photo-tool

本地运行的 AI 修图小工具：上传图片，自动推荐滤镜、一键智能优化、AI 抠图。
后端用 Go 写，前端是单页 HTML，跑在本地 `:8080`。

## 功能

- **20 个预设滤镜**：胶片 / 日系 / 电影感 / VSCO 四大类，参数化像素调整（色温、色调、饱和度、对比度、高光阴影、暗角、色调分离、胶片颗粒、强度）。
- **AI 修图建议**：导入图片后自动分析画面（DeepSeek `deepseek-flash` 能看图，同时结合明暗分布、对比度、锐度、主色、肤色/天空/绿植占比等本地量化特征），给出 4-6 个优化标签；点标签即显示对应提示词，并在本地实时预览效果。未配置密钥时自动走离线规则。
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

## AI 建议配置（可选）

不配置也能用：`/api/ai-suggest` 会自动退回离线规则（标签和提示词照样生成，只是不如大模型懂画面）。

要接 DeepSeek，复制 `.env.example` 为 `.env` 填上密钥，或直接设环境变量：

| 变量 | 说明 |
|---|---|
| `DEEPSEEK_API_KEY` | 必填（也可以用 `AI_API_KEY`） |
| `DEEPSEEK_BASE_URL` | 默认 `https://api.deepseek.com`，兼容 OpenAI 协议的网关也能填 |
| `DEEPSEEK_MODEL` | 默认 `deepseek-flash`（V4.1-Flash，官方当前唯一支持图像理解的在售模型） |
| `AI_VISION_MODEL` | 可选，给图片请求单独指定模型；不填就用 `DEEPSEEK_MODEL` |
| `AI_THINKING` | 可选，`1` 开启思考模式；默认关闭（默认思考是 high，出几个标签要等几十秒） |

关于看图：`deepseek-flash`（V4.1-Flash）**支持**图片输入，`deepseek-v4-pro` **不支持**，所以默认模型是 flash。请求会把 1024px 长边的 JPEG 缩略图和本地量化特征一起发过去（官方单图 token 上限约 1024，约等于 1300×1300，长边 1024 足够判断场景又不浪费额度）；`content` 必须是块数组，图片只能放在 `user` 消息里。旧模型名 `deepseek-v4-flash-vision-exp` 已下线但请求仍由 Flash 承接。

如果换成不支持看图的模型（或别家纯文本模型），代码会只发量化特征（亮度、对比度、锐度、主色、肤色/天空占比等），建议质量会下降但仍可用。

## AI 出图配置（可选，图生图）

提示词卡片上有个「🎨 AI 出图」按钮：点一下会把**当前参考图 + 提示词**发给 OpenAI 兼容的 `/images/edits`，真出一张重绘后的图，可直接下载。

| 变量 | 说明 |
|---|---|
| `AI_IMAGE_MODEL` | 图像模型，例 `gpt-image-1` / `doubao-seedream-4-0` / `wanx-v1`；不填则该功能关闭，按钮置灰 |
| `AI_IMAGE_BASE_URL` | 默认复用 `DEEPSEEK_BASE_URL`，填到 `/v1` 为止 |
| `AI_IMAGE_API_KEY` | 默认复用 `DEEPSEEK_API_KEY` |
| `AI_IMAGE_ENDPOINT` | 默认 `/images/edits`；DALL·E 这类只支持文生图的模型填 `/images/generations` |
| `AI_IMAGE_SIZE` | 默认 `1024x1024` |

注意：这是**生成式重绘**，会改变画面细节（和 `/api/apply-custom` 的调色完全不同），出图分辨率按模型能力来，不保证与原图一致。网关必须真的开通了图像模型渠道，否则会返回 503 `model_not_found`，此时只能生成提示词。

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
analyzer/features.go # AI 用特征：明暗分布/对比度/锐度/主色/肤色天空占比
ai/deepseek.go       # DeepSeek（OpenAI 兼容）客户端 + 提示词与 JSON 解析
ai/suggest.go        # 标签/提示词数据结构、参数夹取、离线规则兜底
ai/imagegen.go       # 图生图客户端（OpenAI 兼容 /images/edits）
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
