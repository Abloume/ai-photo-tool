# Changelog

本项目所有重要变更都记录在此文件中。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)。

## 2026-11-02

### Added
- AI 修图建议：导入图片后自动分析，右侧顶部给出画面概述 + 4-6 个优化标签，点标签展示提示词并实时预览效果
- `ai/` 包：DeepSeek（兼容 OpenAI `/chat/completions`）客户端，JSON 模式输出标签/提示词/参数，失败自动退回离线规则
- `analyzer/features.go`：补充明暗分布、对比度、锐度、主色、肤色/绿植/天空占比等语义特征
- `/api/ai-suggest` 生成建议，`/api/apply-custom` 用自定义参数出图
- `.env` 本地密钥读取（不覆盖已有环境变量）+ `.env.example` 模板

### Changed
- 默认模型改为 `deepseek-flash`（V4.1-Flash），它支持图像理解，请求会附带 1024px 缩略图；`deepseek-v4-pro` 不支持看图，不再作为默认
- 新增 AI 出图（图生图）：提示词卡片加「🎨 AI 出图」，走 `/api/ai-generate` + OpenAI 兼容 `/images/edits`，支持 b64_json / url 两种返回；缺 `AI_IMAGE_MODEL` 时按钮置灰，网关无图像渠道时给出可照做的提示
- 视觉能力改为按模型名自动判断（含 `flash` / `vision` / `-vl`），`AI_VISION_MODEL` 仅用于单独指定
- 默认显式关闭思考模式（`thinking.type=disabled`），避免默认 high 导致出标签等几十秒；`AI_THINKING=1` 可打开
- 补 `max_tokens` 与 JSON 空返回重试，防止结构化输出被截断/清空

## 2026-10-27

### Added
- inference/ 通用 onnx 推理封装：自动探测动态库，支持动态 shape
- matting/ 抠图包：u2netp 快速版 + RMBG-2.0 v2 新版
- superresolution/ Real-ESRGAN 4 倍超分
- autocompose/ 自动裁剪：按黄金比例自动构图
- change-bg 换背景：抠图后填充纯色背景，支持 alpha 混合过渡
- AI 推荐抠图：分析图片特征自动选最合适的模型
- 前端：超分 tab、抠图模型选择、AI 推荐按钮、换背景颜色选择、自动裁剪按钮
- air 热重载开发环境 + Makefile 封装 + 浏览器自动刷新（轮询版）
- 20 个内置参数化滤镜 + 图片色彩分析 + 规则推荐滤镜
- AI 一键优化（自动调曝光/白平衡/对比度）

### Changed
- 批量应用改为只更新预览（缩略图 + 对比区），不再自动下载；确认效果后点「保存全部」或「保存图片」手动下载
- 分版本抠图结构：v1 rmbg 库保留，v2 RMBG-2.0 可选，自动 fallback
- rmbg 库（v1）输出白色背景 → flood fill 转透明通道，避免误判人物身上的白色
- Apply/Analyze 按行并行化，大图更快
- 推荐滤镜落到滤镜栏顶部
- 补全 Grain 胶片颗粒滤镜

### Fixed
- 批量滤镜应用参数名与后端不一致（`filter` → `index`），避免批量处理失败
- onnxruntime 环境初始化容错（rmbg 库已初始化时跳过）
- RMBG-2.0 边缘噪点：alpha 阈值压缩缓解
