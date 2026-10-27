# Changelog

本项目所有重要变更都记录在此文件中。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)。

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
- 分版本抠图结构：v1 rmbg 库保留，v2 RMBG-2.0 可选，自动 fallback
- rmbg 库（v1）输出白色背景 → flood fill 转透明通道，避免误判人物身上的白色
- Apply/Analyze 按行并行化，大图更快
- 推荐滤镜落到滤镜栏顶部
- 补全 Grain 胶片颗粒滤镜

### Fixed
- onnxruntime 环境初始化容错（rmbg 库已初始化时跳过）
- RMBG-2.0 边缘噪点：alpha 阈值压缩缓解
