# ARKAPITEST · GUI 视觉大模型性能评测平台

> 一个统一评测 **GUI 定位 / 推理能力**的视觉大模型测试框架，当前已接入 **9 个主流模型**，覆盖字节火山方舟 + 阿里云百炼两大 Provider。

[![Python](https://img.shields.io/badge/python-3.12-blue)]()
[![Provider](https://img.shields.io/badge/provider-ARK%20%7C%20DashScope-orange)]()
[![Version](https://img.shields.io/badge/version-v3.2-brightgreen)]()

---

## ✨ 特性

- **多 Provider 统一抽象**：一套代码同时调用火山方舟（ARK）和阿里云百炼（DashScope），都走 OpenAI 兼容协议
- **9 模型对比矩阵**：doubao-seed-1-6-vision / 1-5-vision-pro / qwen3-vl-flash/plus/32b × 2 / qwen3.6-flash/plus / gui-plus
- **方案 A Prompt 分化**：同一任务对不同模型自动切换其官方推荐坐标格式（`<point>` / `<bbox>` / `<point>(x,y)</point>` / `computer_use`）
- **双指标评估**：定位任务同时报告 **Precision（点命中率）** + **Recall（GT 覆盖率）**，避免单指标误导
- **9 种输出变体解析**：自动兼容各模型实际输出的包裹变体（含裸坐标兜底）
- **Thinking Matrix**：一键对比 thinking off/on 模式
- **并行执行 + 超时控制**：ThreadPoolExecutor 并发 4，单任务超时 120s，9 模型全矩阵 ~11 分钟
- **自动产出**：Markdown 报告 + JSON 原始数据 + 标记图（预测点 + GT bbox）

---

## 🚀 快速开始

### 环境准备

本项目使用 **`uv` + Python 3.12**（建议）。

```bash
# 克隆仓库
git clone https://github.com/JerryYuan4733/GUIModelTest.git
cd GUIModelTest

# 安装依赖
uv sync
```

### 配置 API Key

复制 `.env.example` 为 `.env` 并填入你的 key：

```bash
# 火山方舟（字节跳动） API Key  https://console.volcengine.com/ark
ARK_API_KEY=your_ark_api_key_here

# 阿里云百炼（DashScope） API Key  https://bailian.console.aliyun.com/
DASHSCOPE_API_KEY=your_dashscope_api_key_here
```

> 也可通过系统环境变量设置。Windows PowerShell：`$env:ARK_API_KEY="xxx"`

### 运行测试

```bash
# 运行全矩阵（9 模型 × 9 用例，thinking off/on 对比）
uv run python run_tests.py --thinking-matrix

# 只跑定位类测试
uv run python run_tests.py --category positioning

# 指定模型
uv run python run_tests.py --models qwen3.6-flash qwen3.6-plus

# 指定测试用例
uv run python run_tests.py --ids POS_001a POS_003

# 并行 + 超时控制
uv run python run_tests.py --parallel 4 --task-timeout 120

# 结果汇总（支持多文件合并）
uv run python scripts/summarize_results.py output/results_xxx.json

# 可视化 GT bbox（人工校准用）
uv run python scripts/visualize_gt.py --test POS_002
```

产物位于 `output/`：

- `report_<timestamp>.md` — 人类可读报告
- `results_<timestamp>.json` — 原始 JSON
- `POS_xxx_<model>_marked.png` — 标记预测点 + GT bbox 的对比图

---

## 📊 当前成绩速览（v3.2 矩阵，2026-04-20）

### 综合最推荐

| 场景 | 首选模型 | 关键指标 |
|---|---|---|
| **综合最强**（单模型走天下） | **`qwen3.6-flash`** 🏆 | 平均 4.0s，推理 91%，POS_003 on 67/67 |
| 极速单点 grounding | `qwen3.6-flash` | 0.5s ⚡ |
| 多目标定位 + 名称识别 | `qwen3.6-flash` on | 100/100 + 91% @ 4.4s |
| 类型过滤 · 追求 Recall | `qwen3.6-plus` off | 50/100（Recall 100% 唯一）🎯 |
| 类型过滤 · 追求 Precision | `qwen3.6-flash` on | 67/67 Precision 最高 |
| 低延迟 Agent 动作规划 | `gui-plus` | 3.3s |

### 平均响应时间

| 模型 | 平均 | |
|---|---:|---|
| gui-plus | 3.30s | 🥇 |
| **qwen3.6-flash** | **4.01s** | 🥈 |
| 1-5-vision-pro | 6.64s | |
| qwen3-vl-flash | 7.63s | |
| qwen3-vl-32b-instruct | 10.34s | |
| qwen3-vl-32b-thinking | 28.84s | |
| seed-1-6-vision | 29.88s | |
| qwen3.6-plus | 36.99s | |

完整矩阵数据见 [`@docs/explain/2026-04-20-1650-GUI-VLM调研与复现实验.md`](docs/explain/2026-04-20-1650-GUI-VLM调研与复现实验.md) §7.8。

---

## 🏗️ 项目结构

```
ARKAPITEST/
├── run_tests.py                  # CLI 入口
├── src/
│   ├── config.py                 # 模型注册表 + Provider 配置 + 全局常量
│   ├── client.py                 # VisionClient（单例 + per-provider 懒加载）
│   ├── coord_formats.py          # 按厂商切换的坐标格式模板
│   ├── gui_plus_prompts.py       # GUI-Plus 官方 system prompt
│   ├── test_cases.py             # 9 个测试用例定义
│   ├── ground_truth.py           # GT 坐标标注（0-1000 空间）
│   └── test_runner.py            # 执行引擎 + 评估 + 报告生成
├── scripts/
│   ├── summarize_results.py      # 结果矩阵汇总（支持多文件合并）
│   ├── visualize_gt.py           # GT bbox 可视化
│   └── convert_image.py          # 图片格式转换
├── resource/                     # 测试图片
├── output/                       # 运行产物（report / json / 标记图）
└── docs/
    ├── README 指南入口            # 本文档引导
    ├── explain/                  # 调研与复现实验报告 ⭐
    ├── architecture/             # 系统架构概览
    ├── development/              # 开发进度与计划
    └── reference/                # 外部 API / Cookbook 参考
```

---

## 📖 文档索引

| 文档 | 说明 |
|---|---|
| [`docs/explain/2026-04-20-1650-GUI-VLM调研与复现实验.md`](docs/explain/2026-04-20-1650-GUI-VLM调研与复现实验.md) | **主报告**：9 模型矩阵、调研思路、prompt 设计、评估方法、完整结果 |
| [`docs/architecture/2026-04-20-2100-系统架构概览.md`](docs/architecture/2026-04-20-2100-系统架构概览.md) | DDD 风格分层架构、核心抽象、数据流图、扩展点 |
| [`docs/development/2026-04-20-2100-开发进度-v3.2.md`](docs/development/2026-04-20-2100-开发进度-v3.2.md) | 版本里程碑、已完成功能清单、已知问题、下一阶段计划 |
| [`docs/reference/api_reference.md`](docs/reference/api_reference.md) | ARK / DashScope API 参数参考 |
| [`docs/reference/2026-04-20-1610-阿里云千问视觉模型接入调研.md`](docs/reference/2026-04-20-1610-阿里云千问视觉模型接入调研.md) | Qwen 视觉模型接入调研笔记 |

---

## 🔧 如何添加新模型

**一行代码**即可（新 provider 需先在 `PROVIDER_CONFIG` 注册）。

在 `@src/config.py` 的 `MODELS` 字典追加：

```python
"my-new-model": ModelConfig(
    name="my-new-model",
    supports_thinking=True,
    display_name="my-model",
    provider="dashscope",          # 或 "ark"
    output_format="qwen_point",    # point | bbox | qwen_point | tool_call
    vl_high_resolution=True,       # DashScope 视觉模型建议开
),
```

如果新模型有特殊输出变体，在 `@src/test_runner.py::_extract_predictions` 加一个解析分支即可。

---

## 🧪 评估方法

### 定位任务

- **点命中率（Precision）**：中心点 `(cx, cy)` 落在任一 GT bbox 内（±5 单位容差）即算命中 → `hits / predictions`
- **GT 覆盖率（Recall）**：被至少一个预测命中的 GT 数 / GT 总数
- **IoU 兜底**（仅当模型输出 bbox）：IoU ≥ 0.3 也判为命中

坐标空间：0-1000 归一化，各模型统一输出此空间的坐标。

### 推理任务

- **群组名称识别准确率**：模型回复中提到的群组名 vs GT 列表的模糊匹配率
- **响应时间**：端到端（含网络）秒数
- **Token 用量**：prompt / completion / total

---

## 🛠 运行参数

```bash
uv run python run_tests.py [OPTIONS]

Options:
  --category {positioning,reasoning,all}  测试类别过滤，默认 all
  --models MODELS ...                     模型列表，默认所有注册模型
  --ids IDS ...                           指定测试用例 ID，如 POS_001a RSN_001
  --thinking-matrix                       对支持 thinking 的模型额外跑 enable_thinking=True
  --parallel N                            并发数（默认 4，设 1 则串行）
  --task-timeout SEC                      单任务超时秒数（默认 120）
  --screenshot                            包含实时截图测试（需 GUI 环境）
```

---

## 📝 版本变更

| 日期 | 版本 | 主要变更 |
|---|---|---|
| 2026-04-20 20:55 | **v3.2** | 新增 `qwen3.6-plus` + 解析器格式 D' |
| 2026-04-20 18:55 | v3.1 | 新增 `qwen3.6-flash` + 裸坐标兜底 |
| 2026-04-20 18:20 | v3.0 | 方案 A 上线：按厂商官方格式切换 prompt |
| 2026-04-20 17:59 | v2.0 | 并行化 + 单任务超时 |
| 2026-04-20 17:10 | v1.2 | 7 模型串行完整矩阵 |
| 2026-04-20 16:18 | v1.0 | 首版 4 模型 |

完整版本历史见 `@docs/development/2026-04-20-2100-开发进度-v3.2.md`。

---

## 🤝 贡献

本项目遵循 DDD 分层架构设计，核心扩展点：

1. **新模型** → `@src/config.py` 注册表
2. **新 Provider** → `@src/config.py` `PROVIDER_CONFIG` + `@src/client.py` 适配
3. **新测试用例** → `@src/test_cases.py` + `@src/ground_truth.py`
4. **新输出格式** → `@src/coord_formats.py` + `@src/test_runner.py::_extract_predictions`

---

## 📄 License

内部研究用途。外部 API 调用遵循各家 Provider 官方服务条款。

---

## 🔗 相关资源

- 火山方舟控制台：https://console.volcengine.com/ark
- 阿里云百炼控制台：https://bailian.console.aliyun.com/
- Doubao-Vision 文档：https://www.volcengine.com/docs/82379/1616136
- Qwen3-VL Cookbook：https://developer.aliyun.com/article/1685124
