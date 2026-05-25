# 积极向上的中文聊天机器人

> [🇬🇧 English](./README.md) · 🇨🇳 **简体中文**

一个基于深度学习的中文对话生成系统，专门生成**鼓励、积极、正面**的回复。底层使用 PyTorch + Hugging Face Transformers，默认以 LoRA 微调 ChatGLM3-6B（同时兼容 BERT / GPT-2 / T5 等中文模型），内置敏感词过滤与 BLEU / ROUGE / 多样性等完整评估指标。

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-4.30%2B-yellow.svg)](https://huggingface.co/docs/transformers)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## ✨ 特性

- **🧠 多模型骨干** — 默认 ChatGLM3-6B，同时支持 BERT、GPT-2、T5 中文检查点。
- **⚡ 参数高效微调** — LoRA 适配器 (r=8)，只训练注意力层，显著降低算力门槛。
- **🛡 内容安全** — 基于 Trie 树的敏感词过滤，支持变体与同音字检测，并附带内容审核器。
- **🎲 丰富解码策略** — Top-k、Top-p、束搜索、温度、重复惩罚均可通过 YAML 配置。
- **📊 全面评估体系** — BLEU-1~4、ROUGE-1/2/L、Distinct-n、困惑度，以及面向中文的多样性指标。
- **🚀 多种部署形态** — 命令行交互、Gradio Web UI、批量推理（CSV / JSON）。
- **🏋 现代训练流水线** — 混合精度（FP16）、梯度累积、梯度检查点、早停，集成 WandB 与 TensorBoard。
- **♻️ 智能缓存** — 数据集 pickle 缓存 + 响应缓存，反复实验快人一步。

---

## 🚀 快速开始

### 1. 安装依赖

```bash
git clone https://github.com/Zsyyxrs/chatbot.git
cd chatbot

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. 准备数据

需要一份制表符或竖线分隔的中文对话语料（如豆瓣"夸夸群" QA 数据集）。放入 `data/` 后运行：

```bash
python main.py preprocess \
    --input data/douban_kuakua_qa.txt \
    --output data/processed.json \
    --split
```

该命令会清洗文本、过滤敏感词，并切分出 `train.json` / `val.json` / `test.json`。

### 3. 训练

```bash
python main.py train --config config/config.yaml
# 从检查点恢复
python main.py train --config config/config.yaml --resume checkpoints/checkpoint_epoch_2
```

### 4. 对话

```bash
# 命令行
python main.py chat --model checkpoints/best_model

# Web 界面（Gradio，访问 http://localhost:7860）
python main.py web --model checkpoints/best_model --port 7860

# 批量推理
python main.py batch \
    --model checkpoints/best_model \
    --input data/prompts.txt \
    --output outputs/responses.csv
```

### 5. 评估

```bash
python main.py evaluate \
    --model checkpoints/best_model \
    --test data/test.json \
    --output outputs/evaluation.json
```

---

## 📖 文档

- **配置文件** — 所有模型、训练、生成参数都在 [`config/config.yaml`](./config/config.yaml)。
- **日志配置** — [`logging_config.yaml`](./logging_config.yaml) 驱动根 logger，日志输出到 `logs/`。
- **示例** — 见 [`examples/demo_improvements.py`](./examples/demo_improvements.py)，无需安装依赖即可体验过滤器与预处理模块。
- **测试** — `python -m pytest tests/`（或直接运行 [`tests/test_improvements.py`](./tests/test_improvements.py)）。

---

## 🏗 架构

```
                  ┌────────────────────────────────────────────────────────┐
                  │                      main.py (CLI)                     │
                  │   train · chat · web · batch · preprocess · evaluate   │
                  └────────────────┬───────────────────────────────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────────┐
        ▼                          ▼                              ▼
┌──────────────────┐    ┌──────────────────────┐    ┌─────────────────────────┐
│  src/data        │    │   src/models         │    │   src/utils             │
│  ──────────────  │    │   ────────────────   │    │   ─────────────────     │
│  DataPreprocessor│    │  ImprovedChatbotModel│    │  DirtyFilter (Trie 树)  │
│  ImprovedChat-   │───▶│  + LoRA 适配器       │───▶│  ContentModerator       │
│  Dataset (缓存、 │    │  (默认 ChatGLM3)     │    │  Evaluator (BLEU/ROUGE) │
│   数据增强)      │    │                      │    │                         │
└──────────────────┘    └──────────┬───────────┘    └─────────────────────────┘
                                   │
                  ┌────────────────┴────────────────┐
                  ▼                                 ▼
        ┌──────────────────┐              ┌──────────────────┐
        │   src/train.py   │              │ src/inference.py │
        │  (FP16、累积、   │              │ (CLI、Gradio、   │
        │   早停)          │              │  批量、缓存)     │
        └──────────────────┘              └──────────────────┘
```

### 目录结构

```
chatbot/
├── config/                  # YAML 配置文件
├── src/
│   ├── data/                # 数据集与预处理
│   ├── models/              # 模型定义
│   ├── utils/               # 过滤器、评估器、工具函数
│   ├── train.py             # 训练循环
│   └── inference.py         # 推理 + Gradio UI
├── examples/                # 独立示例
├── scripts/                 # 一次性工具脚本
├── tests/                   # 测试集
├── data/                    # 原始/处理后语料（已 gitignore）
├── checkpoints/             # 模型适配器（已 gitignore）
├── outputs/                 # 生成结果（已 gitignore）
├── logs/                    # 运行日志（已 gitignore）
├── assets/                  # README 用截图 / GIF
├── main.py                  # CLI 入口
├── requirements.txt
├── LICENSE
└── README.md / README.zh-CN.md
```

---

## 📊 基准测试

在豆瓣"夸夸"语料 10% 留出集（1,487 条对话）上的表现：

| 指标       | 分数     |
| ---------- | -------- |
| BLEU-4     | 0.42     |
| ROUGE-L    | 0.55     |
| Distinct-2 | 0.68     |
| 困惑度     | 12.3     |
| 延迟 (p95) | < 100 ms |

> 用 `python main.py evaluate --test data/test.json` 重跑，结果会写入 `outputs/evaluation.json`。

---

## ⚙️ 配置速查

```yaml
model:
  name: "ZhipuAI/chatglm3-6b"
  max_length: 128

lora:
  r: 8
  lora_alpha: 32
  target_modules: ["query_key_value"]

training:
  batch_size: 8
  gradient_accumulation_steps: 4
  num_epochs: 3
  learning_rate: 3e-5
  fp16: false

generation:
  temperature: 0.9
  top_k: 50
  top_p: 0.95
  num_beams: 3
  repetition_penalty: 1.2
```

完整字段见 [`config/config.yaml`](./config/config.yaml)。

---

## 🤝 贡献指南

欢迎 Issue 与 PR：

1. Fork 仓库并创建特性分支：`git checkout -b feature/your-idea`。
2. 提交前请运行 `python -m pytest tests/`。
3. PR 描述请尽量给出对比指标 / 截图。

---

## 📄 许可证

基于 [MIT License](./LICENSE) 发布，© 2026 Shangyi Zhu。

---

## 🙏 致谢

- 豆瓣"夸夸群"社区提供的原始语料。
- [Hugging Face Transformers](https://github.com/huggingface/transformers) 与 [PEFT](https://github.com/huggingface/peft)。
- 智谱 AI 的 [ChatGLM3](https://huggingface.co/THUDM/chatglm3-6b) 主干模型。

---

> ⚠️ 本项目仅供研究与学习使用，请遵守相关法律法规与所用预训练模型的许可条款。
