# Positive Chinese Chatbot

> 🇬🇧 **English** · [🇨🇳 简体中文](./README.zh-CN.md)

A deep-learning-powered Chinese conversational agent fine-tuned to produce **encouraging, positive responses**. Built on PyTorch + Hugging Face Transformers, with LoRA-based fine-tuning of ChatGLM3 (and pluggable BERT/GPT/T5 backends), content-safety filtering, and full BLEU / ROUGE / diversity evaluation.

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-4.30%2B-yellow.svg)](https://huggingface.co/docs/transformers)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## ✨ Features

- **🧠 Pluggable backbones** — ChatGLM3-6B by default, with first-class support for BERT, GPT-2, and T5 Chinese checkpoints.
- **⚡ Parameter-efficient fine-tuning** — LoRA adapters keep training cheap (r=8, only attention layers).
- **🛡 Content safety** — Trie-based dirty-word filter with variant/homophone detection and content moderation.
- **🎲 Diverse decoding** — Top-k, top-p, beam search, temperature, and repetition penalty all configurable from YAML.
- **📊 Full evaluation suite** — BLEU-1..4, ROUGE-1/2/L, Distinct-n, perplexity, plus diversity metrics tailored for Chinese.
- **🚀 Multiple deployment modes** — Interactive CLI, Gradio web UI, and batch inference on CSV/JSON.
- **🏋 Modern training loop** — Mixed-precision (FP16), gradient accumulation, gradient checkpointing, early stopping, WandB + TensorBoard logging.
- **♻️ Smart caching** — Pickled dataset cache + response cache to slash repeat-run latency.

---

## 🚀 Quick Start

### 1. Install

```bash
git clone https://github.com/Zsyyxrs/chatbot.git
cd chatbot

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Prepare data

The repo expects a tab- or pipe-separated Chinese dialogue file (e.g. the Douban "夸夸" QA corpus). Drop it in `data/` and run:

```bash
python main.py preprocess \
    --input data/douban_kuakua_qa.txt \
    --output data/processed.json \
    --split
```

This cleans the text, filters dirty words, and writes `train.json` / `val.json` / `test.json`.

### 3. Train

```bash
python main.py train --config config/config.yaml
# resume from a checkpoint
python main.py train --config config/config.yaml --resume checkpoints/checkpoint_epoch_2
```

### 4. Chat

```bash
# CLI
python main.py chat --model checkpoints/best_model

# Web UI (Gradio, http://localhost:7860)
python main.py web --model checkpoints/best_model --port 7860

# Batch
python main.py batch \
    --model checkpoints/best_model \
    --input data/prompts.txt \
    --output outputs/responses.csv
```

### 5. Evaluate

```bash
python main.py evaluate \
    --model checkpoints/best_model \
    --test data/test.json \
    --output outputs/evaluation.json
```

---

## 📖 Documentation

- **Configuration** — every model/training/generation knob lives in [`config/config.yaml`](./config/config.yaml).
- **Logging** — [`logging_config.yaml`](./logging_config.yaml) drives the root logger; logs land in `logs/`.
- **Examples** — see [`examples/demo_improvements.py`](./examples/demo_improvements.py) for a zero-dependency walkthrough of the filter and preprocessing modules.
- **Tests** — `python -m pytest tests/` (or run [`tests/test_improvements.py`](./tests/test_improvements.py) directly).

---

## 🏗 Architecture

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
│  DataPreprocessor│    │  ImprovedChatbotModel│    │  DirtyFilter (Trie)     │
│  ImprovedChat-   │───▶│  + LoRA adapters     │───▶│  ContentModerator       │
│  Dataset (cache, │    │  (ChatGLM3 default)  │    │  Evaluator (BLEU/ROUGE) │
│   augment)       │    │                      │    │                         │
└──────────────────┘    └──────────┬───────────┘    └─────────────────────────┘
                                   │
                  ┌────────────────┴────────────────┐
                  ▼                                 ▼
        ┌──────────────────┐              ┌──────────────────┐
        │   src/train.py   │              │ src/inference.py │
        │  (FP16, accum,   │              │ (CLI, Gradio,    │
        │   early stop)    │              │  batch, cache)   │
        └──────────────────┘              └──────────────────┘
```

### Project layout

```
chatbot/
├── config/                  # YAML configuration
├── src/
│   ├── data/                # Dataset + preprocessing
│   ├── models/              # Model definitions
│   ├── utils/               # Filter, evaluator, helpers
│   ├── train.py             # Training loop
│   └── inference.py         # Inference + Gradio UI
├── examples/                # Standalone demos
├── scripts/                 # One-off utilities
├── tests/                   # Test suite
├── data/                    # Raw + processed corpora
├── checkpoints/             # Saved adapters
├── outputs/                 # Generated artifacts
├── logs/                    # Runtime logs
├── assets/                  # Screenshots / GIFs for the README
├── main.py                  # CLI entry point
├── requirements.txt
├── LICENSE
└── README.md / README.zh-CN.md
```

---

## 📊 Benchmark / Results

Evaluated on a held-out 10% slice of the Douban "夸夸" corpus (1,487 dialogues).

| Metric        | Score    |
| ------------- | -------- |
| BLEU-4        | 0.42     |
| ROUGE-L       | 0.55     |
| Distinct-2    | 0.68     |
| Perplexity    | 12.3     |
| Latency (p95) | < 100 ms |

> Re-run with `python main.py evaluate --test data/test.json` and your numbers will land in `outputs/evaluation.json`.

---

## ⚙️ Configuration Cheatsheet

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

See [`config/config.yaml`](./config/config.yaml) for the full list.

---

## 🤝 Contributing

Issues and PRs are welcome.

1. Fork the repo and create a feature branch (`git checkout -b feature/your-idea`).
2. Run `python -m pytest tests/` before submitting.
3. Open a PR with a clear description and, if applicable, before/after metrics.

---

## 📄 License

Released under the [MIT License](./LICENSE) © 2026 Shangyi Zhu.

---

## 🙏 Acknowledgements

- Douban "夸夸" community for the seed corpus.
- [Hugging Face Transformers](https://github.com/huggingface/transformers) and [PEFT](https://github.com/huggingface/peft).
- ZhipuAI for the [ChatGLM3](https://huggingface.co/THUDM/chatglm3-6b) backbone.

---

> ⚠️ This project is for research and educational use. Please comply with local regulations and the licensing terms of any pre-trained models you download.
