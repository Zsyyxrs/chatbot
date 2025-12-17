# CLAUDE.md

此文件为 Claude Code (claude.ai/code) 在此代码库中工作提供指导。

## 项目概述

这是一个基于深度学习的改进中文聊天机器人系统，专注于生成积极正面的回复。这是一个基于 PyTorch 的项目，支持多种预训练模型（BERT、GPT、T5），具有混合精度训练、内容过滤和全面评估指标等现代功能。

## 技术栈

- **框架**: PyTorch + Transformers 库
- **模型**: 支持 BERT、GPT、T5 和其他中文模型
- **语言**: Python 3.7+
- **核心库**: torch, transformers, numpy, pandas, gradio, wandb, tensorboard
- **文本处理**: jieba 中文分词
- **评估指标**: BLEU, ROUGE, 多样性指标

## 开发命令

### 环境配置和安装
```bash
# 安装依赖
pip install -r requirement.txt

# 注意：文件名为 "requirement.txt" (单数形式，不是 "requirements.txt")
```

### 主要命令 (通过 main.py)
```bash
# 训练模型
python main.py train --config config/config.yaml
python main.py train --config config/config.yaml --resume checkpoints/checkpoint_epoch_5.pt

# 交互式聊天
python main.py chat --model checkpoints/best_model

# Web 界面 (Gradio)
python main.py web --model checkpoints/best_model --port 7860

# 批量处理
python main.py batch --model checkpoints/best_model --input data/test.txt --output outputs/responses.csv

# 数据预处理
python main.py preprocess --input data/raw.txt --output data/processed.json --split

# 模型评估
python main.py evaluate --model checkpoints/best_model --test data/test.json --output outputs/evaluation.json
```

### 测试
```bash
# 运行演示测试
python test_improvements.py

# 基础测试文件 (内容较少)
python test.py
```

## 架构概述

### 核心组件
1. **模型** (`src/models/chatbot_model.py`): 支持多种预训练模型的自定义 Transformer 架构
2. **数据处理** (`src/data/dataset.py`): 具有缓存、增强和批处理优化的高级数据集
3. **训练** (`src/train.py`): 现代化训练流水线，包含混合精度、梯度累积、早停机制
4. **推理** (`src/inference.py`): 生产就绪的推理系统，支持 Web 界面
5. **内容过滤** (`src/utils/filter.py`): 基于字典树的敏感词检测，支持变体识别
6. **评估** (`src/utils/evaluator.py`): 综合指标评估，包括 BLEU、ROUGE、多样性

### 关键设计模式
- **模块化架构**: 数据、模型、训练和推理之间清晰分离
- **配置驱动**: 基于 YAML 的配置文件 `config/config.yaml`
- **缓存系统**: 数据预处理和模型响应缓存以提高效率
- **内容安全**: 内置内容审核和敏感词过滤
- **多模式部署**: 支持命令行、Web 界面和批处理

### 数据流
1. 原始对话数据 → `DataPreprocessor` → 清理后的 JSON 格式
2. 处理后数据 → `ImprovedChatDataset` → 分词后批次
3. 训练流水线 → 模型检查点 → 推理部署
4. 用户输入 → 内容过滤 → 模型生成 → 响应审核

## 配置文件

主配置文件：`config/config.yaml`
- 模型设置（架构、预训练模型选择）
- 训练参数（批次大小、学习率、优化器）
- 生成参数（温度、top-k、top-p、束搜索）
- 数据路径和缓存目录

## 重要说明

- 项目使用 jieba 进行中文文本处理和分词
- 内容过滤系统使用 `data/dirty_words.txt` 进行敏感词检测
- 模型检查点保存在 `checkpoints/` 目录
- 所有输出保存在 `outputs/` 目录
- 系统优先生成积极正面的回复
- Web 界面基于 Gradio，支持分享功能
- 支持混合精度训练以提高效率