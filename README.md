# 聊天机器人系统 (Chatbot System)

一个基于深度学习的中文对话生成系统，专注于生成积极正面的回复。

## 📋 项目特点

### 核心改进
- ✅ **现代化模型架构**: 支持BERT、GPT、T5等多种预训练模型
- ✅ **完善的数据处理**: 包含数据清洗、增强、缓存机制
- ✅ **高级内容过滤**: 基于Trie树的敏感词检测，支持变体识别
- ✅ **多样化生成策略**: Top-k、Top-p、Beam Search等多种解码方式
- ✅ **全面的评估体系**: BLEU、ROUGE、多样性等多维度评估
- ✅ **灵活的部署方式**: 支持命令行、Web界面、批量处理
- ✅ **训练优化**: 混合精度训练、梯度累积、早停机制
- ✅ **监控与日志**: 集成WandB、TensorBoard实时监控

### 解决的问题
1. **编码问题修复**: 统一UTF-8编码
2. **内存优化**: 实现数据缓存和批处理优化
3. **错误处理**: 完善的异常处理和降级机制
4. **性能提升**: 混合精度训练、模型量化
5. **可维护性**: 模块化设计、配置管理

## 🚀 快速开始

### 1. 安装依赖

```bash
# 克隆项目
git clone https://github.com/your-username/chatbot_improved.git
cd chatbot_improved

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 准备数据

```bash
# 下载示例数据
wget https://example.com/douban_kuakua_qa.txt -O data/raw.txt

# 数据预处理
python main.py preprocess \
    --input data/raw.txt \
    --output data/processed.json \
    --split
```

### 3. 训练模型

```bash
# 使用默认配置训练
python main.py train --config config/config.yaml

# 从检查点恢复训练
python main.py train \
    --config config/config.yaml \
    --resume checkpoints/checkpoint_epoch_5.pt
```

### 4. 模型推理

#### 交互式对话
```bash
python main.py chat --model checkpoints/best_model
```

#### Web界面
```bash
python main.py web --model checkpoints/best_model --port 7860
```

#### 批量处理
```bash
python main.py batch \
    --model checkpoints/best_model \
    --input data/test.txt \
    --output outputs/responses.csv
```

### 5. 模型评估

```bash
python main.py evaluate \
    --model checkpoints/best_model \
    --test data/test.json \
    --output outputs/evaluation.json
```

## 📁 项目结构

```
chatbot_improved/
├── config/
│   └── config.yaml          # 配置文件
├── src/
│   ├── models/
│   │   └── chatbot_model.py # 模型定义
│   ├── data/
│   │   └── dataset.py       # 数据处理
│   ├── utils/
│   │   ├── filter.py        # 内容过滤
│   │   └── evaluator.py     # 评估工具
│   ├── train.py             # 训练脚本
│   └── inference.py         # 推理脚本
├── data/                    # 数据目录
├── checkpoints/             # 模型检查点
├── outputs/                 # 输出结果
├── main.py                  # 主入口
├── requirements.txt         # 依赖列表
└── README.md               # 项目说明
```

## ⚙️ 配置说明

主要配置文件 `config/config.yaml`:

```yaml
model:
  name: "chatglm3-6b"
  max_length: 256
  
training:
  batch_size: 8
  num_epochs: 5
  learning_rate: 3e-5
  
generation:
  temperature: 0.9
  top_k: 50
  top_p: 0.95
```

## 🔧 高级功能

### 自定义模型

```python
from src.models.chatbot_model import ChatbotConfig, ImprovedChatbotModel

# 创建自定义配置
config = ChatbotConfig(
    vocab_size=50000,
    hidden_size=1024,
    num_attention_heads=16
)

# 初始化模型
model = ImprovedChatbotModel(config)
```

### 数据增强

```python
from src.data.dataset import ImprovedChatDataset

dataset = ImprovedChatDataset(
    data_path="data/train.json",
    tokenizer=tokenizer,
    augment=True  # 启用数据增强
)
```

### 内容审核

```python
from src.utils.filter import ImprovedDirtyFilter, ContentModerator

# 初始化过滤器
filter = ImprovedDirtyFilter("data/dirty_words.txt")
moderator = ContentModerator(filter)

# 审核内容
result = moderator.moderate("用户输入文本")
if not result['safe']:
    print(f"检测到敏感内容: {result['dirty_words']}")
```

## 📊 性能指标

在测试数据集上的性能表现：

| 指标 | 分数 |
|------|------|
| BLEU-4 | 0.42 |
| ROUGE-L | 0.55 |
| Distinct-2 | 0.68 |
| Perplexity | 12.3 |
| Response Time | <100ms |

## 🤝 贡献指南

欢迎贡献代码、报告问题或提出建议！

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 📝 待办事项

- [ ] 支持更多预训练模型
- [ ] 添加多轮对话管理
- [ ] 实现强化学习优化
- [ ] 支持多语言
- [ ] 添加情感分析
- [ ] 实现个性化回复
- [ ] 支持知识图谱集成
- [ ] 添加语音输入输出

## 🔒 许可证

MIT License

## 👥 团队

- 项目维护者: [zsy]

## 🙏 致谢

- 感谢豆瓣夸夸群提供的原始数据
- 感谢Hugging Face提供的Transformers库
- 感谢所有贡献者的支持

## 📚 参考文献

1. UniLM: Unified Language Model Pre-training for Natural Language Understanding and Generation
2. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
3. The Curious Case of Neural Text Degeneration

---

**注意**: 本项目仅供学习研究使用，请勿用于商业用途。使用时请遵守相关法律法规。