#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试脚本 - 演示改进功能
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.utils.filter import DirtyFilter, ContentModerator
from src.utils.evaluator import Evaluator
from src.data.dataset import DataPreprocessor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_dirty_filter():
    """测试敏感词过滤器"""
    print("\n" + "="*50)
    print("测试敏感词过滤器")
    print("="*50)
    
    # 初始化过滤器
    filter = DirtyFilter()
    
    # 添加测试敏感词
    test_words = ["测试敏感词", "bad word", "违禁词"]
    for word in test_words:
        filter.add_word(word)
    
    # 测试文本
    test_texts = [
        "这是一个正常的句子",
        "这个句子包含测试敏感词",
        "This contains a bad word here",
        "这里有违禁词和其他内容",
        "测试变体检测：b@d w0rd",
        "测试重复字符：违违违禁词词词"
    ]
    
    for text in test_texts:
        result = filter.check(text)
        found = filter.find_dirty_words(text) if result else []
        print(f"文本: {text}")
        print(f"  包含敏感词: {result}")
        if found:
            print(f"  发现: {found}")
        print()


def test_content_moderator():
    """测试内容审核器"""
    print("\n" + "="*50)
    print("测试内容审核器")
    print("="*50)
    
    # 初始化
    filter = DirtyFilter()
    filter.add_word("违禁词")
    moderator = ContentModerator(filter)
    
    # 测试文本
    test_texts = [
        "这是一个友好的对话",
        "这个包含违禁词的内容",
        "让我们讨论一些积极的话题"
    ]
    
    for text in test_texts:
        result = moderator.moderate(text)
        print(f"文本: {text}")
        print(f"  安全: {result['safe']}")
        print(f"  动作: {result['action']}")
        if not result['safe']:
            alternative = moderator.suggest_alternative(text)
            print(f"  建议替代: {alternative}")
        print()


def test_evaluator():
    """测试评估器"""
    print("\n" + "="*50)
    print("测试评估器")
    print("="*50)
    
    # 初始化评估器
    evaluator = Evaluator(language='chinese')
    
    # 测试数据
    predictions = [
        "你做得真棒，继续加油！",
        "哇，这真是太厉害了！",
        "你的努力一定会有回报的"
    ]
    
    references = [
        "你做得很好，继续努力！",
        "真的很厉害，太棒了！",
        "努力终将得到回报"
    ]
    
    # 计算指标
    results = evaluator.evaluate_generation(
        predictions=predictions,
        references=references,
        calculate_all=True
    )
    
    # 打印结果
    evaluator.print_evaluation_report(results)


def test_data_preprocessor():
    """测试数据预处理器"""
    print("\n" + "="*50)
    print("测试数据预处理器")
    print("="*50)
    
    # 初始化预处理器
    preprocessor = DataPreprocessor()
    
    # 测试文本清理
    test_texts = [
        "这是一个<b>HTML标签</b>的例子",
        "多个标点符号！！！！！",
        "  多余的   空格   ",
        "特殊字符￥%&*测试"
    ]
    
    print("文本清理测试:")
    for text in test_texts:
        cleaned = preprocessor.clean_text(text)
        print(f"  原始: {text}")
        print(f"  清理后: {cleaned}")
        print()


def test_model_improvements():
    """演示模型改进"""
    print("\n" + "="*50)
    print("模型架构改进")
    print("="*50)
    
    improvements = [
        "1. 支持多种预训练模型（BERT、GPT、T5等）",
        "2. 实现自定义Transformer架构",
        "3. 添加解码器层提升生成质量",
        "4. 混合精度训练支持",
        "5. 梯度累积优化显存使用",
        "6. 学习率调度策略",
        "7. 早停机制防止过拟合",
        "8. 模型缓存提升推理速度"
    ]
    
    for improvement in improvements:
        print(f"✓ {improvement}")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("聊天机器人改进演示")
    print("="*60)
    
    # 运行各项测试
    test_dirty_filter()
    test_content_moderator()
    test_data_preprocessor()
    test_evaluator()
    test_model_improvements()
    
    print("\n" + "="*60)
    print("主要改进总结")
    print("="*60)
    
    summary = """
    🔧 技术改进:
    • 修复了字符编码问题
    • 优化了内存使用和数据处理效率
    • 增强了错误处理和异常恢复机制
    • 实现了模块化设计和配置管理
    
    🚀 功能增强:
    • 升级到现代化的模型架构
    • 实现了多样化的生成策略
    • 添加了全面的评估体系
    • 支持多种部署方式（CLI、Web、批处理）
    
    📊 性能优化:
    • 混合精度训练加速
    • 响应缓存减少重复计算
    • 批处理优化提升吞吐量
    • 模型量化减少资源消耗
    
    🛡️ 安全性:
    • 改进的敏感词过滤系统
    • 内容审核和替代建议
    • 支持变体和同音字检测
    • 实时内容监控
    """
    
    print(summary)


if __name__ == "__main__":
    main()