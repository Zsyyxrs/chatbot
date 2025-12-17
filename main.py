#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
改进的聊天机器人主程序
统一入口，支持训练、推理和数据处理
"""

import argparse
import sys
from pathlib import Path
import logging
import logging.config
import yaml
import os

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def setup_logging():
    """设置日志配置"""
    # 确保日志目录存在
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # 加载配置文件
    config_file = Path('logging_config.yaml')
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            logging.config.dictConfig(config)
    else:
        # 如果配置文件不存在，使用基本配置
        logging.basicConfig(level=logging.INFO)

def main():
    """主函数"""
    
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Improved Chatbot System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 训练模型
  python main.py train --config config/config.yaml
  
  # 交互式对话
  python main.py chat
  
  # 启动Web界面
  python main.py web --port 7860
  
  # 批量推理
  python main.py batch --input data/test.txt --output outputs/responses.csv
  
  # 数据预处理
  python main.py preprocess --input data/douban_kuakua_qa.txt --output data/processed.json --split
  
  # 模型评估
  python main.py evaluate --test data/test1.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # 训练命令
    train_parser = subparsers.add_parser('train', help='Train the chatbot model')
    train_parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Configuration file path'
    )
    train_parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Resume from checkpoint'
    )
    
    # 交互式聊天命令
    chat_parser = subparsers.add_parser('chat', help='Interactive chat mode')
    chat_parser.add_argument(
        '--model',
        type=str,
        default='checkpoints/best_model',
        help='Model path'
    )
    chat_parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Configuration file path'
    )
    
    # Web界面命令
    web_parser = subparsers.add_parser('web', help='Launch web interface')
    web_parser.add_argument(
        '--model',
        type=str,
        default='checkpoints/best_model',
        help='Model path'
    )
    web_parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Configuration file path'
    )
    web_parser.add_argument(
        '--port',
        type=int,
        default=7860,
        help='Server port'
    )
    
    # 批量推理命令
    batch_parser = subparsers.add_parser('batch', help='Batch inference')
    batch_parser.add_argument(
        '--model',
        type=str,
        default='checkpoints/best_model',
        help='Model path'
    )
    batch_parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Configuration file path'
    )
    batch_parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input file path'
    )
    batch_parser.add_argument(
        '--output',
        type=str,
        default='outputs/responses.csv',
        help='Output file path'
    )
    
    # 数据预处理命令
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess data')
    preprocess_parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input data file'
    )
    preprocess_parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output data file'
    )
    preprocess_parser.add_argument(
        '--dirty-words',
        type=str,
        default='data/暴恐词库.txt',
        help='Dirty words file'
    )
    preprocess_parser.add_argument(
        '--split',
        action='store_true',
        help='Split into train/val/test'
    )
    
    # 评估命令
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate model')
    evaluate_parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Configuration file path'
    )
    evaluate_parser.add_argument(
        '--test',
        type=str,
        required=True,
        help='Test data file'
    )
    evaluate_parser.add_argument(
        '--output',
        type=str,
        default='outputs/evaluation.json',
        help='Output evaluation results'
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # 执行相应的命令
    if args.command == 'train':
        from src.train import Trainer
        trainer = Trainer(args.config)
        if args.resume:
            trainer.load_checkpoint(args.resume)
        trainer.train()
    
    elif args.command == 'chat':
        from src.inference import ChatbotInference
        inference = ChatbotInference(args.config)

        inference.interactive_chat()
    
    elif args.command == 'web':
        from src.inference import ChatbotInference, create_gradio_interface
        inference = ChatbotInference(args.config)
        demo = create_gradio_interface(inference)
        # demo.launch(server_port=args.port, share=True)
        demo.launch(
            server_name="0.0.0.0",
            server_port=args.port,
            share=False,
            css="""
            .gradio-container {
                font-family: 'Microsoft YaHei', 'PingFang SC', sans-serif;
            }
            """
        )
    
    elif args.command == 'batch':
        from src.inference import ChatbotInference, BatchInference
        inference = ChatbotInference(args.config)
        batch_processor = BatchInference(inference)
        batch_processor.process_file(
            input_file=args.input,
            output_file=args.output
        )

    elif args.command == 'preprocess':
        from src.utils.filter import DirtyFilter
        from src.data.dataset import DataPreprocessor
        
        # 初始化敏感词过滤器
        dirty_filter = DirtyFilter(args.dirty_words)
        
        # 使用同一个过滤器
        preprocessor = DataPreprocessor(dirty_filter)
        
        preprocessor.process_dialogue_data(
            input_file=args.input,
            output_file=args.output
        )
        
        # 如果需要划分数据集
        if args.split:
            preprocessor.split_dataset(
                data_file=args.output,
                train_ratio=0.8,
                val_ratio=0.1,
                test_ratio=0.1
            )
    
    elif args.command == 'evaluate':
        from src.inference import ChatbotInference
        from src.utils.evaluator import Evaluator
        import json
        import pandas as pd
        
        # 初始化模型
        inference = ChatbotInference(args.config)
        
        # 加载测试数据
        test_data = pd.read_json(args.test, lines=True)
        
        # 生成预测
        predictions = []
        references = []
        
        for _, row in test_data.iterrows():
            src_text = row['src_text']
            tgt_text = row['tgt_text']
            
            # 生成回复
            prediction = inference.generate_response(src_text)
            
            predictions.append(prediction)
            references.append(tgt_text)
        
        # 评估
        evaluator = Evaluator(language='chinese')
        results = evaluator.evaluate_generation(
            predictions=predictions,
            references=references,
            calculate_all=True
        )
        
        # 保存结果
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 打印报告
        evaluator.print_evaluation_report(results)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()