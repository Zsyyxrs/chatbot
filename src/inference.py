# -*- coding: utf-8 -*-
"""
推理脚本 - 改进版
用于模型部署和交互式对话,支持LoRA微调模型
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List
import yaml
import time
import json

import torch
import gradio as gr
from tqdm import tqdm

# 设置日志
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class ChatbotInference:
    """聊天机器人推理类 - 支持LoRA模型"""
    
    def __init__(
        self,
        config_path: str
    ):
        """
        初始化推理器
        """
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        base_model_path = self.config['model']['dir']
        lora_weights_path = self.config['lora']['lora_path']
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        use_fp16 = self.config['training']['fp16']
    
        logger.info("="*60)
        logger.info("Initializing Chatbot Inference")
        logger.info(f"Base model: {base_model_path}")
        logger.info(f"LoRA weights: {lora_weights_path}")
        logger.info(f"Device: {device}")
        logger.info(f"FP16: {use_fp16}")
        logger.info("="*60)
        
        # 加载模型
        self._load_model(base_model_path, lora_weights_path, use_fp16)
        
        # 对话历史
        self.conversation_history = []
        self.max_history = self.config.get('max_history', 10)
        
        # 默认回复
        self.default_responses = {
            'error': "抱歉，我现在有些困惑，能换个话题吗？",
            'empty': "请输入您想说的话。"
        }
        
        logger.info("Chatbot initialized successfully!\n")
    
    def _load_model(
        self,
        base_model_path: str,
        lora_weights_path: Optional[str],
        use_fp16: bool
    ):
        """加载模型"""
        try:
            from modelscope import AutoTokenizer, AutoModel, AutoConfig
            
            logger.info("Loading config...")
            config = AutoConfig.from_pretrained(
                base_model_path,
                trust_remote_code=True
            )
            
            config_defaults = {
                'num_hidden_layers': 28,
                'num_attention_heads': 32,
                'hidden_size': 4096,
                'vocab_size': 65024,
                'padded_vocab_size': 65024,
            }
            
            for attr, default_value in config_defaults.items():
                if not hasattr(config, attr):
                    setattr(config, attr, default_value)
            
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_path,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id or 0
            self.tokenizer.padding_side = 'left'
            
            logger.info("Loading model...")
            self.model = AutoModel.from_pretrained(
                base_model_path,
                config=config,
                trust_remote_code=True,
                torch_dtype=torch.float16 if use_fp16 else torch.float32,
                device_map="auto"
            )
            
            if lora_weights_path:
                logger.info(f"Loading LoRA weights from: {lora_weights_path}")
                from peft import PeftModel
                
                peft_model = PeftModel.from_pretrained(
                    self.model,
                    lora_weights_path,
                    torch_dtype=torch.float16 if use_fp16 else torch.float32
                )
                
                logger.info("Merging LoRA weights into base model...")
                self.model = peft_model.merge_and_unload()
                logger.info("✓ LoRA weights merged successfully")
            
            # ✅ Monkey patch: 添加缺失的方法
            if not hasattr(self.model, '_extract_past_from_model_output'):
                def _extract_past_from_model_output(outputs):
                    """从模型输出中提取 past_key_values"""
                    if hasattr(outputs, 'past_key_values'):
                        return outputs.past_key_values
                    return None
                
                self.model._extract_past_from_model_output = _extract_past_from_model_output
                logger.info("✓ Added _extract_past_from_model_output method")
            
            # ✅ 修复 _update_model_kwargs_for_generation 方法
            if hasattr(self.model, '_update_model_kwargs_for_generation'):
                original_update = self.model._update_model_kwargs_for_generation
                
                def patched_update(outputs, model_kwargs, is_encoder_decoder=False, **kwargs):
                    """修复的 kwargs 更新方法"""
                    # 提取 past_key_values
                    if hasattr(outputs, 'past_key_values') and outputs.past_key_values is not None:
                        model_kwargs["past_key_values"] = outputs.past_key_values
                    else:
                        model_kwargs["past_key_values"] = None
                    
                    return model_kwargs
                
                self.model._update_model_kwargs_for_generation = patched_update
                logger.info("✓ Patched _update_model_kwargs_for_generation method")
            
            self.model.eval()
            logger.info(f"✓ Model loaded in {next(self.model.parameters()).dtype}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def generate_response(
        self,
        user_input: str,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        max_length: int = 100,
        repetition_penalty: float = 1.2,
        use_history: bool = True
    ) -> str:
        """生成回复"""
        
        if not user_input or not user_input.strip():
            return self.default_responses['empty']
        
        try:
            start_time = time.time()
            
            prompt = f"<|user|>\n{user_input}<|assistant|>\n"
            tokens = self.tokenizer.tokenize(prompt)
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_ids = [64790, 64792] + input_ids
            
            input_ids_tensor = torch.tensor([input_ids], dtype=torch.long)
            device = next(self.model.parameters()).device
            input_ids_tensor = input_ids_tensor.to(device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids_tensor,
                    max_length=len(input_ids) + max_length,
                    do_sample=True,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    use_cache=False,  # ✅ 关键
                    pad_token_id=self.tokenizer.pad_token_id or 0,
                    eos_token_id=self.tokenizer.eos_token_id or 2
                )
            
            generated_ids = outputs[0][len(input_ids):]
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            
            logger.info(f"✓ Generated in {time.time() - start_time:.2f}s")
            
            if use_history:
                self._update_history(user_input, response)
            
            return response
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return self.default_responses.get('error', "抱歉，生成回复时出错了。")
    
    def _build_context(self, user_input: str) -> str:
        """构建包含历史的上下文"""
        context_parts = []
        
        # 添加最近的历史对话(最多3轮)
        for user_msg, bot_msg in self.conversation_history[-3:]:
            context_parts.append(f"用户：{user_msg}")
            context_parts.append(f"助手：{bot_msg}")
        
        # 添加当前输入
        context_parts.append(f"用户：{user_input}")
        context_parts.append("助手：")
        
        return "\n".join(context_parts)
    
    def _update_history(self, user_input: str, response: str):
        """更新对话历史"""
        self.conversation_history.append((user_input, response))
        
        # 限制历史长度
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)
    
    def clear_history(self):
        """清空对话历史"""
        self.conversation_history.clear()
        logger.info("Conversation history cleared")
    
    def interactive_chat(self):
        """交互式对话"""
        print("\n" + "="*60)
        print("🤖 聊天机器人 - 交互模式")
        print("="*60)
        print("命令说明:")
        print("  - 输入 'quit' 或 'exit' 退出")
        print("  - 输入 'clear' 清空对话历史")
        print("  - 输入 'history' 查看对话历史")
        print("="*60 + "\n")
        
        while True:
            try:
                # 获取用户输入
                user_input = input("👤 你: ").strip()
                
                # 处理特殊命令
                if user_input.lower() in ['quit', 'exit', '退出']:
                    print("\n👋 再见!")
                    break
                    
                elif user_input.lower() in ['clear', '清空']:
                    self.clear_history()
                    print("✓ 对话历史已清空!\n")
                    continue
                    
                elif user_input.lower() in ['history', '历史']:
                    if self.conversation_history:
                        print("\n对话历史:")
                        for i, (user_msg, bot_msg) in enumerate(self.conversation_history, 1):
                            print(f"\n[{i}]")
                            print(f"  你: {user_msg}")
                            print(f"  机器人: {bot_msg}")
                        print()
                    else:
                        print("暂无对话历史\n")
                    continue
                    
                elif not user_input:
                    continue
                
                # 生成回复
                response = self.generate_response(user_input)
                print(f"🤖 机器人: {response}\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 再见!")
                break
                
            except Exception as e:
                logger.error(f"Error in interactive chat: {e}")
                print(f"🤖 机器人: {self.default_responses['error']}\n")


def create_gradio_interface(inference: ChatbotInference):
    """创建Gradio Web界面"""
    
    def chat_function(
        message: str,
        history: List,
        temperature: float,
        top_k: int,
        top_p: float,
        max_length: int
    ):
        """Gradio聊天函数"""
        
        # 清空内部历史,使用Gradio的历史
        inference.clear_history()
        
        # 从Gradio历史重建对话历史
        # Gradio 4.x 使用字典格式: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
        if history:
            # 检查 history 格式
            if isinstance(history[0], dict):
                # Gradio 4.x 格式
                user_msg = None
                for msg in history:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    
                    if role == "user":
                        user_msg = content
                    elif role == "assistant" and user_msg is not None:
                        # 成对添加用户消息和助手回复
                        inference._update_history(user_msg, content)
                        user_msg = None
            else:
                # Gradio 3.x 格式（兼容旧版）
                for user_msg, bot_msg in history:
                    inference._update_history(user_msg, bot_msg)
        
        # 生成回复
        response = inference.generate_response(
            user_input=message,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_length=max_length,
            repetition_penalty=1.2,
            use_history=True
        )
        
        return response
    
    # 创建Gradio界面
    demo = gr.ChatInterface(
    fn=chat_function,
    title="🤖 AI 聊天机器人",
    description="基于大语言模型的智能对话助手",
    examples=[
        ["你好,请介绍一下你自己", 0.8, 50, 0.9, 100],
        ["今天天气怎么样?", 0.8, 50, 0.9, 100],
        ["请推荐几本好书", 0.8, 50, 0.9, 100],
        ["什么是人工智能?", 0.8, 50, 0.9, 100]
    ],
    additional_inputs=[
        gr.Slider(0.1, 2.0, value=0.8, step=0.1, label="Temperature (温度) - 控制创造性"),
        gr.Slider(1, 100, value=50, step=1, label="Top-k - 采样范围"),
        gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p - 累积概率"),
        gr.Slider(20, 200, value=100, step=10, label="Max Length - 最大生成长度")
    ],
    # theme=gr.themes.Soft()
    )
    
    return demo


class BatchInference:
    """批量推理处理"""
    
    def __init__(self, inference: ChatbotInference):
        """
        初始化批量推理器
        
        Args:
            inference: ChatbotInference实例
        """
        self.inference = inference
    
    def process_file(
        self,
        input_file: str,
        output_file: str,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        max_length: int = 100,
        repetition_penalty: float = 1.2
    ):
        """
        处理文件中的所有输入
        
        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径
            temperature: 温度参数
            top_k: top-k采样
            top_p: top-p采样
            max_length: 最大生成长度
            repetition_penalty: 重复惩罚
        """
        logger.info("="*60)
        logger.info("Batch Inference Mode")
        logger.info(f"Input file: {input_file}")
        logger.info(f"Output file: {output_file}")
        logger.info("="*60)
        
        # 读取输入
        inputs = self._load_inputs(input_file)
        logger.info(f"Loaded {len(inputs)} inputs\n")
        
        # 生成回复
        results = []
        
        for idx, item in enumerate(tqdm(inputs, desc="Processing")):
            try:
                # 获取输入文本
                if isinstance(item, dict):
                    user_input = item.get('query', item.get('input', ''))
                    reference = item.get('response', item.get('output', ''))
                else:
                    user_input = str(item)
                    reference = ''
                
                # 生成回复
                response = self.inference.generate_response(
                    user_input=user_input,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    max_length=max_length,
                    repetition_penalty=repetition_penalty,
                    use_history=False  # 批处理时不使用历史
                )
                
                results.append({
                    'id': idx,
                    'input': user_input,
                    'reference': reference,
                    'generated': response
                })
                
                # 打印示例
                if idx < 3:
                    logger.info(f"\n--- Sample {idx+1} ---")
                    logger.info(f"Input: {user_input}")
                    logger.info(f"Generated: {response}")
                
                # 清空历史
                self.inference.clear_history()
                
            except Exception as e:
                logger.error(f"Error processing item {idx}: {e}")
                results.append({
                    'id': idx,
                    'input': user_input if 'user_input' in locals() else '',
                    'reference': reference if 'reference' in locals() else '',
                    'generated': f"Error: {str(e)}"
                })
                continue
        
        # 保存结果
        self._save_results(results, output_file)
        
        logger.info("\n" + "="*60)
        logger.info(f"Batch processing completed!")
        logger.info(f"Processed {len(results)} items")
        logger.info(f"Results saved to: {output_file}")
        logger.info("="*60)
    
    def _load_inputs(self, input_file: str) -> List:
        """加载输入数据"""
        input_path = Path(input_file)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        # 根据文件格式加载
        if input_path.suffix == '.json':
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
        elif input_path.suffix == '.jsonl':
            data = []
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
                        
        elif input_path.suffix == '.txt':
            with open(input_file, 'r', encoding='utf-8') as f:
                data = [line.strip() for line in f if line.strip()]
                
        elif input_path.suffix == '.csv':
            try:
                import pandas as pd
                df = pd.read_csv(input_file)
                data = df.to_dict('records')
            except ImportError:
                logger.error("pandas is required for CSV files")
                raise
                
        else:
            raise ValueError(f"Unsupported file format: {input_path.suffix}")
        
        return data
    
    def _save_results(self, results: List[Dict], output_file: str):
        """保存结果"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 根据文件格式保存
        if output_path.suffix == '.json':
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
                
        elif output_path.suffix == '.jsonl':
            with open(output_file, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    
        elif output_path.suffix == '.csv':
            try:
                import pandas as pd
                df = pd.DataFrame(results)
                df.to_csv(output_file, index=False, encoding='utf-8-sig')
            except ImportError:
                logger.warning("pandas not available, saving as JSON")
                with open(output_file.replace('.csv', '.json'), 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
        else:
            # 默认保存为JSON
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Chatbot Inference - 支持LoRA模型"
    )
    
    # 模型参数
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Configuration file path'
    )
    parser.add_argument(
        '--base_model',
        type=str,
        required=True,
        help='Path to base model'
    )
    parser.add_argument(
        '--lora_weights',
        type=str,
        default=None,
        help='Path to LoRA weights (optional)'
    )
    
    # 运行模式
    parser.add_argument(
        '--mode',
        type=str,
        choices=['interactive', 'web', 'batch'],
        default='interactive',
        help='Inference mode'
    )
    
    # 批处理参数
    parser.add_argument(
        '--input_file',
        type=str,
        default=None,
        help='Input file for batch mode'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='inference_results.json',
        help='Output file for batch mode'
    )
    
    # 生成参数
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.8,
        help='Temperature for generation'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=50,
        help='Top-k for generation'
    )
    parser.add_argument(
        '--top_p',
        type=float,
        default=0.9,
        help='Top-p for generation'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=100,
        help='Maximum generation length'
    )
    
    # 设备参数
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda or cpu)'
    )
    parser.add_argument(
        '--no_fp16',
        action='store_true',
        help='Disable FP16'
    )
    
    # Web界面参数
    parser.add_argument(
        '--port',
        type=int,
        default=7860,
        help='Port for web interface'
    )
    parser.add_argument(
        '--share',
        action='store_true',
        help='Create public link for Gradio'
    )
    
    args = parser.parse_args()
    
    # 初始化推理器
    logger.info("\n" + "="*60)
    logger.info("Starting Chatbot Inference")
    logger.info("="*60 + "\n")
    
    inference = ChatbotInference(args.config)
    
    # 根据模式运行
    if args.mode == 'interactive':
        # 交互式对话
        inference.interactive_chat()
        
    elif args.mode == 'web':
        # Web界面
        logger.info(f"Starting Gradio web interface on port {args.port}...")
        demo = create_gradio_interface(inference)
        demo.launch(
            server_port=args.port,
            share=args.share,
            server_name="0.0.0.0"
        )
        
    elif args.mode == 'batch':
        # 批量处理
        if not args.input_file:
            logger.error("Batch mode requires --input_file")
            return
        
        batch_processor = BatchInference(inference)
        batch_processor.process_file(
            input_file=args.input_file,
            output_file=args.output_file,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            max_length=args.max_length
        )


if __name__ == "__main__":
    main()