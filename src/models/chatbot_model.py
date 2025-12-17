# -*- coding: utf-8 -*-
"""
改进的聊天机器人模型实现
支持多种预训练模型和生成策略
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple
from transformers import (
    AutoModel, 
    AutoTokenizer,
    AutoModelForCausalLM,
    BertModel,
    BertConfig,
    PreTrainedModel,
    PretrainedConfig
)
import logging

logger = logging.getLogger(__name__)


class ChatbotConfig(PretrainedConfig):
    """聊天机器人配置类"""
    
    def __init__(
        self,
        vocab_size: int = 21128,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        use_cache: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache


class ChatbotModel(PreTrainedModel):
    """聊天机器人模型"""
    
    def __init__(self, config: ChatbotConfig):
        super().__init__(config)
        self.config = config
        
        # 使用BERT作为编码器
        self.encoder = BertModel(config)
        
        # 解码器层
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(config) 
            for _ in range(config.num_hidden_layers // 2)
        ])
        
        # 输出投影层
        self.output_projection = nn.Linear(config.hidden_size, config.vocab_size)
        
        # 初始化权重
        self.init_weights()
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """前向传播"""
        
        # 编码输入
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        
        hidden_states = encoder_outputs.last_hidden_state
        
        # 如果提供了解码器输入
        if decoder_input_ids is not None:
            # 通过解码器层
            for layer in self.decoder_layers:
                hidden_states = layer(
                    hidden_states, 
                    encoder_outputs.last_hidden_state,
                    attention_mask
                )
        
        # 输出投影
        logits = self.output_projection(hidden_states)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1)
            )
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': hidden_states
        }
    
    def generate_response(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 50,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
        **kwargs
    ) -> torch.Tensor:
        """生成响应"""
        self.eval()
        
        with torch.no_grad():
            generated_ids = []
            current_input = input_ids
            
            for _ in range(max_length):
                outputs = self.forward(
                    input_ids=current_input,
                    attention_mask=attention_mask
                )
                
                next_token_logits = outputs['logits'][:, -1, :]
                
                # 应用repetition penalty
                if repetition_penalty != 1.0 and len(generated_ids) > 0:
                    for token_id in set(generated_ids):
                        next_token_logits[:, token_id] /= repetition_penalty
                
                # Temperature调整
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                
                # Top-k和Top-p过滤
                filtered_logits = self._top_k_top_p_filtering(
                    next_token_logits,
                    top_k=top_k,
                    top_p=top_p
                )
                
                # 采样
                probs = F.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated_ids.append(next_token.item())
                
                # 检查是否结束
                if next_token.item() == self.config.eos_token_id:
                    break
                
                # 更新输入
                current_input = torch.cat([current_input, next_token], dim=-1)
                
        return torch.tensor(generated_ids).unsqueeze(0)
    
    @staticmethod
    def _top_k_top_p_filtering(
        logits: torch.Tensor,
        top_k: int = 0,
        top_p: float = 0.0,
        filter_value: float = -float('Inf')
    ) -> torch.Tensor:
        """Top-k和Top-p过滤"""
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value
        
        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = filter_value
        
        return logits


class TransformerDecoderLayer(nn.Module):
    """Transformer解码器层"""
    
    def __init__(self, config: ChatbotConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            config.hidden_size,
            config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            config.hidden_size,
            config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            batch_first=True
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.intermediate_size, config.hidden_size)
        )
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm3 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """前向传播"""
        # 自注意力
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 交叉注意力
        cross_output, _ = self.cross_attn(x, encoder_output, encoder_output, key_padding_mask=src_mask)
        x = self.norm2(x + self.dropout(cross_output))
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


class ModernChatbot:
    """现代化的聊天机器人封装类"""
    
    def __init__(
        self,
        model_name_or_path: str = "bert-base-chinese",
        lora_path: Optional[str] = None,
        device: str = "cuda",
        use_fp16: bool = False,
        merge_lora: bool = False
    ):
        """
        初始化聊天机器人
        
        Args:
            model_name_or_path: 基础模型路径
            lora_path: LoRA 权重路径 (如果使用 LoRA 微调)
            device: 运行设备
            use_fp16: 是否使用半精度
            merge_lora: 是否合并 LoRA 权重到基础模型 (推理加速)
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_fp16 = use_fp16 and torch.cuda.is_available()
        
        # 加载tokenizer
        try:
            from modelscope import AutoTokenizer as MSAutoTokenizer
            self.tokenizer = MSAutoTokenizer.from_pretrained(
                model_name_or_path,
                trust_remote_code=True
            )
            logger.info(f"Loaded tokenizer from ModelScope: {model_name_or_path}")
        except:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                use_fast=True,
                trust_remote_code=True
            )
            logger.info(f"Loaded tokenizer from HuggingFace: {model_name_or_path}")
        
        # 加载模型
        self._load_model(model_name_or_path, lora_path, merge_lora)
        
        # 响应缓存
        self.response_cache = {}
        self.cache_size = 1000
    
    def _load_model(
        self, 
        model_name_or_path: str, 
        lora_path: Optional[str],
        merge_lora: bool
    ):
        """加载模型 (支持 LoRA)"""
        
        # 1. 加载基础模型
        try:
            from modelscope import AutoModel as MSAutoModel
            base_model = MSAutoModel.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.use_fp16 else torch.float32,
                device_map="auto"
            )
            logger.info(f"Loaded base model from ModelScope: {model_name_or_path}")
        except:
            try:
                base_model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if self.use_fp16 else torch.float32,
                    device_map="auto"
                )
                logger.info(f"Loaded base model from HuggingFace: {model_name_or_path}")
            except Exception as e:
                logger.warning(f"Failed to load pretrained model: {e}")
                logger.info("Falling back to custom model")
                config = ChatbotConfig()
                base_model = ChatbotModel(config)
        
        # 2. 如果提供了 LoRA 路径,加载 LoRA 权重
        if lora_path:
            try:
                from peft import PeftModel
                
                logger.info(f"Loading LoRA weights from: {lora_path}")
                self.model = PeftModel.from_pretrained(
                    base_model,
                    lora_path,
                    torch_dtype=torch.float16 if self.use_fp16 else torch.float32
                )
                
                # 3. 可选:合并 LoRA 权重以加速推理
                if merge_lora:
                    logger.info("Merging LoRA weights into base model...")
                    self.model = self.model.merge_and_unload()
                    logger.info("LoRA weights merged successfully")
                else:
                    logger.info("Using LoRA adapter (not merged)")
                
                logger.info(f"Successfully loaded LoRA model")
                
            except Exception as e:
                logger.error(f"Failed to load LoRA weights: {e}")
                logger.warning("Falling back to base model without LoRA")
                self.model = base_model
        else:
            self.model = base_model
            logger.info("Using base model (no LoRA)")
        
        # 4. 移动到设备
        if not hasattr(self.model, 'hf_device_map'):  # 如果没有使用 device_map="auto"
            self.model.to(self.device)
        
        # 5. 设置评估模式
        self.model.eval()
        
        # 6. 打印模型信息
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"="*60)
        logger.info(f"Model Statistics:")
        logger.info(f"  - Total parameters: {total_params:,}")
        logger.info(f"  - Trainable parameters: {trainable_params:,}")
        logger.info(f"  - Model dtype: {next(self.model.parameters()).dtype}")
        logger.info(f"  - Device: {next(self.model.parameters()).device}")
        logger.info(f"="*60)
    
    @torch.no_grad()
    def generate(
        self,
        text: str,
        max_length: int = 50,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
        use_cache: bool = True
    ) -> str:
        """生成回复"""
        
        # 检查缓存
        cache_key = f"{text}_{temperature}_{top_k}_{top_p}"
        if use_cache and cache_key in self.response_cache:
            return self.response_cache[cache_key]
        
        # 编码输入
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        ).to(self.device)
        
        # 生成响应
        try:
            if hasattr(self.model, 'generate'):
                # 使用Hugging Face的generate方法
                output_ids = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            else:
                # 使用自定义生成方法
                output_ids = self.model.generate_response(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty
                )
            
            # 解码响应
            response = self.tokenizer.decode(
                output_ids[0],
                skip_special_tokens=True
            )
            
            # 更新缓存
            if use_cache:
                if len(self.response_cache) >= self.cache_size:
                    # 移除最旧的缓存项
                    self.response_cache.pop(next(iter(self.response_cache)))
                self.response_cache[cache_key] = response
            
            return response
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return "抱歉，我现在有些困惑，能换个话题吗？"
    
    def clear_cache(self):
        """清空缓存"""
        self.response_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @classmethod
    def from_pretrained(
        cls,
        base_model_path: str,
        lora_checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        use_fp16: bool = True,
        merge_lora: bool = False
    ):
        """
        从预训练模型创建聊天机器人实例
        
        Args:
            base_model_path: 基础模型路径
            lora_checkpoint_path: LoRA 检查点路径 (best_model 或 checkpoint_epoch_N)
            device: 运行设备
            use_fp16: 是否使用半精度 (推荐用于推理)
            merge_lora: 是否合并 LoRA 权重 (推理加速,但会占用更多显存)
        
        Returns:
            ModernChatbot 实例
        
        Example:
            # 加载最佳模型
            chatbot = ModernChatbot.from_pretrained(
                base_model_path="/path/to/base/model",
                lora_checkpoint_path="/path/to/checkpoints/best_model",
                use_fp16=True,
                merge_lora=True
            )
        """
        return cls(
            model_name_or_path=base_model_path,
            lora_path=lora_checkpoint_path,
            device=device,
            use_fp16=use_fp16,
            merge_lora=merge_lora
        )
    
    @torch.no_grad()
    def evaluate_batch(
        self,
        texts: List[str],
        max_length: int = 50,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
        batch_size: int = 8
    ) -> List[str]:
        """
        批量生成回复 (用于评估)
        
        Args:
            texts: 输入文本列表
            max_length: 最大生成长度
            temperature: 温度参数
            top_k: Top-K 采样
            top_p: Top-P 采样
            repetition_penalty: 重复惩罚
            batch_size: 批处理大小
        
        Returns:
            生成的回复列表
        """
        responses = []
        
        # 分批处理
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # 编码
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            ).to(self.device)
            
            # 生成
            try:
                if hasattr(self.model, 'generate'):
                    output_ids = self.model.generate(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_length=max_length,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                else:
                    # 如果模型没有 generate 方法,使用自定义生成
                    output_ids = []
                    for j in range(len(batch_texts)):
                        single_output = self.model.generate_response(
                            inputs.input_ids[j:j+1],
                            attention_mask=inputs.attention_mask[j:j+1],
                            max_length=max_length,
                            temperature=temperature,
                            top_k=top_k,
                            top_p=top_p,
                            repetition_penalty=repetition_penalty
                        )
                        output_ids.append(single_output)
                    output_ids = torch.cat(output_ids, dim=0)
                
                # 解码
                batch_responses = self.tokenizer.batch_decode(
                    output_ids,
                    skip_special_tokens=True
                )
                responses.extend(batch_responses)
                
            except Exception as e:
                logger.error(f"Batch generation failed: {e}")
                responses.extend(["生成失败"] * len(batch_texts))
        
        return responses