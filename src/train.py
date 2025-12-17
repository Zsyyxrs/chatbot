# -*- coding: utf-8 -*-
import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import yaml

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from transformers import (
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from modelscope import AutoTokenizer, AutoModel, snapshot_download

from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training,
    PeftModel
)

from tqdm import tqdm
import wandb

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset import ChatDataset
from src.models.chatbot_model import ChatbotModel, ChatbotConfig
from src.utils.evaluator import Evaluator

# 设置日志
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class Trainer:
    """训练器 - LoRA 微调优化版"""
    
    def __init__(self, config_path: str):
        """初始化训练器"""
        
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # 清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc
            gc.collect()
        
        # 设置随机种子
        self._set_seed(self.config['training']['seed'])
        
        # 初始化模型和tokenizer
        self._initialize_model()
        
        # 初始化数据集
        self._initialize_datasets()
        
        # 初始化优化器和调度器
        self._initialize_optimizer()
        
        # 初始化评估器
        self.evaluator = Evaluator(language='chinese')
        
        # 混合精度训练（FP16 模型不需要 GradScaler）
        self.use_fp16 = False
        
        # 初始化wandb（可选）
        self.use_wandb = self.config.get('use_wandb', False)
        if self.use_wandb:
            wandb.init(
                project="chatbot-lora-training",
                config=self.config
            )
        
        # 最佳模型跟踪
        self.best_loss = float('inf')
        self.best_metrics = {}
        self.patience_counter = 0
        self.patience = self.config['training'].get('patience', 5)
        
        # 当前 epoch
        self.current_epoch = 0
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 确保所有数值参数是正确类型
        if 'training' in config:
            numeric_fields = ['learning_rate', 'weight_decay', 'warmup_ratio', 'max_grad_norm']
            for field in numeric_fields:
                if field in config['training']:
                    config['training'][field] = float(config['training'][field])
        
        return config
    
    def _set_seed(self, seed: int):
        """设置随机种子"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        
        # ✅ 启用性能优化
        if torch.cuda.is_available():
            # 检查是否支持 TF32（Ampere 及以上 GPU）
            if torch.cuda.get_device_capability()[0] >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info("✓ TF32 enabled (Ampere+ GPU)")
            else:
                logger.info("✗ TF32 not available (GPU compute capability < 8.0)")
                
            # 启用 cuDNN benchmark
            torch.backends.cudnn.benchmark = True
            logger.info("✓ cuDNN benchmark enabled")
    
    def _initialize_model(self):
        """初始化模型和tokenizer（使用 LoRA）"""
        self.model_dir = self.config['model']['dir']
        
        # 加载基础模型（FP16 以节省显存）
        logger.info(f"Loading base model from {self.model_dir}...")
        base_model = AutoModel.from_pretrained(
            self.model_dir,
            trust_remote_code=True,
            dtype=torch.float16,  
            device_map="auto"
        )
        
        logger.info(f"Base model loaded in {next(base_model.parameters()).dtype}")
        
        # ✅ 梯度检查点配置
        use_gradient_checkpointing = self.config['training'].get('gradient_checkpointing', False)
        
        if use_gradient_checkpointing:
            if hasattr(base_model, 'gradient_checkpointing_enable'):
                if hasattr(base_model.config, 'use_cache'):
                    base_model.config.use_cache = False
                
                base_model.gradient_checkpointing_enable()
                
                try:
                    if hasattr(base_model, 'enable_input_require_grads'):
                        base_model.enable_input_require_grads()
                except Exception:
                    pass
                
                logger.info("✓ Gradient checkpointing enabled (slower but saves memory)")
                logger.info("  - use_cache=False")
        else:
            if hasattr(base_model.config, 'use_cache'):
                base_model.config.use_cache = True
            
            logger.info("✓ Gradient checkpointing disabled (faster training)")
            logger.info("  - use_cache=True (enabled for speed)")
        
        # 配置 LoRA
        lora_config = LoraConfig(
            r=self.config['lora'].get('r', 8),
            lora_alpha=self.config['lora'].get('lora_alpha', 32),
            target_modules=self.config['lora'].get('target_modules', ["query_key_value"]),
            lora_dropout=self.config['lora'].get('lora_dropout', 0.1),
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        # 应用 LoRA
        self.model = get_peft_model(base_model, lora_config)
        
        # 打印可训练参数统计
        trainable_params = 0
        all_params = 0
        for _, param in self.model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        logger.info(f"="*60)
        logger.info(f"LoRA Configuration:")
        logger.info(f"  - Rank (r): {lora_config.r}")
        logger.info(f"  - Alpha: {lora_config.lora_alpha}")
        logger.info(f"  - Target modules: {lora_config.target_modules}")
        logger.info(f"  - Dropout: {lora_config.lora_dropout}")
        logger.info(f"="*60)
        logger.info(f"Model Statistics:")
        logger.info(f"  - Total parameters: {all_params:,}")
        logger.info(f"  - Trainable parameters: {trainable_params:,}")
        logger.info(f"  - Trainable %: {100 * trainable_params / all_params:.2f}%")
        logger.info(f"="*60)
        
        # 显存统计
        if torch.cuda.is_available():
            logger.info(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            logger.info(f"GPU Memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    def _initialize_datasets(self):
        """初始化数据集"""
        data_config = self.config['data']
        
        # 训练集
        self.train_dataset = ChatDataset(
            data_path=data_config['train_file'],
            tokenizer_path=self.model_dir,
            max_length=self.config['model']['max_length'],
            cache_dir=data_config.get('cache_dir'),
            augment=data_config.get('augment', False)
        )
        
        # 性能优化的 DataLoader 配置
        self.train_loader = self.train_dataset.get_dataloader(
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=data_config.get('num_workers', 16), 
            pin_memory=True,
            persistent_workers=True
        )
        
        # 验证集
        self.val_dataset = ChatDataset(
            data_path=data_config['val_file'],
            tokenizer_path=self.model_dir,
            max_length=self.config['model']['max_length'],
            cache_dir=data_config.get('cache_dir'),
            augment=False
        )
        
        # 验证时用更大的 batch
        self.val_loader = self.val_dataset.get_dataloader(
            batch_size=self.config['training']['batch_size'] * 4,
            shuffle=False,
            num_workers=data_config.get('num_workers', 16),
            pin_memory=True,
            persistent_workers=True
        )
        
        logger.info(f"Train dataset size: {len(self.train_dataset)}")
        logger.info(f"Val dataset size: {len(self.val_dataset)}")
        logger.info(f"Train batches per epoch: {len(self.train_loader)}")
    
    def _initialize_optimizer(self):
        """初始化优化器和学习率调度器"""
        training_config = self.config['training']
        
        # 参数分组（对不同层使用不同学习率）
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if p.requires_grad and not any(nd in n for nd in no_decay)],
                'weight_decay': float(training_config.get('weight_decay', 0.01))
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if p.requires_grad and any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        # 优化器
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=float(training_config['learning_rate']),
            eps=1e-8
        )
        
        # 学习率调度器
        num_training_steps = (
            len(self.train_loader) // training_config.get('gradient_accumulation_steps', 1)
            * training_config['num_epochs']
        )
        
        num_warmup_steps = int(num_training_steps * float(training_config.get('warmup_ratio', 0.1)))
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        logger.info(f"Total training steps: {num_training_steps}")
        logger.info(f"Warmup steps: {num_warmup_steps}")
    
    def train(self):
        """训练主循环"""
        
        training_config = self.config['training']
        best_model_path = None
        
        for epoch in range(training_config['num_epochs']):
            self.current_epoch = epoch + 1
            
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch + 1}/{training_config['num_epochs']}")
            logger.info(f"{'='*50}")
            
            # 训练
            train_loss = self._train_epoch()
            
            # 验证
            val_loss, val_metrics = self._validate()
            
            # 记录
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}")
            
            for key, value in val_metrics.items():
                logger.info(f"Val {key}: {value:.4f}")
            
            # wandb记录
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    **{f'val_{k}': v for k, v in val_metrics.items()},
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
            
            # 保存最佳模型
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_metrics = val_metrics
                self.patience_counter = 0
                
                # 保存模型
                best_model_path = self._save_model(
                    epoch=epoch + 1,
                    loss=val_loss,
                    metrics=val_metrics,
                    is_best=True
                )
                
                logger.info(f"✓ New best model saved: {best_model_path}")
            else:
                self.patience_counter += 1
                
                # 早停
                if self.patience_counter >= self.patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
            
            # 定期保存检查点
            if (epoch + 1) % training_config.get('save_epochs', 5) == 0:
                self._save_model(
                    epoch=epoch + 1,
                    loss=val_loss,
                    metrics=val_metrics,
                    is_best=False
                )
        
        # 训练结束
        logger.info("\n" + "="*50)
        logger.info("Training completed!")
        logger.info(f"Best Val Loss: {self.best_loss:.4f}")
        logger.info(f"Best Model: {best_model_path}")
        logger.info("="*50)
    
    def _train_epoch(self) -> float:
        """训练一个 epoch（完整 NaN 防护）"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        nan_count = 0  # ✅ 统计 NaN 次数
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch}/{self.config['training']['num_epochs']}"
        )
        
        accumulation_steps = self.config['training'].get('gradient_accumulation_steps', 1)
        logging_steps = self.config['training'].get('logging_steps', 50)
        
        for step, batch in enumerate(progress_bar):
            # ✅ 1. 首先检查模型参数是否已经是 NaN
            # if self._check_model_nan():
            #     print(f"\n❌ 模型参数在 step {step} 变成 NaN，停止训练！")
            #     print("建议：降低学习率或检查数据")
            #     break
            
            # 将数据移到设备
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # ✅ 2. 检查输入数据
            # if any(torch.isnan(v).any() or torch.isinf(v).any() for v in batch.values()):
            #     print(f"\n⚠️ Warning: 输入数据在 step {step} 包含 NaN/Inf，跳过...")
            #     nan_count += 1
            #     continue
            
            try:
                # 前向传播
                outputs = self.model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs['loss']
                
                # ✅ 3. 检查损失值（在除法之前）
                if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 1e5:
                    print(f"\n⚠️ Warning: 异常 loss={loss.item():.4f} at step {step}")
                    nan_count += 1
                    
                    # ✅ 清空梯度，避免污染
                    self.optimizer.zero_grad()
                    
                    # ✅ 如果连续多次 NaN，停止训练
                    if nan_count >= 10:
                        print(f"\n❌ 连续 {nan_count} 次 NaN，停止训练！")
                        break
                    continue
                
                loss = loss / accumulation_steps
                
                # 反向传播
                loss.backward()
                
                # ✅ 4. 梯度累积后检查梯度
                if (step + 1) % accumulation_steps == 0:
                    # 检查梯度范数
                    total_norm = 0.0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                            
                            # ✅ 检查单个参数梯度
                            if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                                print(f"\n⚠️ Warning: 参数梯度包含 NaN/Inf at step {step}")
                                self.optimizer.zero_grad()
                                nan_count += 1
                                break
                    else:
                        # 只有当所有梯度都正常时才执行优化
                        total_norm = total_norm ** 0.5
                        
                        # ✅ 记录异常大的梯度
                        if total_norm > 100:
                            print(f"\n⚠️ 梯度范数过大: {total_norm:.2f} at step {step}")
                        
                        # 梯度裁剪
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            float(self.config['training'].get('max_grad_norm', 1.0))
                        )
                        
                        # 优化器步骤
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                        
                        # ✅ 重置 NaN 计数
                        nan_count = 0
            
            except RuntimeError as e:
                print(f"\n❌ Runtime error at step {step}: {e}")
                self.optimizer.zero_grad()
                nan_count += 1
                if nan_count >= 10:
                    break
                continue
            
            # 记录损失
            total_loss += loss.item() * accumulation_steps
            num_batches += 1
            
            if step % logging_steps == 0:
                progress_bar.set_postfix({
                    'loss': f'{loss.item() * accumulation_steps:.4f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                    'nan_skip': nan_count,
                    'mem': f'{torch.cuda.memory_allocated() / 1024**3:.1f}GB' if torch.cuda.is_available() else 'N/A'
                })
        
        if num_batches == 0:
            print("\n⚠️ Warning: 本 epoch 没有有效的 batch！")
            return float('inf')
        
        return total_loss / num_batches

    def _check_model_nan(self) -> bool:
        """检查模型参数是否包含 NaN"""
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"参数 {name} 包含 NaN/Inf")
                return True
        return False
    
    @torch.no_grad()
    def _validate(self) -> Tuple[float, Dict[str, float]]:
        """验证模型"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(self.val_loader, desc="Validating")
        
        for batch in progress_bar:
            # 将数据移到设备
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # 前向传播
            outputs = self.model(**batch)
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs['loss']
            
            # 记录损失
            total_loss += loss.item()
            num_batches += 1
            
            # 更新进度条
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # 计算指标
        avg_loss = total_loss / num_batches
        
        # 这里可以添加更多评估指标（BLEU, ROUGE 等）
        metrics = {}
        
        return avg_loss, metrics
    
    def _save_model(
        self,
        epoch: int,
        loss: float,
        metrics: Dict[str, float],
        is_best: bool
    ) -> str:
        """保存模型（LoRA 权重）"""
        
        output_dir = Path(self.config['paths']['checkpoint_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 文件名
        if is_best:
            model_dir = output_dir / "best_model"
        else:
            model_dir = output_dir / f"checkpoint_epoch_{epoch}"
        
        model_dir.mkdir(exist_ok=True)
        
        # 保存 LoRA 权重
        self.model.save_pretrained(model_dir)
        
        # 保存训练状态
        state_path = model_dir / "training_state.pt"
        torch.save({
            'epoch': epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'metrics': metrics,
            'config': self.config
        }, state_path)
        
        logger.info(f"LoRA weights saved to: {model_dir}")
        logger.info(f"Training state saved to: {state_path}")
        
        return str(model_dir)
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint_path = Path(checkpoint_path)
        
        # 加载 LoRA 权重
        if checkpoint_path.is_dir():
            self.model = PeftModel.from_pretrained(
                self.model.base_model,
                checkpoint_path
            )
            logger.info(f"Loaded LoRA weights from {checkpoint_path}")
            
            # 加载训练状态
            state_path = checkpoint_path / "training_state.pt"
            if state_path.exists():
                state = torch.load(state_path, map_location=self.device)
                self.optimizer.load_state_dict(state['optimizer_state_dict'])
                self.scheduler.load_state_dict(state['scheduler_state_dict'])
                logger.info(f"Loaded training state: Epoch {state['epoch']}, Loss {state['loss']:.4f}")
                return state
        else:
            logger.error(f"Checkpoint directory not found: {checkpoint_path}")
        
        return None


def main():
    """主函数"""
    
    parser = argparse.ArgumentParser(description="Train chatbot with LoRA")
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    args = parser.parse_args()
    
    # 初始化训练器
    trainer = Trainer(args.config)
    
    # 恢复训练
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()