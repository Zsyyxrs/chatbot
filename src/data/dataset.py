# -*- coding: utf-8 -*-
"""
改进的数据集实现
包含数据缓存、批处理优化和数据增强
✅ 修复多进程数据加载问题
✅ 修复ChatGLM tokenizer兼容性问题
"""

import json
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import logging
from tqdm import tqdm
import pickle
from src.utils.filter import DirtyFilter
import re
import hashlib

logger = logging.getLogger(__name__)


class ChatDataset(Dataset):
    """聊天数据集（优化版 - 支持多进程 + ChatGLM兼容）"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer_path: str,
        max_length: int = 256,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        augment: bool = False,
        augment_seed: int = 42  
    ):
        self.data_path = Path(data_path)
        self.tokenizer_path = tokenizer_path
        self._tokenizer = None
        self.max_length = max_length
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.augment = augment
        self.augment_seed = augment_seed
        
        # 数据增强和缓存互斥
        if augment and use_cache:
            logger.warning("⚠️  数据增强模式：自动禁用缓存以保证随机性")
            use_cache = False
        
        self.use_cache = use_cache
        
        if augment:
            self.rng = np.random.RandomState(augment_seed)
            logger.info(f"✓ 数据增强已启用 (seed={augment_seed})")
        else:
            self.rng = None
        
        # 加载数据
        self.data = self._load_data()
        
        # 缓存相关
        self.cache: Dict[int, Any] = {}
        self._cache_dirty = False
        self._last_save_size = 0
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_key = self._get_cache_key()
            self.cache_file = self.cache_dir / f"{cache_key}_cache.pkl"
            
            if self.cache_file.exists() and self.use_cache:
                self._load_cache()
    
    @property
    def tokenizer(self):
        """延迟加载 tokenizer，每个 worker 进程独立加载"""
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            logger.info(f"Loading tokenizer from {self.tokenizer_path}")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_path,
                trust_remote_code=True
            )
            
            # ✅ 确保有 pad_token（ChatGLM 可能没有）
            if self._tokenizer.pad_token is None:
                if self._tokenizer.eos_token:
                    self._tokenizer.pad_token = self._tokenizer.eos_token
                    logger.info(f"Set pad_token to eos_token: {self._tokenizer.eos_token}")
                else:
                    self._tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    logger.info("Added [PAD] as pad_token")
            
            logger.info(f"Tokenizer loaded. pad_token_id: {self._tokenizer.pad_token_id}")
        
        return self._tokenizer
    
    def _get_cache_key(self) -> str:
        """生成缓存key"""
        key_string = (
            f"{self.data_path}_{self.max_length}_"
            f"{self.tokenizer_path}"
        )
        return hashlib.md5(key_string.encode()).hexdigest()[:16]
    
    def _load_data(self) -> pd.DataFrame:
        """加载数据"""
        try:
            suffix = self.data_path.suffix.lower()
            
            if suffix in ['.jsonl', '.json']:
                try:
                    data = pd.read_json(self.data_path, lines=True, encoding='utf-8')
                except ValueError:
                    logger.info("尝试标准 JSON 格式...")
                    data = pd.read_json(self.data_path, encoding='utf-8')
                    
            elif suffix == '.csv':
                data = pd.read_csv(self.data_path, encoding='utf-8')
            elif suffix == '.tsv':
                data = pd.read_csv(self.data_path, sep='\t', encoding='utf-8')
            else:
                data_list = []
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    for line in tqdm(f, desc="Loading data"):
                        line = line.strip()
                        if line:
                            try:
                                data_list.append(json.loads(line))
                            except json.JSONDecodeError as e:
                                logger.warning(f"Failed to parse line: {line[:50]}... Error: {e}")
                
                if not data_list:
                    raise ValueError("No valid data loaded")
                data = pd.DataFrame(data_list)
            
            logger.info(f"Loaded {len(data)} samples from {self.data_path}")
            
            # 数据验证
            required_columns = ['src_text', 'tgt_text']
            missing_cols = [col for col in required_columns if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            original_len = len(data)
            
            # 清理空值和空字符串
            data = data.dropna(subset=required_columns)
            for col in required_columns:
                data = data[data[col].astype(str).str.strip() != '']
            
            data = data.reset_index(drop=True)
            
            dropped = original_len - len(data)
            if dropped > 0:
                logger.info(f"Dropped {dropped} invalid samples ({dropped/original_len*100:.1f}%)")
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to load data from {self.data_path}: {e}")
            raise
    
    def _load_cache(self):
        """加载缓存"""
        try:
            with open(self.cache_file, 'rb') as f:
                self.cache = pickle.load(f)
            self._last_save_size = len(self.cache)
            logger.info(f"Loaded cache with {len(self.cache)} items")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            self.cache = {}
            self._last_save_size = 0
    
    def _save_cache(self):
        """保存缓存"""
        if self.cache_dir and self.cache:
            try:
                with open(self.cache_file, 'wb') as f:
                    pickle.dump(self.cache, f)
                logger.info(f"Saved cache with {len(self.cache)} items")
                self._cache_dirty = False
            except Exception as e:
                logger.warning(f"Failed to save cache: {e}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        """获取单个样本 - 避免 ChatGLM tokenizer bug"""
        
        if not self.augment and idx in self.cache:
            return self.cache[idx]
        
        try:
            row = self.data.iloc[idx]
            src_text = str(row.get('src_text', ''))
            tgt_text = str(row.get('tgt_text', ''))
            
            if not src_text.strip() or not tgt_text.strip():
                raise ValueError(f"Empty text at idx={idx}")
            
        except Exception as e:
            logger.error(f"Data loading failed for idx={idx}: {e}")
            return self._get_fallback_item()
        
        if self.augment and self.rng is not None:
            try:
                src_text, tgt_text = self._augment_data(src_text, tgt_text)
            except:
                pass
        
        try:
            # ✅ 使用 tokenize 而不是 encode（避免 padding bug）
            # ChatGLM3 格式：<|user|>\n{question}<|assistant|>\n{answer}
            prompt = f"<|user|>\n{src_text}<|assistant|>\n"
            full_text = prompt + tgt_text
            
            # Tokenize（不会触发 padding）
            prompt_tokens = self.tokenizer.tokenize(prompt)
            full_tokens = self.tokenizer.tokenize(full_text)
            
            # 转换为 IDs
            prompt_ids = self.tokenizer.convert_tokens_to_ids(prompt_tokens)
            full_ids = self.tokenizer.convert_tokens_to_ids(full_tokens)
            
            # 验证
            if not prompt_ids or not full_ids:
                raise ValueError("Tokenization returned empty")
            
            # ✅ 添加 ChatGLM3 特殊 tokens
            # [gMASK]=64790, sop=64792
            input_ids = [64790, 64792] + full_ids
            prompt_length = 2 + len(prompt_ids)
            
            # 添加 EOS (</s>=2)
            input_ids.append(2)
            
            # 截断
            if len(input_ids) > self.max_length:
                input_ids = input_ids[:self.max_length]
                # 确保有答案空间
                min_answer_length = max(10, int(self.max_length * 0.1))
                if prompt_length >= len(input_ids) - min_answer_length:
                    prompt_length = len(input_ids) - min_answer_length
            
            # 生成 labels
            labels = input_ids.copy()
            for i in range(min(prompt_length, len(labels))):
                labels[i] = -100
            
            # 验证至少有一个有效标签
            valid_count = sum(1 for x in labels if x != -100)
            if valid_count == 0 and len(labels) > 0:
                labels[-1] = input_ids[-1]
            
            # 类型转换
            input_ids = [int(x) for x in input_ids]
            labels = [int(x) for x in labels]
            
            # 最终检查
            if len(input_ids) != len(labels) or not input_ids:
                raise ValueError(f"Invalid result: len(input)={len(input_ids)}, len(labels)={len(labels)}")
            
        except Exception as e:
            logger.error(f"Processing failed for idx={idx}: {e}")
            logger.error(f"  src: {src_text[:50] if src_text else 'None'}...")
            logger.error(f"  tgt: {tgt_text[:50] if tgt_text else 'None'}...")
            return self._get_fallback_item()
        
        item = {
            'input_ids': input_ids,
            'labels': labels
        }
        
        # 缓存逻辑
        if self.use_cache and not self.augment:
            self.cache[idx] = item
            self._cache_dirty = True
            
            cache_growth = len(self.cache) - self._last_save_size
            if cache_growth >= 5000 or (self._last_save_size > 0 and cache_growth / self._last_save_size > 0.2):
                self._save_cache()
                self._last_save_size = len(self.cache)
        
        return item

    def _get_fallback_item(self) -> Dict[str, List[int]]:
        """返回有效的 fallback item"""
        # [gMASK], sop, <unk>
        return {
            'input_ids': [64790, 64792, 0],
            'labels': [-100, -100, 0]
        }
    
    def _augment_data(self, src_text: str, tgt_text: str) -> Tuple[str, str]:
        """数据增强"""
        if self.rng.random() < 0.1:
            src_text = self._add_typos(src_text)
        
        if self.rng.random() < 0.1:
            tgt_text = self._synonym_replacement(tgt_text)
        
        return src_text, tgt_text
    
    def _add_typos(self, text: str) -> str:
        """添加拼写错误"""
        chars = list(text)
        if len(chars) > 3:
            idx = self.rng.randint(0, len(chars) - 1)
            chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        return ''.join(chars)
    
    def _synonym_replacement(self, text: str) -> str:
        """同义词替换"""
        simple_synonyms = {
            '好': ['不错', '很好', '优秀'],
            '棒': ['厉害', '出色', '优秀'],
            '开心': ['高兴', '快乐', '愉快']
        }
        
        for word, synonyms in simple_synonyms.items():
            if word in text and self.rng.random() < 0.5:
                replacement = self.rng.choice(synonyms)
                text = text.replace(word, replacement, 1)
        
        return text
    
    def get_dataloader(
        self,
        batch_size: int = 16,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True
    ) -> DataLoader:
        """创建DataLoader"""
        import platform
        if platform.system() == 'Windows':
            num_workers = 0
            persistent_workers = False
            logger.warning("Windows系统：自动设置 num_workers=0")
        
        # ✅ 提前获取 pad_token_id，避免在 worker 进程中访问 tokenizer
        pad_token_id = self.tokenizer.pad_token_id
        max_length = self.max_length
        
        # ✅ 关键验证：确保 pad_token_id 不是 None
        if pad_token_id is None:
            logger.error("pad_token_id is None! Using 0 as fallback")
            pad_token_id = 0
        
        logger.info(f"DataLoader config: pad_token_id={pad_token_id}, max_length={max_length}")
        
        # ✅ 创建 collate_fn 工厂函数
        def collate_fn(batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
            """
            批处理函数：动态padding到batch内最大长度
            ✅ 解决ChatGLM tokenizer的padding_side问题
            """
            # ✅ 调试和验证
            if not batch:
                raise ValueError("Empty batch received in collate_fn")
            
            # ✅ 检查每个 item 的有效性
            for i, item in enumerate(batch):
                if not isinstance(item, dict):
                    raise TypeError(f"Batch item {i} is not a dict: {type(item)}")
                if 'input_ids' not in item or 'labels' not in item:
                    raise KeyError(f"Batch item {i} missing keys. Keys: {item.keys()}")
                if not isinstance(item['input_ids'], list):
                    raise TypeError(f"Batch item {i} input_ids is not a list: {type(item['input_ids'])}")
                if not isinstance(item['labels'], list):
                    raise TypeError(f"Batch item {i} labels is not a list: {type(item['labels'])}")
                if len(item['input_ids']) == 0:
                    logger.warning(f"Batch item {i} has empty input_ids, using fallback")
                    item['input_ids'] = [pad_token_id]
                if len(item['labels']) == 0:
                    logger.warning(f"Batch item {i} has empty labels, using fallback")
                    item['labels'] = [pad_token_id]
                
                # ✅ 确保所有元素都是整数
                item['input_ids'] = [int(x) if x is not None else pad_token_id for x in item['input_ids']]
                item['labels'] = [int(x) if x is not None else pad_token_id for x in item['labels']]
            
            # 获取 batch 中最长序列的长度
            # ✅ 对于 decoder-only 模型，input 和 labels 应该使用相同的长度
            max_len = max(
                max(len(item['input_ids']) for item in batch),
                max(len(item['labels']) for item in batch)
            )
            
            # 限制最大长度
            max_len = min(max_len, max_length)
            
            input_ids_list = []
            attention_mask_list = []
            labels_list = []
            # 移除 decoder_attention_mask_list，ChatGLM 不需要
            
            for item in batch:
                src_ids = item['input_ids'][:max_len]  # 截断到统一长度
                tgt_ids = item['labels'][:max_len]
                
                # Padding source
                src_len = len(src_ids)
                src_pad_len = max_len - src_len
                padded_src_ids = src_ids + [pad_token_id] * src_pad_len
                src_mask = [1] * src_len + [0] * src_pad_len
                
                # Padding target (labels)
                tgt_len = len(tgt_ids)
                tgt_pad_len = max_len - tgt_len
                padded_tgt_ids = tgt_ids + [pad_token_id] * tgt_pad_len
                
                input_ids_list.append(padded_src_ids)
                attention_mask_list.append(src_mask)
                labels_list.append(padded_tgt_ids)
            
            # ✅ 最后验证
            try:
                # ✅ ChatGLM 不需要 decoder_attention_mask
                return {
                    'input_ids': torch.tensor(input_ids_list, dtype=torch.long),
                    'attention_mask': torch.tensor(attention_mask_list, dtype=torch.long),
                    'labels': torch.tensor(labels_list, dtype=torch.long)
                    # 移除 decoder_attention_mask
                }
            except Exception as e:
                logger.error(f"Failed to create tensors in collate_fn: {e}")
                logger.error(f"input_ids_list[0]: {input_ids_list[0] if input_ids_list else 'empty'}")
                logger.error(f"pad_token_id: {pad_token_id}")
                raise
        
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            persistent_workers=persistent_workers if num_workers > 0 else False,
            collate_fn=collate_fn,  # ✅ 使用闭包，捕获 pad_token_id 和 max_length
            prefetch_factor=2 if num_workers > 0 else None
        )
    
    def __del__(self):
        """析构函数：保存未保存的缓存"""
        if hasattr(self, '_cache_dirty') and self._cache_dirty and hasattr(self, 'cache') and self.cache:
            try:
                self._save_cache()
            except Exception as e:
                logger.warning(f"Failed to save cache in destructor: {e}")


class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self, dirty_filter: Optional[DirtyFilter] = None):
        self.dirty_filter = dirty_filter or DirtyFilter()
        
        # 预编译正则表达式
        self._emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"
            u"\U0001F300-\U0001F5FF"
            u"\U0001F680-\U0001F6FF"
            u"\U0001F1E0-\U0001F1FF"
            u"\U00002600-\U000026FF"
            u"\U00002700-\U000027BF"
            u"\U0001F900-\U0001F9FF"
            u"\U0001FA00-\U0001FA6F"
            u"\U0001FA70-\U0001FAFF"
            u"\U00002300-\U000023FF"
            "]+",
            flags=re.UNICODE
        )
        
        # 低质量回复模式
        self._low_quality_patterns = [
            re.compile(r'^谢谢+[。！]*$'),
            re.compile(r'^好的+[。！]*$'),
            re.compile(r'^嗯+[。！]*$'),
            re.compile(r'^哈哈+[。！]*$'),
        ]
    
    def clean_text(self, text: str) -> str:
        """清理文本"""
        if not text:
            return ""
        
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        text = re.sub(r'[!！]{2,}', '！', text)
        text = re.sub(r'[?？]{2,}', '？', text)
        text = re.sub(r'[.。]{3,}', '...', text)
        
        return text
    
    def contains_dirty_words(self, text: str) -> bool:
        """检查是否包含敏感词"""
        if not text:
            return False
        if self.dirty_filter:
            return self.dirty_filter.check(text)
        return False
    
    def process_dialogue_data(
        self,
        input_file: str,
        output_file: str,
        min_length: int = 2,
        max_length: int = 256,
        remove_emoji: bool = True,
        strict_filter: bool = True,
        batch_size: int = 1000 
    ):
        """处理对话数据（批处理优化）"""
        
        kuakua_triggers = [
            "大家来留言吧！我来夸你们", "求表扬", "有人夸我吗", "求安慰",
            "求祝福", "能被表扬吗", "求夸奖", "求鼓励", "来表扬我一下好吗",
            "求夸", "我好棒啊", "球表演", "求彩虹屁", "快来夸我嘛",
            "快来夸夸我", "再来夸一次哈哈"
        ]
        
        processed_data = []
        skipped_stats = {
            'incomplete': 0,
            'too_short': 0,
            'too_long': 0,
            'low_quality': 0,
            'dirty_words': 0,
            'duplicates': 0
        }
        seen_pairs = set()
        current_question = None
        current_answers = []
        
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        qa_pairs_batch = []
        
        for line in tqdm(lines, desc="Processing dialogues"):
            line = line.strip()
            
            if line.startswith('Q:') or line.startswith('Q：'):
                if current_question and current_answers:
                    for answer in current_answers:
                        qa_pairs_batch.append((current_question, answer))
                        
                        if len(qa_pairs_batch) >= batch_size:
                            self._process_batch(
                                qa_pairs_batch,
                                kuakua_triggers,
                                min_length,
                                max_length,
                                remove_emoji,
                                strict_filter,
                                seen_pairs,
                                skipped_stats,
                                processed_data
                            )
                            qa_pairs_batch = []
                
                current_question = line.split('\t', 1)[1] if '\t' in line else line[2:].strip()
                current_answers = []
                
            elif line.startswith('A:') or line.startswith('A：'):
                answer = line.split('\t', 1)[1] if '\t' in line else line[2:].strip()
                if answer:
                    current_answers.append(answer)
        
        if current_question and current_answers:
            for answer in current_answers:
                qa_pairs_batch.append((current_question, answer))
        
        if qa_pairs_batch:
            self._process_batch(
                qa_pairs_batch,
                kuakua_triggers,
                min_length,
                max_length,
                remove_emoji,
                strict_filter,
                seen_pairs,
                skipped_stats,
                processed_data
            )
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in processed_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        total_skipped = sum(skipped_stats.values())
        total_processed = len(processed_data) + total_skipped
        
        logger.info(f"""
        ================== Data Processing Report ==================
        Total pairs processed: {len(processed_data)}
        
        Filtering Statistics:
        - Incomplete questions (...): {skipped_stats['incomplete']}
        - Too short (< {min_length}): {skipped_stats['too_short']}
        - Low quality replies: {skipped_stats['low_quality']}
        - Contains dirty words: {skipped_stats['dirty_words']}
        - Duplicates: {skipped_stats['duplicates']}
        
        Total skipped: {total_skipped}
        Success rate: {len(processed_data) / total_processed * 100:.1f}%
        ============================================================
        """)
        
        return processed_data
    
    def _process_batch(
        self,
        qa_pairs: List[Tuple[str, str]],
        kuakua_triggers: List[str],
        min_length: int,
        max_length: int,
        remove_emoji: bool,
        strict_filter: bool,
        seen_pairs: set,
        skipped_stats: dict,
        processed_data: list
    ):
        """批处理问答对"""
        texts_to_check = []
        for question, answer in qa_pairs:
            texts_to_check.append(question)
            texts_to_check.append(answer)
        
        dirty_flags = [self.contains_dirty_words(text) for text in texts_to_check]
        
        for i, (question, answer) in enumerate(qa_pairs):
            has_dirty_q = dirty_flags[i * 2]
            has_dirty_a = dirty_flags[i * 2 + 1]
            
            if has_dirty_q or has_dirty_a:
                skipped_stats['dirty_words'] += 1
                continue
            
            processed_pair = self._process_qa_pair(
                question,
                answer,
                kuakua_triggers,
                min_length,
                max_length,
                remove_emoji,
                strict_filter,
                seen_pairs,
                skipped_stats
            )
            
            if processed_pair:
                processed_data.append(processed_pair)
    
    def _process_qa_pair(
        self,
        question: str,
        answer: str,
        kuakua_triggers: list,
        min_length: int,
        max_length: int,
        remove_emoji: bool,
        strict_filter: bool,
        seen_pairs: set,
        skipped_stats: dict
    ) -> Optional[dict]:
        """处理单个问答对"""
        
        if strict_filter and "..." in question:
            skipped_stats['incomplete'] += 1
            return None
        
        for trigger in kuakua_triggers:
            question = question.replace(trigger, "")
        
        question = self.clean_text(question)
        answer = self.clean_text(answer)
        
        if remove_emoji:
            question = self._remove_emojis(question)
            answer = self._remove_emojis(answer)
        
        if strict_filter:
            for pattern in self._low_quality_patterns:
                if pattern.match(answer):
                    skipped_stats['low_quality'] += 1
                    return None
            
            if len(answer) <= 10 and any(word in answer for word in ['谢谢', '感谢', '多谢']):
                skipped_stats['low_quality'] += 1
                return None
        
        if len(question) < min_length or len(answer) < min_length:
            skipped_stats['too_short'] += 1
            return None
        
        if len(question) > max_length:
            question = question[:max_length]
        if len(answer) > max_length:
            answer = answer[:max_length]
        
        pair_id = f"{question}|||{answer}"
        if pair_id in seen_pairs:
            skipped_stats['duplicates'] += 1
            return None
        seen_pairs.add(pair_id)
        
        return {
            'src_text': question,
            'tgt_text': answer
        }
    
    def _remove_emojis(self, text: str) -> str:
        """去除emoji表情"""
        return self._emoji_pattern.sub(r'', text)
    
    def split_dataset(
        self,
        data_file: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42
    ) -> Tuple[List[dict], List[dict], List[dict]]:
        """划分数据集"""
        np.random.seed(seed)
        
        data = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        
        np.random.shuffle(data)
        
        total = len(data)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)
        
        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]
        
        base_path = Path(data_file).parent
        
        for split_name, split_data in [
            ('train', train_data),
            ('val', val_data),
            ('test', test_data)
        ]:
            output_file = base_path / f'{split_name}.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in split_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            logger.info(f"Saved {len(split_data)} samples to {output_file}")
        
        return train_data, val_data, test_data