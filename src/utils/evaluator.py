# -*- coding: utf-8 -*-
"""
模型评估器
包含BLEU、ROUGE、困惑度等指标计算
"""

import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import logging
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
import jieba
from rouge import Rouge
from sklearn.metrics import accuracy_score, f1_score
import math

logger = logging.getLogger(__name__)


class Evaluator:
    """综合评估器"""
    
    def __init__(self, language: str = 'chinese'):
        self.language = language
        self.rouge = Rouge()
        self.smoothing = SmoothingFunction()
        
        # 初始化jieba分词（用于中文）
        if language == 'chinese':
            jieba.initialize()
    
    def tokenize(self, text: str) -> List[str]:
        """分词"""
        if self.language == 'chinese':
            return list(jieba.cut(text))
        else:
            return text.split()
    
    def calculate_bleu(
        self,
        predictions: List[str],
        references: List[str],
        n_gram: int = 4,
        smooth: bool = True
    ) -> Dict[str, float]:
        """计算BLEU分数"""
        
        # 分词
        pred_tokens = [self.tokenize(pred) for pred in predictions]
        ref_tokens = [[self.tokenize(ref)] for ref in references]
        
        # 计算不同n-gram的BLEU
        bleu_scores = {}
        for n in range(1, min(n_gram + 1, 5)):
            weights = [1.0 / n] * n + [0.0] * (4 - n)
            
            if smooth:
                score = corpus_bleu(
                    ref_tokens,
                    pred_tokens,
                    weights=weights,
                    smoothing_function=self.smoothing.method1
                )
            else:
                score = corpus_bleu(ref_tokens, pred_tokens, weights=weights)
            
            bleu_scores[f'bleu_{n}'] = score
        
        # 计算句子级BLEU
        sentence_bleus = []
        for pred, refs in zip(pred_tokens, ref_tokens):
            score = sentence_bleu(
                refs,
                pred,
                smoothing_function=self.smoothing.method1 if smooth else None
            )
            sentence_bleus.append(score)
        
        bleu_scores['bleu_avg'] = np.mean(sentence_bleus)
        
        return bleu_scores
    
    def calculate_rouge(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """计算ROUGE分数"""
        
        # 对于中文，需要在字符之间添加空格
        if self.language == 'chinese':
            predictions = [' '.join(self.tokenize(pred)) for pred in predictions]
            references = [' '.join(self.tokenize(ref)) for ref in references]
        
        try:
            scores = self.rouge.get_scores(predictions, references, avg=True)
            
            # 提取关键指标
            rouge_scores = {
                'rouge_1_f': scores['rouge-1']['f'],
                'rouge_2_f': scores['rouge-2']['f'],
                'rouge_l_f': scores['rouge-l']['f'],
                'rouge_1_p': scores['rouge-1']['p'],
                'rouge_1_r': scores['rouge-1']['r'],
            }
            
        except Exception as e:
            logger.warning(f"ROUGE calculation failed: {e}")
            rouge_scores = {
                'rouge_1_f': 0.0,
                'rouge_2_f': 0.0,
                'rouge_l_f': 0.0,
            }
        
        return rouge_scores
    
    def calculate_perplexity(
        self,
        model,
        dataloader,
        device: torch.device
    ) -> float:
        """计算困惑度"""
        model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # 将批次移到设备
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # 前向传播
                outputs = model(**batch)
                
                if hasattr(outputs, 'loss'):
                    loss = outputs.loss
                elif isinstance(outputs, dict) and 'loss' in outputs:
                    loss = outputs['loss']
                else:
                    continue
                
                # 累计损失
                total_loss += loss.item() * batch['labels'].numel()
                total_tokens += batch['labels'].numel()
        
        # 计算平均损失和困惑度
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')
        
        return perplexity
    
    def calculate_diversity(
        self,
        texts: List[str],
        n_gram: int = 2
    ) -> Dict[str, float]:
        """计算多样性指标（distinct-n）"""
        
        all_tokens = []
        all_ngrams = {i: [] for i in range(1, n_gram + 1)}
        
        for text in texts:
            tokens = self.tokenize(text)
            all_tokens.extend(tokens)
            
            # 收集n-grams
            for n in range(1, n_gram + 1):
                for i in range(len(tokens) - n + 1):
                    ngram = tuple(tokens[i:i+n])
                    all_ngrams[n].append(ngram)
        
        # 计算distinct-n
        diversity_scores = {}
        for n in range(1, n_gram + 1):
            if all_ngrams[n]:
                distinct = len(set(all_ngrams[n])) / len(all_ngrams[n])
                diversity_scores[f'distinct_{n}'] = distinct
            else:
                diversity_scores[f'distinct_{n}'] = 0.0
        
        # 计算熵
        token_counts = Counter(all_tokens)
        total = sum(token_counts.values())
        entropy = -sum(
            (count / total) * math.log(count / total)
            for count in token_counts.values()
        )
        diversity_scores['entropy'] = entropy
        
        return diversity_scores
    
    def calculate_semantic_similarity(
        self,
        predictions: List[str],
        references: List[str],
        model=None,
        tokenizer=None
    ) -> float:
        """计算语义相似度（需要预训练模型）"""
        
        if model is None or tokenizer is None:
            logger.warning("Semantic similarity requires model and tokenizer")
            return 0.0
        
        similarities = []
        model.eval()
        
        with torch.no_grad():
            for pred, ref in zip(predictions, references):
                # 编码文本
                pred_inputs = tokenizer(pred, return_tensors='pt', truncation=True)
                ref_inputs = tokenizer(ref, return_tensors='pt', truncation=True)
                
                # 获取嵌入
                pred_outputs = model(**pred_inputs)
                ref_outputs = model(**ref_inputs)
                
                # 使用[CLS] token的嵌入或平均池化
                if hasattr(pred_outputs, 'pooler_output'):
                    pred_emb = pred_outputs.pooler_output
                    ref_emb = ref_outputs.pooler_output
                else:
                    pred_emb = pred_outputs.last_hidden_state.mean(dim=1)
                    ref_emb = ref_outputs.last_hidden_state.mean(dim=1)
                
                # 计算余弦相似度
                similarity = torch.cosine_similarity(pred_emb, ref_emb)
                similarities.append(similarity.item())
        
        return np.mean(similarities)
    
    def evaluate_generation(
        self,
        predictions: List[str],
        references: List[str],
        calculate_all: bool = True
    ) -> Dict[str, Any]:
        """综合评估生成结果"""
        
        results = {}
        
        # BLEU分数
        if calculate_all or 'bleu' in calculate_all:
            results.update(self.calculate_bleu(predictions, references))
        
        # ROUGE分数
        if calculate_all or 'rouge' in calculate_all:
            results.update(self.calculate_rouge(predictions, references))
        
        # 多样性
        if calculate_all or 'diversity' in calculate_all:
            results.update(self.calculate_diversity(predictions))
        
        # 基本统计
        pred_lengths = [len(self.tokenize(pred)) for pred in predictions]
        ref_lengths = [len(self.tokenize(ref)) for ref in references]
        
        results.update({
            'avg_pred_length': np.mean(pred_lengths),
            'avg_ref_length': np.mean(ref_lengths),
            'length_ratio': np.mean(pred_lengths) / np.mean(ref_lengths) if np.mean(ref_lengths) > 0 else 0
        })
        
        return results
    
    def print_evaluation_report(self, results: Dict[str, Any]):
        """打印评估报告"""
        
        print("\n" + "="*50)
        print("EVALUATION REPORT")
        print("="*50)
        
        # BLEU分数
        print("\nBLEU Scores:")
        for key, value in results.items():
            if key.startswith('bleu'):
                print(f"  {key}: {value:.4f}")
        
        # ROUGE分数
        print("\nROUGE Scores:")
        for key, value in results.items():
            if key.startswith('rouge'):
                print(f"  {key}: {value:.4f}")
        
        # 多样性分数
        print("\nDiversity Scores:")
        for key, value in results.items():
            if key.startswith('distinct') or key == 'entropy':
                print(f"  {key}: {value:.4f}")
        
        # 其他指标
        print("\nOther Metrics:")
        other_keys = ['avg_pred_length', 'avg_ref_length', 'length_ratio', 'perplexity']
        for key in other_keys:
            if key in results:
                print(f"  {key}: {results[key]:.4f}")
        
        print("="*50)


class HumanEvaluator:
    """人工评估辅助工具"""
    
    def __init__(self):
        self.criteria = {
            'fluency': '流畅度（1-5）：语言是否通顺自然',
            'relevance': '相关性（1-5）：回复是否与输入相关',
            'informativeness': '信息量（1-5）：回复是否提供有用信息',
            'diversity': '多样性（1-5）：回复是否有创意，避免重复',
            'overall': '总体评分（1-5）：整体质量'
        }
    
    def create_evaluation_template(
        self,
        dialogues: List[Tuple[str, str, str]],
        output_file: str
    ):
        """创建人工评估模板"""
        
        import pandas as pd
        
        data = []
        for i, (input_text, prediction, reference) in enumerate(dialogues):
            for criterion in self.criteria.keys():
                data.append({
                    'id': i,
                    'input': input_text,
                    'prediction': prediction,
                    'reference': reference,
                    'criterion': criterion,
                    'description': self.criteria[criterion],
                    'score': None,
                    'comment': None
                })
        
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        logger.info(f"Created evaluation template: {output_file}")
    
    def analyze_human_evaluation(self, evaluation_file: str) -> Dict[str, float]:
        """分析人工评估结果"""
        
        import pandas as pd
        
        df = pd.read_csv(evaluation_file, encoding='utf-8-sig')
        
        # 计算各项平均分
        results = {}
        for criterion in self.criteria.keys():
            scores = df[df['criterion'] == criterion]['score'].dropna()
            if len(scores) > 0:
                results[f'{criterion}_mean'] = scores.mean()
                results[f'{criterion}_std'] = scores.std()
        
        # 计算一致性（如果有多个评估者）
        if 'evaluator' in df.columns:
            from sklearn.metrics import cohen_kappa_score
            evaluators = df['evaluator'].unique()
            if len(evaluators) >= 2:
                eval1 = df[df['evaluator'] == evaluators[0]]['score'].values
                eval2 = df[df['evaluator'] == evaluators[1]]['score'].values
                kappa = cohen_kappa_score(eval1, eval2)
                results['inter_rater_agreement'] = kappa
        
        return results