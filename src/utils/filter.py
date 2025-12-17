# -*- coding: utf-8 -*-
"""
改进的敏感词过滤系统
包含Trie树、同音字检测和变体识别
"""

import re
from typing import Set, List, Dict, Optional, Tuple, Any
from pathlib import Path
import logging
import json
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrieNode:
    """Trie树节点"""
    
    def __init__(self):
        self.children: Dict[str, TrieNode] = {}
        self.is_end: bool = False
        self.word: Optional[str] = None


class Trie:
    """Trie树实现"""
    
    def __init__(self):
        self.root = TrieNode()
        self.word_count = 0
        self._unique_words = set()  # 🔧 修复：添加唯一词集合
    
    def insert(self, word: str):
        """插入单词"""
        if not word:
            return
        
        word_lower = word.lower()
        
        # 🔧 修复：避免重复计数
        if word_lower in self._unique_words:
            return
        
        node = self.root
        for char in word_lower:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        if not node.is_end:  # 只在第一次插入时计数
            node.is_end = True
            node.word = word
            self.word_count += 1
            self._unique_words.add(word_lower)
    
    def search(self, word: str) -> bool:
        """搜索单词"""
        node = self.root
        for char in word.lower():
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end
    
    def search_prefix(self, prefix: str) -> List[str]:
        """搜索前缀匹配的所有单词"""
        node = self.root
        for char in prefix.lower():
            if char not in node.children:
                return []
            node = node.children[char]
        
        # DFS收集所有单词
        words = []
        self._dfs_collect(node, words)
        return words
    
    def _dfs_collect(self, node: TrieNode, words: List[str]):
        """深度优先搜索收集单词"""
        if node.is_end and node.word:
            words.append(node.word)
        for child in node.children.values():
            self._dfs_collect(child, words)
    
    def find_all_matches(self, text: str) -> List[Tuple[int, int, str]]:
        """找出文本中所有匹配的敏感词（最长匹配优先）"""
        matches = []
        text_lower = text.lower()
        n = len(text)
        i = 0
        
        while i < n:
            node = self.root
            longest_match = None
            
            # 从位置i开始尝试匹配
            for j in range(i, n):
                char = text_lower[j]
                if char not in node.children:
                    break
                node = node.children[char]
                
                # 🔧 优化：记录最长匹配
                if node.is_end and node.word:
                    longest_match = (i, j + 1, node.word)
            
            if longest_match:
                matches.append(longest_match)
                i = longest_match[1]  # 跳过已匹配部分
            else:
                i += 1
        
        return matches


class DirtyFilter:
    """敏感词过滤器"""
    
    # 预编译正则表达式
    _ZERO_WIDTH_PATTERN = re.compile(r'[\u200b\u200c\u200d\ufeff]')
    _REPEAT_PATTERN = re.compile(r'(.)\1{2,}')
    
    def __init__(
        self,
        dirty_words_path: Optional[str] = None,
        enable_variants: bool = True,
        enable_pinyin: bool = False,
        max_variant_depth: int = 1
    ):
        self.trie = Trie()
        self.enable_variants = enable_variants
        self.enable_pinyin = enable_pinyin
        self.max_variant_depth = max_variant_depth
        
        # 🔧 修复：添加缺失的属性
        self.dirty_words = set()
        
        # 变体映射
        self.variants_map = self._build_variants_map()
        
        # 同音字映射（简化版）
        self.homophones = self._build_homophones_map()
        
        if dirty_words_path:
            self.load_dirty_words(dirty_words_path)
    
    def _build_variants_map(self) -> Dict[str, List[str]]:
        """构建变体映射（形近字、符号替换等）"""
        return {
            '0': ['o', 'O', '零', '〇'],
            '1': ['i', 'I', 'l', '一', '壹'],
            '2': ['二', '贰'],
            '3': ['三', '叁'],
            '4': ['四', '肆', 'for'],
            '5': ['五', '伍'],
            '6': ['六', '陆'],
            '7': ['七', '柒'],
            '8': ['八', '捌'],
            '9': ['九', '玖'],
            'a': ['@', 'а', 'ɑ'],
            'e': ['3', 'є'],
            'i': ['!', '1', 'і'],
            'o': ['0', 'о'],
            's': ['$', '5'],
            'g': ['9'],
            'b': ['6'],
        }
    
    def _build_homophones_map(self) -> Dict[str, List[str]]:
        """构建同音字映射（简化版）"""
        return {
            '操': ['草', '曹', '槽'],
            '妈': ['马', '吗', '玛', '骂'],
            '逼': ['比', '笔', '毙', '币'],
            '死': ['四', '私', '思', '斯'],
        }
    
    def load_dirty_words(self, path: str):
        """加载敏感词库"""
        try:
            file_path = Path(path)
            if not file_path.exists():
                logger.warning(f"Dirty words file not found: {path}")
                return
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip()
                    if word and not word.startswith('#'):
                        self.add_word(word)
            
            logger.info(f"Loaded {self.trie.word_count} dirty words")
            
        except Exception as e:
            logger.error(f"Failed to load dirty words: {e}")
    
    def add_word(self, word: str):
        """添加敏感词"""
        word = word.lower().strip()
        if not word:
            return
        
        self.dirty_words.add(word)  # 🔧 修复：更新dirty_words集合
        self.trie.insert(word)
        
        # 🔧 优化：限制变体生成，避免组合爆炸
        if self.enable_variants:
            variants = self._generate_variants(word, depth=self.max_variant_depth)
            for variant in variants:
                self.trie.insert(variant)
    
    def _generate_variants(self, word: str, depth: int = 1) -> Set[str]:
        """生成单词的变体（限制递归深度）"""
        if depth <= 0:
            return set()
        
        variants = set()
        
        # 生成单字符替换变体
        for i, char in enumerate(word):
            if char in self.variants_map:
                for variant_char in self.variants_map[char]:
                    variant = word[:i] + variant_char + word[i+1:]
                    if variant != word:
                        variants.add(variant)
        
        # 生成同音字变体
        if self.enable_pinyin:
            for i, char in enumerate(word):
                if char in self.homophones:
                    for homophone in self.homophones[char]:
                        variant = word[:i] + homophone + word[i+1:]
                        if variant != word:
                            variants.add(variant)
        
        return variants
    
    def normalize_text(self, text: str) -> str:
        """最小化标准化 - 只处理明确的绕过手段"""
        # 1. 移除零宽字符
        text = self._ZERO_WIDTH_PATTERN.sub('', text)
        
        # 2. 全角转半角
        text = self._full_to_half(text)
        
        # 3. 压缩重复字符（保留2个）
        text = self._REPEAT_PATTERN.sub(r'\1\1', text)
        
        return text.lower()
    
    def _full_to_half(self, text: str) -> str:
        """全角转半角"""
        result = []
        for char in text:
            code = ord(char)
            if code == 0x3000:
                code = 0x20
            elif 0xFF01 <= code <= 0xFF5E:
                code -= 0xFEE0
            result.append(chr(code))
        return ''.join(result)
    
    def check(self, text: str) -> bool:
        """检查文本是否包含敏感词"""
        if not text:
            return False
        normalized = self.normalize_text(text)
        return bool(self.trie.find_all_matches(normalized))
    
    def find_dirty_words(self, text: str) -> List[str]:
        """找出文本中的所有敏感词"""
        if not text:
            return []
        
        found_words = set()
        
        # 标准化文本匹配
        normalized = self.normalize_text(text)
        matches = self.trie.find_all_matches(normalized)
        for _, _, word in matches:
            found_words.add(word)
        
        return list(found_words)
    
    def replace_dirty_words(self, text: str, replacement: str = "***") -> str:
        """替换文本中的敏感词"""
        if not text:
            return text
        
        # 找出所有匹配
        normalized = self.normalize_text(text)
        matches = self.trie.find_all_matches(normalized)
        
        # 按位置从后往前替换（避免位置偏移）
        matches.sort(key=lambda x: x[0], reverse=True)
        
        result = text
        for start, end, _ in matches:
            result = result[:start] + replacement + result[end:]
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'total_words': self.trie.word_count,
            'unique_words': len(self.dirty_words), 
            'variants_enabled': self.enable_variants,
            'pinyin_enabled': self.enable_pinyin
        }


class ContentModerator:
    """内容审核器（更高级的内容审核）"""
    
    def __init__(
        self,
        dirty_filter: DirtyFilter,
        toxicity_threshold: float = 0.7
    ):
        self.dirty_filter = dirty_filter
        self.toxicity_threshold = toxicity_threshold
        
        # 内容类别
        self.categories = {
            'profanity': ['脏话', '侮辱'],
            'violence': ['暴力', '威胁'],
            'sexual': ['色情', '性'],
            'hate': ['歧视', '仇恨'],
            'spam': ['广告', '垃圾']
        }
    
    def moderate(self, text: str) -> Dict[str, Any]:
        """审核内容"""
        result = {
            'safe': True,
            'categories': {},
            'dirty_words': [],
            'confidence': 0.0,
            'action': 'pass'
        }
        
        # 敏感词检查
        if self.dirty_filter.check(text):
            result['safe'] = False
            result['dirty_words'] = self.dirty_filter.find_dirty_words(text)
            result['action'] = 'block'
            result['confidence'] = 0.9
        
        # 基于规则的分类
        for category, keywords in self.categories.items():
            for keyword in keywords:
                if keyword in text.lower():
                    result['categories'][category] = True
                    result['safe'] = False
                    if result['action'] == 'pass':
                        result['action'] = 'warning'
        
        return result
    
    def suggest_alternative(self, text: str) -> str:
        """建议替代文本"""
        # 替换敏感词
        cleaned = self.dirty_filter.replace_dirty_words(text)
        
        # 进一步的文本清理
        cleaned = re.sub(r'[!！]{3,}', '！', cleaned)
        cleaned = re.sub(r'[?？]{3,}', '？', cleaned)
        cleaned = re.sub(r'[.。]{3,}', '...', cleaned)
        
        return cleaned