# data/

数据目录。大型语料（`*.json`、`douban_kuakua_qa.txt`）与 `cache/` 已被 `.gitignore` 忽略，仅保留小型参考词库。

## 已跟踪文件

- `dirty_words.txt` — 主敏感词词库
- `暴恐词库.txt` / `网易前端过滤敏感词库.txt` / `零时-Tencent.txt` — 第三方敏感词词库

## 需自行准备

- `douban_kuakua_qa.txt` — 原始对话语料（豆瓣"夸夸群"数据集）
- `train.json` / `val.json` / `test.json` — 由 `python main.py preprocess --split` 生成
- `cache/` — 数据集 pickle 缓存，首次训练时自动创建
