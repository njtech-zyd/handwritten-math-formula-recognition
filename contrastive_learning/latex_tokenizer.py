"""
LaTeX Tokenizer — 将 LaTeX 字符串转成 token 序列。

词汇表包含：
- 单个字符（字母、数字、符号）
- LaTeX 命令 (\\frac, \\sqrt, \\phi 等，含前导反斜杠)
- 特殊标记：<pad>, <sos>, <eos>, <unk>
"""

import re
from pathlib import Path
from typing import Optional

SPECIAL = {
    "<pad>": 0,
    "<sos>": 1,
    "<eos>": 2,
    "<unk>": 3,
}

# 正则：匹配 LaTeX 命令（\单词）或单个字符
_TOKEN_RE = re.compile(r"\\([a-zA-Z]+)|(.)")


def build_vocab(latex_list: list[str], min_freq: int = 2) -> dict[str, int]:
    """从 LaTeX 列表构建词汇表。

    Args:
        latex_list: LaTeX 字符串列表。
        min_freq: 最低出现频率（过滤罕见 token）。

    Returns:
        {token: id} 字典。
    """
    counts: dict[str, int] = {}
    for latex in latex_list:
        text = latex.strip().strip("$").strip()
        for cmd, char in _TOKEN_RE.findall(text):
            token = f"\\{cmd}" if cmd else char
            counts[token] = counts.get(token, 0) + 1

    # 按频率排序，构建词汇表
    vocab = dict(SPECIAL)
    for token, count in sorted(counts.items(), key=lambda x: -x[1]):
        if count >= min_freq:
            vocab[token] = len(vocab)
    return vocab


def tokenize(latex: str, vocab: dict[str, int]) -> list[int]:
    """将 LaTeX 字符串转为 token ID 列表（含 <sos> 和 <eos>）。"""
    text = latex.strip().strip("$").strip()
    ids = [SPECIAL["<sos>"]]
    unk_id = SPECIAL["<unk>"]
    for cmd, char in _TOKEN_RE.findall(text):
        token = f"\\{cmd}" if cmd else char
        ids.append(vocab.get(token, unk_id))
    ids.append(SPECIAL["<eos>"])
    return ids


def detokenize(ids: list[int], vocab: dict[str, int]) -> str:
    """将 token ID 列表恢复为 LaTeX 字符串。"""
    id_to_token = {v: k for k, v in vocab.items()}
    tokens = []
    for tid in ids:
        if tid in (SPECIAL["<pad>"], SPECIAL["<sos>"]):
            continue
        if tid == SPECIAL["<eos>"]:
            break
        token = id_to_token.get(tid, "")
        if token == "<unk>":
            continue
        tokens.append(token)

    # 组装回字符串：普通字符直接拼接，LaTeX 命令保留 \
    result = ""
    for t in tokens:
        if t.startswith("\\"):
            result += t
        else:
            result += t
    return result


def latex_collate(
    batch: list[dict],
    vocab: dict[str, int],
    max_len: int = 128,
) -> tuple:
    """DataLoader collate 函数：将 batch 中的 LaTeX 转成 padded tensor。

    Args:
        batch: [{"features": [49,256], "latex": str}, ...]
        vocab: 词汇表。
        max_len: 最大序列长度。

    Returns:
        (features, tokens) — features: [B, 49, 256], tokens: [B, L]
    """
    import torch
    import torch.nn.utils.rnn as rnn_utils

    features = []
    token_ids = []
    pad_id = SPECIAL["<pad>"]

    for item in batch:
        features.append(torch.from_numpy(item["features"]))
        ids = tokenize(item["latex"], vocab)
        if len(ids) > max_len:
            ids = ids[:max_len]
        token_ids.append(torch.tensor(ids, dtype=torch.long))

    # Pad token sequences
    padded = rnn_utils.pad_sequence(token_ids, batch_first=True, padding_value=pad_id)
    # Truncate to max_len
    if padded.shape[1] > max_len:
        padded = padded[:, :max_len]

    return torch.stack(features), padded
