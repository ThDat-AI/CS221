from __future__ import annotations

import random
import re
from collections import Counter
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset


PAD = "<pad>"
UNK = "<unk>"
_URL_TOK = "URLPLACEHOLDER"
_USER_TOK = "USERPLACEHOLDER"
_NAME_TOK = "NAMEPLACEHOLDER"
_NUMBER_TOK = "NUMBERPLACEHOLDER"

_DEMOJIZE_COLON_BLOCK = re.compile(r":([-+a-zA-Z0-9][-a-zA-Z0-9_]*):")


def expand_demojized_tokens(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text if isinstance(text, str) else ""

    def repl(m: re.Match[str]) -> str:
        inner = m.group(1)
        if inner.isdigit():
            return m.group(0)
        phrase = inner.replace("_", " ").strip()
        if not phrase:
            return m.group(0)
        return f" {phrase} "

    t = _DEMOJIZE_COLON_BLOCK.sub(repl, text)
    return re.sub(r"\s+", " ", t).strip()


def normalize_placeholder_tokens(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = re.sub(r"<username>", f" {_USER_TOK} ", t, flags=re.I)
    t = re.sub(r"<url>", f" {_URL_TOK} ", t, flags=re.I)
    t = re.sub(r"<name>", f" {_NAME_TOK} ", t, flags=re.I)
    t = re.sub(r"<number>", f" {_NUMBER_TOK} ", t, flags=re.I)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tokenize(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    t = normalize_placeholder_tokens(text.strip())
    t = expand_demojized_tokens(t)
    return word_tokenize(t)


def build_vocab(
    texts: list[str],
    min_freq: int = 2,
) -> tuple[dict[str, int], dict[int, str]]:
    cnt = Counter()
    for t in texts:
        cnt.update(tokenize(t))
    word2idx: dict[str, int] = {PAD: 0, UNK: 1}
    for w, c in cnt.items():
        if c >= min_freq:
            word2idx[w] = len(word2idx)
    idx2word = {i: w for w, i in word2idx.items()}
    return word2idx, idx2word


def encode_texts(
    texts: list[str],
    word2idx: dict[str, int],
    max_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    unk = word2idx["<unk>"]
    pad = word2idx["<pad>"]
    batch_ids: list[list[int]] = []
    batch_lens: list[int] = []
    for t in texts:
        ids = [word2idx.get(w, unk) for w in tokenize(t)]
        if len(ids) > max_len:
            ids = ids[:max_len]
        batch_lens.append(len(ids) if ids else 1)
        while len(ids) < max_len:
            ids.append(pad)
        batch_ids.append(ids[:max_len])
    x = torch.tensor(batch_ids, dtype=torch.long)
    lens = torch.tensor(batch_lens, dtype=torch.long)
    return x, lens


class TextClassificationDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        word2idx: dict[str, int],
        max_len: int,
    ):
        self.texts = texts
        self.labels = labels
        self.word2idx = word2idx
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        t = self.texts[idx]
        y = self.labels[idx]
        x, lens = encode_texts([t], self.word2idx, self.max_len)
        return x.squeeze(0), lens.squeeze(0), torch.tensor(y, dtype=torch.long)


def collate_batch(batch):
    xs = torch.stack([b[0] for b in batch], dim=0)
    lens = torch.stack([b[1] for b in batch], dim=0)
    ys = torch.stack([b[2] for b in batch], dim=0)
    return xs, lens, ys


def load_glove_embeddings(
    glove_path: str | Path,
    word2idx: dict[str, int],
    embed_dim: int,
) -> tuple[torch.Tensor, dict[str, float | int | str]]:
    path = Path(glove_path)
    n = len(word2idx)
    mat = np.random.uniform(-0.25, 0.25, (n, embed_dim)).astype(np.float32)
    mat[0] = 0.0
    hit_indices: set[int] = set()
    applied = 0
    with path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip().split()
            if len(parts) != embed_dim + 1:
                continue
            w = parts[0]
            if w not in word2idx:
                continue
            idx = word2idx[w]
            mat[idx] = np.asarray(parts[1:], dtype=np.float32)
            hit_indices.add(idx)
            applied += 1
    matched = len(hit_indices)
    stats: dict[str, float | int | str] = {
        "path": str(path.resolve()),
        "format": "glove",
        "embed_dim": embed_dim,
        "vocab_size": n,
        "matched_vocab_rows": matched,
        "lines_used": applied,
        "coverage": float(matched / max(n, 1)),
    }
    return torch.from_numpy(mat), stats


def load_fasttext_embeddings(
    ft_path: str | Path,
    word2idx: dict[str, int],
    embed_dim: int,
) -> tuple[torch.Tensor, dict[str, float | int | str]]:
    path = Path(ft_path)
    n = len(word2idx)
    mat = np.random.uniform(-0.25, 0.25, (n, embed_dim)).astype(np.float32)
    mat[0] = 0.0
    hit_indices: set[int] = set()
    applied = 0

    if path.suffix == ".bin":
        import fasttext
        fasttext.FastText.eprint = lambda x: None
        model = fasttext.load_model(str(path))
        if model.get_dimension() != embed_dim:
            raise ValueError(f"FastText file declares dim={model.get_dimension()} but embed_dim={embed_dim}")
        
        for w, idx in word2idx.items():
            if idx == 0:
                continue
            mat[idx] = model.get_word_vector(w)
            hit_indices.add(idx)
            applied += 1
            
        stats = {
            "path": str(path.resolve()),
            "format": "fasttext-bin",
            "embed_dim": embed_dim,
            "vocab_size": n,
            "matched_vocab_rows": applied,
            "lines_used": applied,
            "coverage": 1.0,
        }
        return torch.from_numpy(mat), stats

    with path.open(encoding="utf-8", errors="ignore") as f:
        first = f.readline()
        if not first:
            stats: dict[str, float | int | str] = {
                "path": str(path.resolve()),
                "format": "fasttext",
                "embed_dim": embed_dim,
                "vocab_size": n,
                "matched_vocab_rows": 0,
                "lines_used": 0,
                "coverage": 0.0,
            }
            return torch.from_numpy(mat), stats
        parts0 = first.rstrip().split()
        skip_header = (
            len(parts0) == 2
            and parts0[0].isdigit()
            and parts0[1].isdigit()
        )
        if skip_header:
            header_dim = int(parts0[1])
            if header_dim != embed_dim:
                raise ValueError(
                    f"FastText file declares dim={header_dim} but embed_dim={embed_dim}"
                )
            line_iter = f
        else:
            line_iter = chain([first], f)
        for line in line_iter:
            parts = line.rstrip().split()
            if len(parts) != embed_dim + 1:
                continue
            w = parts[0]
            if w not in word2idx:
                continue
            idx = word2idx[w]
            mat[idx] = np.asarray(parts[1:], dtype=np.float32)
            hit_indices.add(idx)
            applied += 1
    matched = len(hit_indices)
    stats = {
        "path": str(path.resolve()),
        "format": "fasttext",
        "embed_dim": embed_dim,
        "vocab_size": n,
        "matched_vocab_rows": matched,
        "lines_used": applied,
        "coverage": float(matched / max(n, 1)),
    }
    return torch.from_numpy(mat), stats


def load_csv_split(path: str | Path) -> tuple[list[str], list[str]]:
    df = pd.read_csv(path, encoding="utf-8")
    text_col = "text" if "text" in df.columns else df.columns[0]
    label_col = "label" if "label" in df.columns else df.columns[1]
    texts = df[text_col].fillna("").astype(str).tolist()
    labels = df[label_col].astype(str).str.strip().tolist()
    return texts, labels


def build_label_maps(label_order: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    str2id = {s: i for i, s in enumerate(label_order)}
    id2str = {i: s for s, i in str2id.items()}
    return str2id, id2str


def labels_to_ids(raw: list[str], str2id: dict[str, int]) -> list[int]:
    return [str2id[x] for x in raw]
