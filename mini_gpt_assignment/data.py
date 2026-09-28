"""
data.py
-------
Handles turning tokenized text into (x, y) training batches for
next-token prediction.

This assignment says to "use preprocessed dataset from Assignment 1
(tokenized and cleaned)". This module supports FOUR input shapes so you
can point it at whatever you already have:

  0. Nidhi's actual Assignment 1 output: a flat, memory-mapped token-id
     binary + sidecar meta.json, as written by
     data_collection_preprocessing.py's tokenize_corpus() step
     -> data_path=".../pipeline_output/tokenized/tokens.bin"
     (meta.json in the same folder must have "dtype" and "vocab_size")
     This is loaded via np.memmap, so even a full ~1GB-of-text run (many
     hundreds of millions of tokens) is NOT pulled into RAM all at once --
     only the small windows get_batch() actually samples are touched.
  1. A .pt file containing a 1D torch.LongTensor of token ids
     -> data_path="data/assignment1_tokens.pt"
  2. A .npy file containing a 1D numpy array of token ids
     -> data_path="data/assignment1_tokens.npy"
  3. A plain .txt file of raw (already-cleaned) text
     -> data_path="data/assignment1_clean.txt"
     (a simple built-in character-level tokenizer is fit on it)

If none of the above exist, it falls back to a small bundled sample
corpus (data/sample_corpus.txt) purely so the pipeline is runnable
end-to-end out of the box. Replace this with your real Assignment 1
output before you write up results for the report.
"""

import os
import json
import numpy as np
import torch


class CharTokenizer:
    """Minimal character-level tokenizer, used only as a fallback.

    Swap this out if Assignment 1 already produced a BPE/word-piece
    vocabulary and integer ids -- in that case just load the ids
    directly (see load_token_ids below) and skip this class entirely.
    """

    def __init__(self, text: str):
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, s: str):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)


def _ensure_sample_corpus(path: str):
    """Writes a tiny original placeholder corpus if nothing else is found,
    so the code always runs. This is NOT meant to stand in for your real
    Assignment 1 dataset -- it exists only so train.py never crashes on
    a fresh checkout."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        return
    placeholder = (
        "this is a small placeholder corpus used only to verify that the "
        "training pipeline runs end to end. replace this file, or better, "
        "point config.data_path at your real assignment one output, "
        "before you record any results for your report. a transformer "
        "learns to predict the next token in a sequence by attending to "
        "the tokens that came before it. training on this placeholder "
        "text will not produce meaningful language modeling results, it "
        "only exercises the code path. " * 40
    )
    with open(path, "w") as f:
        f.write(placeholder)


def load_packed_bin(path: str):
    """Loads the tokens.bin + meta.json pair written by Nidhi's Assignment 1
    tokenize_corpus() step. Returns (memmap array, vocab_size). Kept as a
    numpy memmap (NOT converted to a torch.Tensor here) so a large corpus is
    never fully read into RAM -- get_batch() below reads only the small
    windows it actually samples."""
    meta_path = os.path.join(os.path.dirname(path), "meta.json")
    with open(meta_path) as f:
        meta = json.load(f)

    dtype_map = {"uint16": np.uint16, "uint32": np.uint32, "int32": np.int32, "int64": np.int64}
    dtype = dtype_map.get(meta.get("dtype", "uint32"), np.uint32)

    arr = np.memmap(path, dtype=dtype, mode="r")
    vocab_size = max(int(meta.get("vocab_size", 0)), int(arr.max()) + 1)
    print(
        f"[data.py] loaded packed token stream: {arr.shape[0]:,} tokens, "
        f"tokenizer={meta.get('tokenizer')}, vocab_size={vocab_size}"
    )
    return arr, vocab_size


def load_token_ids(cfg):
    """Returns (all_token_ids, vocab_size). all_token_ids is either a 1D
    torch.LongTensor (small in-memory datasets) or a numpy memmap (the
    packed .bin case) -- get_batch() and make_splits() below handle both."""
    path = cfg.data_path

    if path.endswith(".bin") and os.path.exists(path):
        return load_packed_bin(path)

    if path.endswith(".pt") and os.path.exists(path):
        ids = torch.load(path)
        if not torch.is_tensor(ids):
            ids = torch.tensor(ids, dtype=torch.long)
        vocab_size = int(ids.max().item()) + 1
        return ids.long(), vocab_size

    if path.endswith(".npy") and os.path.exists(path):
        arr = np.load(path)
        ids = torch.tensor(arr, dtype=torch.long)
        vocab_size = int(ids.max().item()) + 1
        return ids, vocab_size

    if path.endswith(".txt") and os.path.exists(path):
        with open(path, "r") as f:
            text = f.read()
        tok = CharTokenizer(text)
        ids = torch.tensor(tok.encode(text), dtype=torch.long)
        return ids, tok.vocab_size

    # Nothing found -> fall back to bundled sample corpus.
    _ensure_sample_corpus(cfg.fallback_raw_text_path)
    with open(cfg.fallback_raw_text_path, "r") as f:
        text = f.read()
    tok = CharTokenizer(text)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    print(
        f"[data.py] WARNING: '{cfg.data_path}' not found. Using bundled "
        f"placeholder corpus at '{cfg.fallback_raw_text_path}' instead. "
        f"Point cfg.data_path at your real Assignment 1 output before "
        f"you record results for the report."
    )
    return ids, tok.vocab_size


def make_splits(ids: torch.Tensor, val_fraction: float = 0.1):
    n = len(ids)
    n_val = int(n * val_fraction)
    train_ids = ids[: n - n_val]
    val_ids = ids[n - n_val:]
    return train_ids, val_ids


def get_batch(data, batch_size: int, block_size: int, device: str):
    """Randomly samples `batch_size` chunks of length `block_size` and
    builds (x, y) pairs for next-token prediction: y is x shifted by one.

    `data` may be a torch.LongTensor (already fully in memory) or a numpy
    memmap (the packed .bin case) -- either way, only the sampled windows
    (batch_size * block_size elements) ever get materialized, never the
    whole dataset."""
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))

    if isinstance(data, torch.Tensor):
        x = torch.stack([data[i: i + block_size] for i in ix])
        y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])
    else:
        x_np = np.stack([np.asarray(data[i: i + block_size], dtype=np.int64) for i in ix])
        y_np = np.stack([np.asarray(data[i + 1: i + block_size + 1], dtype=np.int64) for i in ix])
        x, y = torch.from_numpy(x_np), torch.from_numpy(y_np)

    return x.to(device), y.to(device)
