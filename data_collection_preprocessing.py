"""
Assignment 1: Data Collection and Preprocessing for Foundation Model Pre-Training
==================================================================================

Pipeline stages (each is a clearly separated section below so you can run,
debug, or explain any single stage independently in your report):

    1. DATA COLLECTION   - stream >=1GB of raw text from 3 diverse public
                            sources (encyclopedic / news / general web) using
                            HuggingFace `datasets`, so we never have to
                            download a full multi-hundred-GB corpus to disk.
    2. CLEANING          - strip HTML/markdown/reference markers, normalize
                            whitespace, drop very short / boilerplate docs,
                            exact-duplicate removal via content hashing.
    3. TOKENIZATION      - GPT-2 BPE tokenizer (AutoTokenizer), documents are
                            concatenated with an EOS separator and "packed"
                            into fixed-length blocks (the standard, memory-
                            efficient scheme used by GPT-2/GPT-3/nanoGPT-style
                            pretraining -- avoids wasting compute on padding).
    4. DATA LOADER       - two custom PyTorch datasets are provided:
                              a) PackedBlockDataset  (IterableDataset) - the
                                 primary, efficient, streaming-friendly loader
                                 used for actual pretraining.
                              b) PaddedDocDataset     (Dataset) - a per-
                                 document loader with a padding/truncation
                                 collate_fn, included specifically to satisfy
                                 the "handle variable-length sequences with
                                 padding" requirement and to let you compare
                                 padding-waste vs packing efficiency in your
                                 report.

Run modes
---------
--dry_run    : uses a small built-in synthetic corpus, no internet needed.
               Use this FIRST to confirm the whole pipeline runs end to end
               on your machine before kicking off the full ~1GB collection.
--full       : streams real data from HF Hub until --target_gb is reached.

Example:
    python data_collection_preprocessing.py --dry_run
    python data_collection_preprocessing.py --full --target_gb 1.0 --block_size 1024

Requirements: see requirements.txt (pip install -r requirements.txt)
"""

import argparse
import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional

import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

@dataclass
class Config:
    out_dir: str = "pipeline_output"
    target_gb: float = 1.0                 # total raw text size to collect
    min_words: int = 50                    # drop documents shorter than this
    lowercase: bool = False                # see report: NOT recommended for
                                            # BPE pretraining, kept configurable
                                            # only because the assignment asks
                                            # for it as an optional cleaning step
    tokenizer_name: str = "gpt2"
    block_size: int = 1024                 # sequence length fed to the model
    stride: int = 0                        # 0 = non-overlapping packed blocks
    batch_size: int = 8
    num_workers: int = 2
    seed: int = 42
    sample_batches: int = 8                # how many batches to save as sample
    # HF streaming sources: (dataset_name, config_or_None, split, text_field, domain_tag)
    sources: List[tuple] = field(default_factory=lambda: [
        ("wikimedia/wikipedia", "20231101.en", "train", "text", "encyclopedic"),
        ("vblagoje/cc_news", None, "train", "text", "news"),
        ("Skylion007/openwebtext", None, "train", "text", "web"),
    ])


# --------------------------------------------------------------------------
# 1. DATA COLLECTION
# --------------------------------------------------------------------------

def collect_raw_text(cfg: Config) -> Path:
    """Stream text from each configured HF source until the combined size
    target is hit, writing one raw .jsonl shard per domain. Streaming mode
    (`streaming=True`) means HF never downloads/caches the whole underlying
    dataset -- it pulls examples lazily from the hub, which is essential
    here since e.g. OpenWebText alone is ~40GB and full Wikipedia dumps are
    several GB; we only need a representative ~300-400MB slice of each.
    """
    from datasets import load_dataset  # imported lazily so --dry_run works
                                        # without the `datasets` package too

    raw_dir = Path(cfg.out_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    target_bytes = int(cfg.target_gb * (1024 ** 3))
    per_source_target = target_bytes // len(cfg.sources)

    manifest = {}
    for name, subset, split, text_field, domain in cfg.sources:
        shard_path = raw_dir / f"{domain}.jsonl"
        written = 0
        t0 = time.time()
        print(f"[collect] streaming {name} ({domain}) -> target "
              f"{per_source_target / 1e6:.0f} MB")
        try:
            ds = load_dataset(name, subset, split=split, streaming=True)
        except Exception as e:
            print(f"[collect] WARNING: failed to open {name}: {e}")
            print(f"[collect] skipping this source -- see report 'Challenges'")
            continue

        with open(shard_path, "w", encoding="utf-8") as f:
            for ex in ds:
                text = ex.get(text_field, "")
                if not text:
                    continue
                f.write(json.dumps({"text": text, "domain": domain}) + "\n")
                written += len(text.encode("utf-8"))
                if written >= per_source_target:
                    break
        dt = time.time() - t0
        print(f"[collect] {domain}: wrote {written / 1e6:.1f} MB in {dt:.1f}s")
        manifest[domain] = {"path": str(shard_path), "bytes": written}

    with open(raw_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return raw_dir


def make_dry_run_corpus(cfg: Config) -> Path:
    """No-internet smoke test: writes a tiny synthetic multi-domain corpus so
    every downstream stage (cleaning, tokenization, data loader) can be
    exercised quickly on a laptop with no network access.
    """
    raw_dir = Path(cfg.out_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # NOTE: real corpora vary naturally, so exact-duplicate documents are
    # relatively rare relative to total volume. A synthetic smoke-test corpus
    # built by repeating a handful of *identical* sentences is unrealistic in
    # the opposite direction -- nearly everything gets correctly deduped away,
    # leaving too little text to fill even one training block. Each repeat
    # below gets a small unique suffix (e.g. "(variant 7)") so dedup behaves
    # the way it would on a real corpus (dropping literal repeats, not
    # collapsing everything down to a handful of docs), while still giving
    # the loader enough post-cleaning tokens to actually produce batches.
    # "too short" is the one deliberate exception, kept identical every time,
    # to demonstrate the length filter working.
    encyclopedic_templates = [
        "<p>The <b>Python programming language</b> was created by Guido van "
        "Rossum and first released in 1991.</p> It emphasizes code "
        "readability through significant indentation and supports multiple "
        "programming paradigms, including procedural, object-oriented, and "
        "functional programming styles. Over the following decades it grew "
        "into one of the most widely used languages for web development, "
        "scientific computing, automation, and machine learning research, in "
        "large part because of its extensive standard library and "
        "third-party package ecosystem. [1]",
        "# Transformers\nThe transformer architecture, introduced in "
        "'Attention Is All You Need' (Vaswani et al., 2017), relies "
        "entirely on self-attention mechanisms rather than recurrence or "
        "convolution to model relationships between tokens in a sequence. "
        "This design allows far greater parallelization during training "
        "compared to recurrent neural networks, since every position in a "
        "sequence can be processed simultaneously instead of one step at a "
        "time, which is a major reason transformers scaled so well to very "
        "large datasets and model sizes.",
    ]
    news_templates = [
        "Reuters -- Markets rallied today after the central bank signaled "
        "it would hold interest rates steady through the next quarter, "
        "easing concerns among investors about further monetary "
        "tightening. Analysts said the announcement removed a major "
        "source of uncertainty that had been weighing on equity markets "
        "for several weeks, and several major indexes closed higher in "
        "response, with technology and financial shares leading the "
        "gains across most major exchanges.",
    ]
    web_templates = [
        "So I finally got around to rebuilding my old gaming PC this "
        "weekend and honestly it was way easier than I remembered, "
        "mostly just cable management that took forever lol. Swapped in "
        "a new graphics card, added more RAM, and cleaned out about "
        "three years of dust bunnies. Runs so much quieter now and the "
        "frame rates in pretty much everything I tried have basically "
        "doubled compared to before, so overall a pretty solid weekend "
        "project.",
    ]
    REPEATS = 200  # -> a few thousand post-dedup tokens per domain, enough
                   # for the default batch_size=8 / sample_batches=8 to fill
    synthetic = {
        "encyclopedic": [f"{t} (variant {i})" for i in range(REPEATS) for t in encyclopedic_templates],
        "news": [f"{t} (variant {i})" for i in range(REPEATS) for t in news_templates],
        "web": ([f"{t} (variant {i})" for i in range(REPEATS) for t in web_templates]
                + ["too short"] * REPEATS),
    }
    manifest = {}
    for domain, docs in synthetic.items():
        shard_path = raw_dir / f"{domain}.jsonl"
        nbytes = 0
        with open(shard_path, "w", encoding="utf-8") as f:
            for text in docs:
                f.write(json.dumps({"text": text, "domain": domain}) + "\n")
                nbytes += len(text.encode("utf-8"))
        manifest[domain] = {"path": str(shard_path), "bytes": nbytes}
        print(f"[dry_run] {domain}: wrote {nbytes} bytes (synthetic)")
    with open(raw_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return raw_dir


# --------------------------------------------------------------------------
# 2. CLEANING & NORMALIZATION
# --------------------------------------------------------------------------

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_MD_HEADER_RE = re.compile(r"^#{1,6}\s*", re.MULTILINE)
_MD_EMPHASIS_RE = re.compile(r"(\*\*|\*|__|`)")
_REF_MARKER_RE = re.compile(r"\[\d+\]|\[citation needed\]", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str, lowercase: bool = False) -> str:
    text = _HTML_TAG_RE.sub(" ", text)
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _MD_HEADER_RE.sub("", text)
    text = _MD_EMPHASIS_RE.sub("", text)
    text = _REF_MARKER_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    if lowercase:
        text = text.lower()
    return text


def doc_hash(text: str) -> str:
    """Content hash used for exact-duplicate removal. (See report: for a
    larger production run you'd extend this to near-duplicate removal via
    MinHash/LSH -- listed as an optional extension.)"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_corpus(cfg: Config, raw_dir: Path) -> Path:
    """Reads every raw shard, cleans + normalizes + dedups + quality-filters,
    and writes a single clean/corpus.jsonl ready for tokenization.
    """
    clean_dir = Path(cfg.out_dir) / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)
    out_path = clean_dir / "corpus.jsonl"

    seen_hashes = set()
    stats = {"read": 0, "dropped_short": 0, "dropped_dup": 0, "kept": 0}

    with open(out_path, "w", encoding="utf-8") as out_f:
        for shard in sorted(raw_dir.glob("*.jsonl")):
            with open(shard, encoding="utf-8") as in_f:
                for line in in_f:
                    stats["read"] += 1
                    rec = json.loads(line)
                    text = normalize_text(rec["text"], lowercase=cfg.lowercase)

                    if len(text.split()) < cfg.min_words:
                        stats["dropped_short"] += 1
                        continue

                    h = doc_hash(text)
                    if h in seen_hashes:
                        stats["dropped_dup"] += 1
                        continue
                    seen_hashes.add(h)

                    out_f.write(json.dumps({"text": text, "domain": rec["domain"]}) + "\n")
                    stats["kept"] += 1

    with open(clean_dir / "clean_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[clean] {stats}")
    return out_path


# --------------------------------------------------------------------------
# 3. TOKENIZATION
# --------------------------------------------------------------------------

def tokenize_corpus(cfg: Config, clean_path: Path) -> Path:
    """Encodes every cleaned document, concatenates them with an EOS
    separator (standard GPT-style pretraining format), and writes the full
    token-id stream to a single binary file that the packed data loader
    below memory-maps -- this keeps RAM usage low even for corpora far
    larger than what fits in memory.
    """
    from transformers import AutoTokenizer
    import numpy as np

    tok_dir = Path(cfg.out_dir) / "tokenized"
    tok_dir.mkdir(parents=True, exist_ok=True)
    tokens_path = tok_dir / "tokens.bin"

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    if tokenizer.eos_token_id is None:
        tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
    eos_id = tokenizer.eos_token_id
    # We chunk/pack sequences ourselves (see PackedBlockDataset), so the
    # tokenizer's own "sequence too long for the model" check is irrelevant
    # here and would otherwise print one warning per long document -- with
    # a 265k-document corpus that's a lot of noisy, slow terminal output.
    tokenizer.model_max_length = int(1e9)

    n_docs, n_tokens = 0, 0
    # Write incrementally as raw uint16/uint32 to avoid holding all token ids
    # in memory at once. GPT-2's vocab (50257) needs uint32 (uint16 maxes at
    # 65535, which is technically enough here, but uint32 is used for safety
    # with other tokenizers/vocabularies).
    dtype = np.uint32
    with open(clean_path, encoding="utf-8") as f, open(tokens_path, "wb") as out_f:
        for line in f:
            rec = json.loads(line)
            ids = tokenizer.encode(rec["text"])
            ids.append(eos_id)
            np.array(ids, dtype=dtype).tofile(out_f)
            n_docs += 1
            n_tokens += len(ids)

    meta = {
        "tokenizer": cfg.tokenizer_name,
        "vocab_size": tokenizer.vocab_size,
        "dtype": "uint32",
        "n_docs": n_docs,
        "n_tokens": n_tokens,
    }
    with open(tok_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[tokenize] {meta}")
    return tokens_path


# --------------------------------------------------------------------------
# 4a. CUSTOM DATA LOADER -- packed, fixed-length blocks (primary / efficient)
# --------------------------------------------------------------------------

class PackedBlockDataset(IterableDataset):
    """Streams fixed-length (block_size + 1) windows out of the flat token-id
    binary produced by tokenize_corpus(), via numpy memmap so the file is
    never fully loaded into RAM. Each yielded example is a contiguous slice;
    input_ids = tokens[:-1], labels = tokens[1:] (standard causal-LM shift).
    This is the same core scheme used by nanoGPT / GPT-2 reproduction
    training scripts, chosen because it (a) needs no padding at all, so
    every token in every batch is "real" signal, and (b) trivially supports
    shuffling by shuffling block *start offsets* rather than raw text.
    """

    def __init__(self, tokens_path: Path, block_size: int, stride: int = 0,
                 dtype=None, shuffle: bool = True, seed: int = 0):
        import numpy as np
        self.tokens_path = tokens_path
        self.block_size = block_size
        self.step = block_size - stride if stride > 0 else block_size
        self.dtype = dtype or np.uint32
        self.shuffle = shuffle
        self.seed = seed
        self._np = np

        arr = self._np.memmap(tokens_path, dtype=self.dtype, mode="r")
        self.n_tokens = arr.shape[0]
        # +1 because each window needs one extra token for the label shift
        self.n_blocks = max(0, (self.n_tokens - (block_size + 1)) // self.step + 1)

    def __len__(self):
        return self.n_blocks

    def __iter__(self) -> Iterator[torch.Tensor]:
        np = self._np
        arr = np.memmap(self.tokens_path, dtype=self.dtype, mode="r")

        worker_info = torch.utils.data.get_worker_info()
        order = list(range(self.n_blocks))
        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(order)
        if worker_info is not None:  # shard blocks across DataLoader workers
            order = order[worker_info.id::worker_info.num_workers]

        for block_idx in order:
            start = block_idx * self.step
            window = arr[start:start + self.block_size + 1].astype(np.int64)
            if window.shape[0] < self.block_size + 1:
                continue
            x = torch.from_numpy(window[:-1].copy())
            y = torch.from_numpy(window[1:].copy())
            yield x, y


def packed_collate_fn(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.stack(ys)


# --------------------------------------------------------------------------
# 4b. CUSTOM DATA LOADER -- per-document, padded/truncated (comparison loader)
# --------------------------------------------------------------------------

class PaddedDocDataset(Dataset):
    """Map-style dataset over individual (already tokenized) documents,
    kept short/variable length on purpose. Included to demonstrate handling
    of variable-length sequences via padding + attention masks + truncation,
    as distinct from the packing strategy above. Good for reasoning about
    the padding-waste vs packing-efficiency trade-off in your report.
    """

    def __init__(self, doc_token_lists: List[List[int]], max_len: int, pad_id: int):
        self.docs = doc_token_lists
        self.max_len = max_len
        self.pad_id = pad_id

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        ids = self.docs[idx][: self.max_len]  # truncate
        return torch.tensor(ids, dtype=torch.long)


def make_padded_collate_fn(pad_id: int):
    def collate(batch: List[torch.Tensor]):
        max_len = max(t.size(0) for t in batch)
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, t in enumerate(batch):
            input_ids[i, : t.size(0)] = t
            attention_mask[i, : t.size(0)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}
    return collate


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def run(cfg: Config, dry_run: bool):
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    raw_dir = make_dry_run_corpus(cfg) if dry_run else collect_raw_text(cfg)
    clean_path = clean_corpus(cfg, raw_dir)
    tokens_path = tokenize_corpus(cfg, clean_path)

    if tokens_path.stat().st_size == 0:
        raise SystemExit(
            "[error] tokens.bin is empty -- every document was filtered out "
            "before tokenization. Check pipeline_output/clean/clean_stats.json: "
            "if 'kept' is 0, either --min_words is too high for your corpus, or "
            "(in --full mode) a source failed to stream any text. Fix that "
            "before re-running rather than proceeding to the data loader."
        )

    packed_ds = PackedBlockDataset(
        tokens_path, block_size=cfg.block_size, stride=cfg.stride,
        shuffle=True, seed=cfg.seed,
    )
    loader = DataLoader(
        packed_ds, batch_size=cfg.batch_size,
        num_workers=0 if dry_run else cfg.num_workers,
        collate_fn=packed_collate_fn,
    )

    print(f"[loader] {len(packed_ds)} total blocks of size {cfg.block_size}")

    sample_batches = []
    for i, (x, y) in enumerate(loader):
        if i == 0:
            print(f"[loader] example batch shapes: x={tuple(x.shape)} y={tuple(y.shape)}")
        sample_batches.append({"input_ids": x, "labels": y})
        if i + 1 >= cfg.sample_batches:
            break

    sample_path = Path(cfg.out_dir) / "sample_dataset.pt"
    torch.save(sample_batches, sample_path)
    print(f"[done] saved {len(sample_batches)} sample batches -> {sample_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry_run", action="store_true",
                    help="Use tiny synthetic data, no internet required.")
    p.add_argument("--full", action="store_true",
                    help="Run the real pipeline against HF-hub streamed data.")
    p.add_argument("--out_dir", default="pipeline_output")
    p.add_argument("--target_gb", type=float, default=1.0)
    p.add_argument("--min_words", type=int, default=50)
    p.add_argument("--lowercase", action="store_true")
    p.add_argument("--tokenizer_name", default="gpt2")
    p.add_argument("--block_size", type=int, default=1024)
    p.add_argument("--stride", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--sample_batches", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.dry_run and not args.full:
        raise SystemExit("Pass --dry_run (fast, offline smoke test) or "
                          "--full (real ~1GB pipeline). See --help.")

    cfg = Config(
        out_dir=args.out_dir, target_gb=args.target_gb, min_words=args.min_words,
        lowercase=args.lowercase, tokenizer_name=args.tokenizer_name,
        block_size=args.block_size, stride=args.stride, batch_size=args.batch_size,
        num_workers=args.num_workers, sample_batches=args.sample_batches,
        seed=args.seed,
    )
    run(cfg, dry_run=args.dry_run)