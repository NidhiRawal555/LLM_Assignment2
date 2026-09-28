"""
config.py
---------
Central place for every hyperparameter. Keeping them here (instead of
scattered through train.py) makes it trivial to do the "experiment with
hyperparameters" part of the assignment: just instantiate several Config
objects with different values and pass each to train.run_training().
"""

from dataclasses import dataclass


@dataclass
class Config:
    # ---- Model architecture (assignment: 1-2 layers, emb 64-256, 2-4 heads) ----
    n_layer: int = 2           # number of transformer blocks
    n_head: int = 4            # number of attention heads
    n_embd: int = 128          # embedding / hidden dimension (must be divisible by n_head)
    block_size: int = 64       # sequence length per training example (32-128 per spec)
    dropout: float = 0.1

    # ---- Data ----
    vocab_size: int = 0        # filled in automatically once the tokenizer is built
    # Points at the tokens.bin written by data_collection_preprocessing.py's
    # tokenize_corpus() step (sibling meta.json is read automatically).
    # Adjust the leading "../" if pipeline_output/ lives somewhere else
    # relative to where you run train.py from.
    data_path: str = "../pipeline_output/tokenized/tokens.bin"
    fallback_raw_text_path: str = "data/sample_corpus.txt"  # used only if data_path is missing

    # ---- Training ----
    batch_size: int = 32
    learning_rate: float = 3e-4
    max_epochs: int = 10
    eval_every: int = 100      # steps between quick validation-loss checks
    grad_clip: float = 1.0
    weight_decay: float = 0.01

    # ---- Housekeeping ----
    seed: int = 1337
    device: str = "cuda"       # falls back to "cpu" automatically in train.py if unavailable
    out_dir: str = "runs/exp1"
    checkpoint_name: str = "mini_gpt_checkpoint.pt"

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"