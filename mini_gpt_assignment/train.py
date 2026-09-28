"""
train.py
--------
The training loop, plus checkpoint save/load and CSV metric logging.

Run directly for a single training run:
    python train.py

Or import run_training(cfg) from a notebook / sweep script to try several
hyperparameter configurations (see sweep.py).
"""

import os
import csv
import math
import time
import torch

from config import Config
from data import load_token_ids, make_splits, get_batch
from model import MiniGPT


def get_device(requested: str) -> str:
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    if requested == "mps" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def estimate_loss(model, data, cfg, eval_iters=20):
    """Average loss over a handful of random batches (cheaper than a full pass)."""
    model.eval()
    losses = torch.zeros(eval_iters)
    for i in range(eval_iters):
        x, y = get_batch(data, cfg.batch_size, cfg.block_size, cfg.device)
        _, loss = model(x, y)
        losses[i] = loss.item()
    model.train()
    return losses.mean().item()


def perplexity(loss: float) -> float:
    """Perplexity = exp(cross-entropy loss). Lower is better; it's the
    standard metric for language-model quality (roughly: the effective
    number of tokens the model is 'choosing between' at each step)."""
    return math.exp(loss) if loss < 20 else float("inf")  # guard against overflow


def save_checkpoint(model, optimizer, cfg, epoch, step, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg.__dict__,
            "epoch": epoch,
            "step": step,
        },
        path,
    )
    print(f"[train.py] checkpoint saved -> {path}")


def load_checkpoint(path, cfg, device):
    """Rebuilds a MiniGPT + optimizer from a saved checkpoint, so training
    can resume, or the model can be used purely for evaluation/generation."""
    ckpt = torch.load(path, map_location=device)
    model = MiniGPT(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return model, optimizer, ckpt["epoch"], ckpt["step"]


def run_training(cfg: Config, log_tag: str = "run"):
    """Full training run for one Config. Returns the path to the metrics
    CSV so visualize.py can plot it, and the path to the final checkpoint."""
    torch.manual_seed(cfg.seed)
    cfg.device = get_device(cfg.device)
    print(f"[train.py] device = {cfg.device}")

    # ---- data ----
    ids, vocab_size = load_token_ids(cfg)
    cfg.vocab_size = vocab_size
    train_ids, val_ids = make_splits(ids)
    steps_per_epoch = min(max(1, len(train_ids) // (cfg.batch_size * cfg.block_size)), cfg.max_steps_per_epoch)

    # ---- model / optimizer ----
    model = MiniGPT(cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    # ---- logging setup ----
    os.makedirs(cfg.out_dir, exist_ok=True)
    csv_path = os.path.join(cfg.out_dir, f"metrics_{log_tag}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "step", "train_loss", "val_loss", "val_perplexity", "seconds"])

    start_time = time.time()
    global_step = 0
    for epoch in range(cfg.max_epochs):
        epoch_losses = []
        for _ in range(steps_per_epoch):
            x, y = get_batch(train_ids, cfg.batch_size, cfg.block_size, cfg.device)

            logits, loss = model(x, y)           # forward pass
            optimizer.zero_grad(set_to_none=True)
            loss.backward()                       # backward pass
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()                       # optimizer step

            epoch_losses.append(loss.item())
            global_step += 1

            if global_step % cfg.eval_every == 0:
                val_loss = estimate_loss(model, val_ids, cfg)
                print(
                    f"  step {global_step:5d} | train_loss {loss.item():.4f} "
                    f"| val_loss {val_loss:.4f} | val_ppl {perplexity(val_loss):.2f}"
                )

        avg_train_loss = sum(epoch_losses) / len(epoch_losses)
        val_loss = estimate_loss(model, val_ids, cfg)
        val_ppl = perplexity(val_loss)
        elapsed = time.time() - start_time

        print(
            f"[epoch {epoch+1}/{cfg.max_epochs}] avg_train_loss={avg_train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_perplexity={val_ppl:.2f} ({elapsed:.1f}s elapsed)"
        )
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, global_step, avg_train_loss, val_loss, val_ppl, round(elapsed, 1)])

    ckpt_path = os.path.join(cfg.out_dir, cfg.checkpoint_name)
    save_checkpoint(model, optimizer, cfg, cfg.max_epochs, global_step, ckpt_path)

    return csv_path, ckpt_path


if __name__ == "__main__":
    cfg = Config()
    run_training(cfg, log_tag="baseline")
