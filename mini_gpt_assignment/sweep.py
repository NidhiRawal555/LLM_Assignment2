"""
sweep.py
--------
Runs several Config variations back-to-back so you can compare them in
the report (learning rate, batch size, n_layer, n_embd -- exactly the
knobs the assignment asks you to experiment with). Each run gets its own
tagged CSV log and checkpoint under runs/.

Usage:
    python sweep.py
"""

from dataclasses import replace
from config import Config
from train import run_training

BASE = Config(out_dir="runs/sweep", max_epochs=5, data_path="../pipeline_output/tokenized/tokens.bin")

EXPERIMENTS = {
    "lr_1e-3": replace(BASE, learning_rate=1e-3),
    "lr_5e-4": replace(BASE, learning_rate=5e-4),
    "batch_16": replace(BASE, batch_size=16),
    "batch_64": replace(BASE, batch_size=64),
    "1_layer": replace(BASE, n_layer=1),
    "2_layer": replace(BASE, n_layer=2),
    "embd_64": replace(BASE, n_embd=64, n_head=4),
    "embd_256": replace(BASE, n_embd=256, n_head=4),
}

if __name__ == "__main__":
    results = {}
    for tag, cfg in EXPERIMENTS.items():
        print(f"\n===== running experiment: {tag} =====")
        csv_path, ckpt_path = run_training(cfg, log_tag=tag)
        results[tag] = {"csv": csv_path, "checkpoint": ckpt_path}

    print("\nAll experiments complete:")
    for tag, paths in results.items():
        print(f"  {tag}: {paths['csv']}")
