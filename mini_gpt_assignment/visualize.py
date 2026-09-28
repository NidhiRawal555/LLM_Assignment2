"""
visualize.py
------------
Reads the metrics_*.csv files produced by train.py / sweep.py and plots
loss and perplexity curves -- the "Visualizations of training loss and
perplexity curves" deliverable.

Usage:
    python visualize.py runs/exp1/metrics_baseline.csv
    python visualize.py runs/sweep/*.csv        # overlay multiple runs
"""

import sys
import glob
import csv
import matplotlib.pyplot as plt


def read_metrics(path):
    epochs, train_loss, val_loss, val_ppl = [], [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))
            val_ppl.append(float(row["val_perplexity"]))
    return epochs, train_loss, val_loss, val_ppl


def plot_all(paths, out_path="training_curves.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for path in paths:
        tag = path.split("metrics_")[-1].replace(".csv", "")
        epochs, train_loss, val_loss, val_ppl = read_metrics(path)
        ax1.plot(epochs, train_loss, label=f"{tag} (train)", linestyle="--")
        ax1.plot(epochs, val_loss, label=f"{tag} (val)")
        ax2.plot(epochs, val_ppl, label=tag)

    ax1.set_xlabel("epoch")
    ax1.set_ylabel("cross-entropy loss")
    ax1.set_title("Training / validation loss")
    ax1.legend(fontsize=8)

    ax2.set_xlabel("epoch")
    ax2.set_ylabel("perplexity")
    ax2.set_title("Validation perplexity")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"[visualize.py] saved -> {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python visualize.py <metrics_csv_path_or_glob> [...]")
        sys.exit(1)

    paths = []
    for arg in sys.argv[1:]:
        paths.extend(glob.glob(arg))
    if not paths:
        print("No CSV files matched.")
        sys.exit(1)

    plot_all(paths)
