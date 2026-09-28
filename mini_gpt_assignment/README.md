# Mini-GPT from Scratch — Assignment 2

A small GPT-style transformer trained for next-token prediction, built with
plain PyTorch (no `nn.TransformerEncoder` shortcuts) so every piece of the
architecture is visible and explainable.

## Files

| File | What it does |
|---|---|
| `config.py` | All hyperparameters in one dataclass |
| `data.py` | Loads your Assignment 1 tokenized data (or falls back to a placeholder corpus) and builds training batches |
| `model.py` | The `MiniGPT` model: embeddings, causal multi-head attention, feed-forward blocks, output head |
| `train.py` | Training loop, loss/perplexity logging to CSV, checkpoint save/load |
| `sweep.py` | Runs a set of hyperparameter variations (learning rate, batch size, layers, embedding size) |
| `visualize.py` | Plots loss + perplexity curves from the CSV logs |
| `generate.py` | Loads a checkpoint and generates a short text sample |

## 1. Plug in your Assignment 1 dataset

This is the one thing you must change before your results mean anything.
In `config.py`, set `data_path` to whichever of these you have:

- `data/assignment1_tokens.pt` — a saved `torch.LongTensor` of token ids
- `data/assignment1_tokens.npy` — a saved numpy array of token ids
- `data/assignment1_clean.txt` — cleaned raw text (a simple char-level
  tokenizer will be fit on it automatically)

If none of those paths exist, `data.py` silently falls back to a tiny
bundled placeholder corpus **only so the pipeline is runnable out of the
box** — it will print a warning when it does this. Don't use results from
the placeholder corpus in your report.

If your Assignment 1 tokenizer already produced integer ids with a custom
vocabulary (e.g. a BPE tokenizer), the `.pt`/`.npy` path is what you want —
just save `ids` and `data.py` will use them directly and infer `vocab_size`.

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

Use a machine/notebook with a GPU (e.g. Google Colab, free tier is enough
for this model size) — training is much faster than on CPU, though CPU
will still work for a model this small.

## 3. Train

```bash
python train.py
```

This trains one run using the settings in `config.py`, prints progress,
and writes:
- `runs/exp1/metrics_baseline.csv` — per-epoch loss/perplexity log
- `runs/exp1/mini_gpt_checkpoint.pt` — final model + optimizer checkpoint

## 4. Run the hyperparameter sweep

```bash
python sweep.py
```

Trains 8 variations (2 learning rates, 2 batch sizes, 1 vs 2 layers, 2
embedding sizes) and logs each to its own CSV under `runs/sweep/`. Use
these to write the "hyperparameter experiments" section of the report.

## 5. Visualize

```bash
python visualize.py runs/exp1/metrics_baseline.csv
python visualize.py "runs/sweep/*.csv"   # overlay every sweep run
```

Produces `training_curves.png` (loss + perplexity side by side).

## 6. Generate a sample (optional, good for the report)

```bash
python generate.py runs/exp1/mini_gpt_checkpoint.pt --prompt "the movie" \
    --corpus_for_tokenizer data/assignment1_clean.txt
```

## Checkpointing details

`save_checkpoint` (in `train.py`) stores `model_state_dict`,
`optimizer_state_dict`, the full `config`, and the epoch/step — enough to
either resume training or reload the model purely for inference via
`load_checkpoint`.

## Writing the report

See `report_template.docx` for a skeleton with a section for each graded
item (architecture, dataset, training setup, hyperparameter experiments,
observations/challenges). Fill in the actual numbers and screenshots of
`training_curves.png` after you run the code on your own Assignment 1 data —
the numbers depend on your dataset, so they can't be filled in ahead of time.
