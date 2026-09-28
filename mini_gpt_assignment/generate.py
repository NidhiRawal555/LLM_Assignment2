"""
generate.py
-----------
Loads a saved checkpoint and generates a short continuation. Not a listed
requirement, but a nice concrete way to show (in your report) that the
model learned *something* beyond the loss number.

Usage:
    python generate.py runs/exp1/mini_gpt_checkpoint.pt --prompt "hello"
"""

import argparse
import torch

from config import Config
from data import CharTokenizer
from train import load_checkpoint, get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--prompt", type=str, default="the")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--corpus_for_tokenizer", type=str, default=None,
                         help="Path to the raw text used to build the char tokenizer, "
                              "needed only if your Assignment 1 data used the fallback "
                              "CharTokenizer rather than pre-existing integer ids.")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = Config(**ckpt["config"])
    device = get_device(cfg.device)

    model, _, epoch, step = load_checkpoint(args.checkpoint, cfg, device)
    print(f"[generate.py] loaded checkpoint from epoch {epoch}, step {step}")

    if args.corpus_for_tokenizer:
        with open(args.corpus_for_tokenizer) as f:
            tok = CharTokenizer(f.read())
        start_ids = torch.tensor([tok.encode(args.prompt)], dtype=torch.long, device=device)
        out = model.generate(start_ids, args.max_new_tokens)
        print(tok.decode(out[0].tolist()))
    else:
        print(
            "No tokenizer corpus given, so ids can't be decoded back to text. "
            "Re-run with --corpus_for_tokenizer pointing at the text file your "
            "vocabulary was built from (or adapt this script to your own "
            "Assignment 1 tokenizer's decode() function)."
        )


if __name__ == "__main__":
    main()
