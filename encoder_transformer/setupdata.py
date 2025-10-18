"""
Simple monolingual English data setup for MLM.

Usage (from repo root or encoder_transformer/):
  python -m encoder_transformer.setupdata \
      --dataset wikitext-2 \
      --out_dir encoder_transformer/archive_mlm

Writes three CSV files with a single column 'en':
  - encoder_transformer/archive_mlm/train.csv
  - encoder_transformer/archive_mlm/val.csv
  - encoder_transformer/archive_mlm/test.csv

Default dataset is a small, fast set (wikitext-2). You can switch to
larger corpora via --dataset.
"""

from __future__ import annotations

import argparse
import os
import re
import unicodedata
from typing import List


def _require(module_name: str, pip_hint: str):
    try:
        return __import__(module_name)
    except Exception as e:
        raise RuntimeError(
            f"Missing dependency: {module_name}. Install with: pip install {pip_hint}.\n"
            f"Original error: {e}"
        )


datasets = _require("datasets", "datasets")
pd = _require("pandas", "pandas")


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u200b", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def save_split(lines: List[str], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame({"en": lines})
    df.to_csv(path, index=False)


def load_wikitext_2() -> dict:
    # Uses raw variant to avoid tokenized artifacts.
    ds = datasets.load_dataset("wikitext", "wikitext-2-raw-v1")
    return {
        "train": [r["text"] for r in ds["train"]],
        "val": [r["text"] for r in ds["validation"]],
        "test": [r["text"] for r in ds["test"]],
    }


DATASETS = {
    "wikitext-2": load_wikitext_2,
}


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare monolingual EN data for MLM")
    p.add_argument("--dataset", type=str, default="wikitext-2", choices=sorted(DATASETS.keys()))
    p.add_argument("--out_dir", type=str, default=os.path.join("archive_mlm"))
    p.add_argument("--min_chars", type=int, default=1, help="Drop lines shorter than this many chars after cleaning")
    p.add_argument("--max_chars", type=int, default=1024, help="Drop lines longer than this many chars")
    p.add_argument("--lowercase", action="store_true", help="Lowercase during normalization")
    return p


def main():
    args = build_argparser().parse_args()
    if args.dataset not in DATASETS:
        raise SystemExit(f"Unknown dataset {args.dataset}")
    print(f"Loading dataset: {args.dataset}")
    raw = DATASETS[args.dataset]()

    def clean(lines: List[str]) -> List[str]:
        out = []
        for s in lines:
            s = normalize_text(s)
            if args.lowercase:
                s = s.lower()
            if not s:
                continue
            if len(s) < args.min_chars or len(s) > args.max_chars:
                continue
            out.append(s)
        return out

    train = clean(raw.get("train", []))
    val = clean(raw.get("val", raw.get("validation", [])))
    test = clean(raw.get("test", []))

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    save_split(train, os.path.join(out_dir, "train.csv"))
    save_split(val, os.path.join(out_dir, "val.csv"))
    save_split(test, os.path.join(out_dir, "test.csv"))
    print(
        f"Wrote train/val/test to {out_dir} -> "
        f"train.csv({len(train)}), val.csv({len(val)}), test.csv({len(test)})"
    )


if __name__ == "__main__":
    main()

