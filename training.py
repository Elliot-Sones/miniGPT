from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterator, List

import pandas as pd


def read_texts_from_csv(
    csv_path: Path,
    text_column: str = "text",
    limit: int | None = None,
) -> List[str]:
    """
    Read a CSV file that contains a text column and return a list of strings.

    - csv_path: path to the CSV (expects multi-line text to be quoted, as in TinyShakespeare)
    - text_column: name of the column containing raw text
    - limit: optionally cap the number of rows for quick iterations
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if text_column not in df.columns:
        raise ValueError(
            f"Column '{text_column}' not found in {csv_path}. Available: {list(df.columns)}"
        )

    series = df[text_column].dropna().astype(str)
    if limit is not None:
        series = series.iloc[:limit]
    return series.tolist()


def iter_texts(texts: List[str]) -> Iterator[str]:
    for t in texts:
        # Normalize line endings; keep internal newlines for language modeling
        yield t.replace("\r\n", "\n").replace("\r", "\n").strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load CSV text for tokenization")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("archive/train.csv"),
        help="Path to the training CSV (default: archive/train.csv)",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Name of the text column in the CSV",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of rows to load",
    )

    args = parser.parse_args(argv)

    texts = read_texts_from_csv(args.csv, text_column=args.text_column, limit=args.limit)
    num_texts = len(texts)
    total_chars = sum(len(t) for t in texts)

    print(f"Loaded {num_texts} samples from {args.csv}")
    print(f"Total characters: {total_chars:,}")
    if num_texts:
        preview = texts[0]
        preview_snippet = preview[:200].replace("\n", " ⏎ ")
        print("First sample (truncated to 200 chars, newlines shown as ⏎):")
        print(preview_snippet)

    # This iterator is what you'd feed into a tokenizer trainer next
    _ = iter_texts(texts)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

