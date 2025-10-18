r"""
Masked Language Modeling (MLM) trainer that uses the Encoder as a backbone.

What this file does (high level):
- Loads monolingual English text prepared by setupdata.py.
- Trains (or loads) a tokenizer and builds batches.
- Applies BERT-style masking on-the-fly for MLM.
- Runs the Encoder and a tied output head to predict masked tokens.
- Trains with cross-entropy on masked positions; reports val/test.

Usage (super simple):
- Activate your venv once
    Mac/Linux:  source .venv/bin/activate
    Windows:    .venv\Scripts\activate
- From project root (or after cd encoder_transformer):
    python encoder_transformer/setupdata.py   # writes encoder_transformer/archive_mlm/*.csv
    python encoder_transformer/mlm.py         # trains + evaluates MLM

Notes:
- Encoder API is unchanged; this is a thin training harness around it.
- All key configuration lives in GLOBAL_DEFAULTS below; CLI flags can override.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

"""Import Encoder both as a package and as a direct script.
Allows: `python encoder_transformer/mlm.py` and `python -m encoder_transformer.mlm`.
"""
try:
    # When run as a package module
    from .encode import Encoder, EncoderConfig
except Exception:  # pragma: no cover - fallback for direct script execution
    import pathlib, sys as _sys
    _sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
    from encoder_transformer.encode import Encoder, EncoderConfig


def _require(module_name: str, pip_hint: str):
    try:
        return __import__(module_name)
    except Exception as e:
        raise RuntimeError(
            f"Missing dependency: {module_name}. Install with: pip install {pip_hint}.\n"
            f"Original error: {e}"
        )


pd = _require("pandas", "pandas")
tokenizers = _require("tokenizers", "tokenizers")


###############################################################################
# Section: Configuration (single source of truth for defaults)
###############################################################################

SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

# Make the default data_dir robust whether you run from project root or from
# inside the encoder_transformer folder.
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "archive_mlm")

GLOBAL_DEFAULTS = {
    "data_dir": DEFAULT_DATA_DIR,
    "vocab_size": 30000,
    "max_len": 128,
    "batch_size": 32,
    "epochs": 3,
    "lr": 3e-4,
    "weight_decay": 0.01,
    "warmup_steps": 0,
    "seed": 1337,
    "device": "mps",  # 'auto' | 'cpu' | 'cuda' | 'mps'
    "mask_prob": 0.15,
    "max_grad_norm": 1.0,
}


###############################################################################
# Section: Tokenizer (train once and cache)
###############################################################################

def build_or_load_tokenizer(train_texts: List[str], out_dir: str, vocab_size: int = 30000):
    """Train a BPE tokenizer on the provided texts or load a cached one.

    Saved to `<out_dir>/tokenizer.json` so subsequent runs reuse the same vocab
    and special token IDs (PAD/UNK/CLS/SEP/MASK).
    """
    os.makedirs(out_dir, exist_ok=True)
    tok_path = os.path.join(out_dir, "tokenizer.json")
    if os.path.exists(tok_path):
        tok = tokenizers.Tokenizer.from_file(tok_path)
        return tok

    # Train a simple BPE tokenizer
    model = tokenizers.models.BPE(unk_token="[UNK]")
    tok = tokenizers.Tokenizer(model)
    tok.normalizer = tokenizers.normalizers.Sequence([
        tokenizers.normalizers.NFKC()
    ])
    tok.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    trainer = tokenizers.trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIALS)

    class _Iterator:
        def __init__(self, texts):
            self.texts = texts
        def __iter__(self):
            for t in self.texts:
                if t:
                    yield t

    tok.train_from_iterator(_Iterator(train_texts), trainer=trainer)
    tok.save(tok_path)
    return tok


###############################################################################
# Section: Data utilities (padding and attention masks)
###############################################################################

def ids_and_pad(tokens: List[int], max_len: int, pad_id: int) -> List[int]:
    """Truncate/pad a list of token IDs to exactly `max_len`, using `pad_id`."""
    t = tokens[:max_len]
    if len(t) < max_len:
        t = t + [pad_id] * (max_len - len(t))
    return t


def build_attention_mask(input_ids: torch.LongTensor, pad_id: int) -> torch.Tensor:
    """Create a 1/0 mask matching Encoder expectations (1=keep, 0=pad)."""
    return (input_ids != pad_id).to(input_ids.dtype)


###############################################################################
# Section: Masking (BERT-style 15% with 80/10/10 replacement)
###############################################################################

def create_mlm_targets(
    input_ids: torch.LongTensor,
    pad_id: int,
    mask_id: int,
    vocab_size: int,
    mask_prob: float = GLOBAL_DEFAULTS["mask_prob"],
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """
    Apply BERT-style masking to a batch of input_ids (B, S).
    Returns (masked_input_ids, labels) where labels[i,j] is -100 for non-masked.
    """
    device = input_ids.device
    B, S = input_ids.shape
    labels = input_ids.clone()

    # Do not consider PAD tokens for masking
    is_token = input_ids != pad_id
    mask = (torch.rand((B, S), device=device) < mask_prob) & is_token

    labels[~mask] = -100  # ignore non-masked

    # 80% -> [MASK]
    replace_mask = torch.rand((B, S), device=device)
    mask_mask = mask & (replace_mask < 0.8)
    # 10% -> random token
    rand_mask = mask & (replace_mask >= 0.8) & (replace_mask < 0.9)
    # 10% -> keep original (do nothing)

    out = input_ids.clone()
    out[mask_mask] = mask_id
    if rand_mask.any():
        out[rand_mask] = torch.randint(low=0, high=vocab_size, size=(rand_mask.sum().item(),), device=device)
        # Avoid sampling PAD as random replacement (optional)
        out[rand_mask] = torch.where(out[rand_mask] == pad_id, mask_id, out[rand_mask])

    return out, labels


###############################################################################
# Section: Dataset (tokenize each line lazily)
###############################################################################

class MLMDataset(Dataset):
    def __init__(self, texts: List[str], tok, max_len: int, pad_id: int):
        self.texts = texts
        self.tok = tok
        self.max_len = max_len
        self.pad_id = pad_id

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int):
        t = self.texts[idx]
        enc = self.tok.encode(t)
        ids = ids_and_pad(enc.ids, self.max_len, self.pad_id)
        return torch.tensor(ids, dtype=torch.long)


###############################################################################
# Section: Model heads (tied output projection)
###############################################################################

class TiedOutputHead(nn.Module):
    def __init__(self, embed_weight: nn.Parameter, embed_dim: int, vocab_size: int):
        super().__init__()
        self.embed_weight = embed_weight  # weight tying (no gradient assignment here; comes from embedding)
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)
        nn.init.eye_(self.proj.weight)  # start as identity

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # hidden: [B, S, D]
        # logits = hidden @ W^T + b  (tied W)
        h = self.proj(hidden)
        logits = torch.matmul(h, self.embed_weight.t()) + self.bias
        return logits


###############################################################################
# Section: Full model (Encoder + tied head)
###############################################################################

class MaskedLanguageModel(nn.Module):
    def __init__(self, encoder: Encoder, vocab_size: int):
        super().__init__()
        self.encoder = encoder
        # Tie to token embedding weights
        tok_emb = self.encoder.embeddings.token_embedding
        self.lm_head = TiedOutputHead(tok_emb.weight, encoder.config.embed_dim, vocab_size)

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(input_ids, attention_mask)  # [B, S, D]
        logits = self.lm_head(hidden)                    # [B, S, V]
        return logits


###############################################################################
# Section: Arguments container
###############################################################################

@dataclass
class TrainArgs:
    data_dir: str
    vocab_size: int
    max_len: int
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    warmup_steps: int
    seed: int
    device: str
    mask_prob: float
    max_grad_norm: float


###############################################################################
# Section: Reproducibility helpers
###############################################################################

def set_seed(seed: int):
    import random
    import numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


###############################################################################
# Section: I/O (load CSVs with a single column 'en')
###############################################################################

def load_texts(data_dir: str) -> Tuple[List[str], List[str], List[str]]:
    def col(path):
        df = pd.read_csv(path)
        if "en" not in df.columns:
            raise RuntimeError(f"Expected column 'en' in {path}")
        return df["en"].astype(str).tolist()
    train = col(os.path.join(data_dir, "train.csv"))
    val = col(os.path.join(data_dir, "val.csv"))
    test = col(os.path.join(data_dir, "test.csv"))
    return train, val, test


###############################################################################
# Section: Evaluation (masked-token loss and accuracy)
###############################################################################

def evaluate(model: MaskedLanguageModel, data_loader: DataLoader, pad_id: int, mask_id: int, device: torch.device, mask_prob: float) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_masked = 0
    with torch.no_grad():
        for input_ids in data_loader:
            input_ids = input_ids.to(device)
            attn = build_attention_mask(input_ids, pad_id)
            masked_ids, labels = create_mlm_targets(
                input_ids, pad_id, mask_id, model.lm_head.bias.numel(), mask_prob=mask_prob
            )
            logits = model(masked_ids, attn)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
            total_loss += loss.item() * input_ids.size(0)

            # Accuracy on masked positions
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                mask = labels != -100
                total_masked += mask.sum().item()
                if total_masked > 0:
                    total_correct += (preds[mask] == labels[mask]).sum().item()

    avg_loss = total_loss / max(1, len(data_loader.dataset))
    acc = (total_correct / max(1, total_masked)) if total_masked > 0 else 0.0
    return avg_loss, acc


###############################################################################
# Section: Device selection (MPS/CUDA/CPU)
###############################################################################

def resolve_device(sel: str) -> torch.device:
    sel = (sel or "auto").lower()
    if sel == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        print("[warn] MPS requested but not available; falling back to CPU.")
        return torch.device("cpu")
    if sel == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if sel in ("cpu",):
        return torch.device("cpu")
    # auto: prefer mps, then cuda, then cpu
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


###############################################################################
# Section: Training loop (fit on train, select on val, test at end)
###############################################################################

def train_loop(args: TrainArgs):
    set_seed(args.seed)
    device = resolve_device(args.device)

    train_texts, val_texts, test_texts = load_texts(args.data_dir)
    tok = build_or_load_tokenizer(train_texts, out_dir=args.data_dir, vocab_size=args.vocab_size)

    # IDs for specials
    pad_id = tok.token_to_id("[PAD]")
    unk_id = tok.token_to_id("[UNK]")
    mask_id = tok.token_to_id("[MASK]")
    if pad_id is None or unk_id is None or mask_id is None:
        raise RuntimeError("Tokenizer missing required special tokens [PAD]/[UNK]/[MASK].")

    # Datasets
    train_ds = MLMDataset(train_texts, tok, args.max_len, pad_id)
    val_ds = MLMDataset(val_texts, tok, args.max_len, pad_id)
    test_ds = MLMDataset(test_texts, tok, args.max_len, pad_id)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # Model: build Encoder with the tokenizer's vocab and PAD id
    config = EncoderConfig(
        src_vocab_size=tok.get_vocab_size(),
        pad_token_id=pad_id,
    )
    encoder = Encoder(config).to(device)
    model = MaskedLanguageModel(encoder, vocab_size=tok.get_vocab_size()).to(device)

    # Log run configuration summary before training starts
    try:
        dev_name = str(device)
    except Exception:
        dev_name = device.type if hasattr(device, "type") else "unknown"
    print(
        "Run setup: "
        f"device={dev_name}, "
        f"train_samples={len(train_ds)}, val_samples={len(val_ds)}, test_samples={len(test_ds)}, "
        f"vocab_size={tok.get_vocab_size()}, max_len={args.max_len}, batch_size={args.batch_size}"
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Simple training loop with validation
    best_val = float("inf")
    ckpt_path = os.path.join(args.data_dir, "checkpoint.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0
        total_loss = 0.0
        for input_ids in train_loader:
            input_ids = input_ids.to(device)
            attn = build_attention_mask(input_ids, pad_id)
            masked_ids, labels = create_mlm_targets(
                input_ids, pad_id, mask_id, tok.get_vocab_size(), mask_prob=args.mask_prob
            )

            logits = model(masked_ids, attn)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            bs = input_ids.size(0)
            total += bs
            total_loss += loss.item() * bs

        train_loss = total_loss / max(1, total)
        val_loss, val_acc = evaluate(model, val_loader, pad_id, mask_id, device, args.mask_prob)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model": model.state_dict(),
                "config": config.__dict__,
                "tokenizer_path": os.path.join(args.data_dir, "tokenizer.json"),
            }, ckpt_path)
            print(f"Saved checkpoint -> {ckpt_path}")

    # Final test eval (load best)
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
    test_loss, test_acc = evaluate(model, test_loader, pad_id, mask_id, device, args.mask_prob)
    print(f"Test: loss={test_loss:.4f} acc={test_acc:.4f}")


###############################################################################
# Section: CLI (optional overrides; defaults come from GLOBAL_DEFAULTS)
###############################################################################

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a simple MLM using the Encoder")
    p.add_argument("--data_dir", type=str, default=GLOBAL_DEFAULTS["data_dir"]) 
    p.add_argument("--vocab_size", type=int, default=GLOBAL_DEFAULTS["vocab_size"]) 
    p.add_argument("--max_len", type=int, default=GLOBAL_DEFAULTS["max_len"]) 
    p.add_argument("--batch_size", type=int, default=GLOBAL_DEFAULTS["batch_size"]) 
    p.add_argument("--epochs", type=int, default=GLOBAL_DEFAULTS["epochs"]) 
    p.add_argument("--lr", type=float, default=GLOBAL_DEFAULTS["lr"]) 
    p.add_argument("--weight_decay", type=float, default=GLOBAL_DEFAULTS["weight_decay"]) 
    p.add_argument("--warmup_steps", type=int, default=GLOBAL_DEFAULTS["warmup_steps"]) 
    p.add_argument("--seed", type=int, default=GLOBAL_DEFAULTS["seed"]) 
    p.add_argument("--device", type=str, default=GLOBAL_DEFAULTS["device"], choices=["auto","cpu","cuda","mps"]) 
    p.add_argument("--mask_prob", type=float, default=GLOBAL_DEFAULTS["mask_prob"]) 
    p.add_argument("--max_grad_norm", type=float, default=GLOBAL_DEFAULTS["max_grad_norm"]) 
    return p


###############################################################################
# Section: Entrypoint
###############################################################################

def main():
    args_ns = build_argparser().parse_args()
    args = TrainArgs(
        data_dir=args_ns.data_dir,
        vocab_size=args_ns.vocab_size,
        max_len=args_ns.max_len,
        batch_size=args_ns.batch_size,
        epochs=args_ns.epochs,
        lr=args_ns.lr,
        weight_decay=args_ns.weight_decay,
        warmup_steps=args_ns.warmup_steps,
        seed=args_ns.seed,
        device=args_ns.device,
        mask_prob=args_ns.mask_prob,
        max_grad_norm=args_ns.max_grad_norm,
    )
    train_loop(args)


if __name__ == "__main__":
    main()
