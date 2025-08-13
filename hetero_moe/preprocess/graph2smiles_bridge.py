"""Bridge script to reuse references/Graph2SMILES preprocessing.

This script wires the well-tested Graph2SMILES tokenizer/vocab + graph featurizer
to produce NPZ files under hetero_moe/data/processed/uspto.

Usage (from repo root):
  python -m hetero_moe.preprocess.graph2smiles_bridge \
    --raw_dir hetero_moe/data/raw/uspto \
    --out_dir hetero_moe/data/processed/uspto/graph2smiles_npz \
    --model g2s --repr smiles --max_src_len 512 --max_tgt_len 512 --workers 4

Expected files in raw_dir:
  train.src, train.tgt, val.src, val.tgt, test.src, test.tgt

Outputs:
  - vocab_smiles.txt
  - {train,val,test}_0.npz with token ids and graph features
"""

from __future__ import annotations

import argparse
import os
import sys


def _get_repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _import_graph2smiles_preprocess():
    repo_root = _get_repo_root()
    g2s_dir = os.path.join(repo_root, "references", "Graph2SMILES")
    if g2s_dir not in sys.path:
        sys.path.insert(0, g2s_dir)
    try:
        import preprocess as g2s_pre
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            f"Failed to import Graph2SMILES preprocess module from {g2s_dir}. "
            f"Ensure the references/Graph2SMILES repo is present."
        ) from exc
    return g2s_pre


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("graph2smiles_bridge")
    parser.add_argument("--raw_dir", type=str, default=os.path.join("hetero_moe", "data", "raw", "uspto"))
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join("hetero_moe", "data", "processed", "uspto", "graph2smiles_npz"),
    )
    parser.add_argument("--model", choices=["s2s", "g2s"], default="g2s")
    parser.add_argument("--repr", choices=["smiles", "selfies"], default="smiles")
    parser.add_argument("--max_src_len", type=int, default=512)
    parser.add_argument("--max_tgt_len", type=int, default=512)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    def pick(existing_candidates: list[str]) -> str:
        for cand in existing_candidates:
            if os.path.exists(cand):
                return cand
        return ""

    # Support multiple naming schemes
    train_src = pick([
        os.path.join(args.raw_dir, "train.src"),
        os.path.join(args.raw_dir, "src-train.txt"),
    ])
    train_tgt = pick([
        os.path.join(args.raw_dir, "train.tgt"),
        os.path.join(args.raw_dir, "tgt-train.txt"),
    ])
    val_src = pick([
        os.path.join(args.raw_dir, "val.src"),
        os.path.join(args.raw_dir, "src-val.txt"),
    ])
    val_tgt = pick([
        os.path.join(args.raw_dir, "val.tgt"),
        os.path.join(args.raw_dir, "tgt-val.txt"),
    ])
    test_src = pick([
        os.path.join(args.raw_dir, "test.src"),
        os.path.join(args.raw_dir, "src-test.txt"),
    ])
    test_tgt = pick([
        os.path.join(args.raw_dir, "test.tgt"),
        os.path.join(args.raw_dir, "tgt-test.txt"),
    ])

    # Validate presence of expected files early
    missing = [p for p in [train_src, train_tgt, val_src, val_tgt, test_src, test_tgt] if not p or not os.path.exists(p)]
    if missing:
        missing_rel = [os.path.relpath(p) for p in missing]
        raise FileNotFoundError(
            f"Missing expected raw files in {args.raw_dir}: {', '.join(missing_rel)}"
        )

    g2s_pre = _import_graph2smiles_preprocess()

    # Build the argument list for Graph2SMILES preprocess
    g2s_cli = [
        "--model", args.model,
        "--task", "reaction_prediction",
        "--representation_start", "smiles",
        "--representation_end", args.repr,
        "--max_src_len", str(args.max_src_len),
        "--max_tgt_len", str(args.max_tgt_len),
        "--num_workers", str(args.workers),
        "--do_tokenize",
        "--train_src", train_src,
        "--train_tgt", train_tgt,
        "--val_src", val_src,
        "--val_tgt", val_tgt,
        "--test_src", test_src,
        "--test_tgt", test_tgt,
        "--preprocess_output_path", args.out_dir,
    ]

    parser = g2s_pre.get_preprocess_parser()
    g2s_args = parser.parse_args(g2s_cli)

    # Run preprocessing
    g2s_pre.preprocess_main(g2s_args)


if __name__ == "__main__":  # pragma: no cover
    main()


