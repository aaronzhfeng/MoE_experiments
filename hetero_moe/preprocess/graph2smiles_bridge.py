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
from typing import Optional


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

    # Suppress RDKit verbosity if available
    try:
        from rdkit import RDLogger  # type: ignore
        RDLogger.DisableLog('rdApp.*')
    except Exception:
        pass

    # If inputs look space-tokenized, create de-tokenized temporary copies for RDKit parsing
    def _needs_detok(path: str) -> bool:
        try:
            with open(path, "r") as f:
                for _ in range(5):
                    line = f.readline()
                    if not line:
                        break
                    # Heuristic: spaces or reaction arrows suggest sanitization needed
                    if (" " in line) or (">>" in line) or (">" in line):
                        return True
            return False
        except Exception:
            return False

    def _sanitize_one(s: str, *, is_src: bool, stats: Optional[dict] = None) -> str:
        s = s.replace(" ", "").strip()
        if not s:
            return "CC"
        # If reaction arrow present, pick side based on file role
        if ">>" in s:
            left, right = s.split(">>", 1)
            s = left if is_src else right
            if stats is not None:
                stats["arrow_split"] = stats.get("arrow_split", 0) + 1
        elif ">" in s:
            parts = s.split(">")
            # common reaction format: reactants>reagents>products
            if len(parts) >= 3:
                s = parts[0] if is_src else parts[-1]
            else:
                s = parts[0] if is_src else parts[-1]
            if stats is not None:
                stats["arrow_split"] = stats.get("arrow_split", 0) + 1
        try:
            from rdkit import Chem  # type: ignore
            use_rdkit = True
        except Exception:
            use_rdkit = False

        if not use_rdkit:
            return s or "CC"
        try:
            mol = Chem.MolFromSmiles(s)
            if mol is not None:
                can = Chem.MolToSmiles(mol)
                if stats is not None and can != s:
                    stats["canonicalized"] = stats.get("canonicalized", 0) + 1
                return can
        except Exception:
            pass
        # Salvage: split mixture and keep valid parts
        salvage = []
        for part in s.split('.'):
            if not part:
                continue
            try:
                m = Chem.MolFromSmiles(part)
                if m is not None:
                    salvage.append(Chem.MolToSmiles(m))
            except Exception:
                continue
        if salvage:
            if stats is not None:
                stats["salvaged"] = stats.get("salvaged", 0) + 1
            return ".".join(salvage)
        return "CC"

    def _detok_file(src_path: str, dst_path: str, *, is_src: bool) -> dict:
        # Best-effort de-tokenize and canonicalize with RDKit; fallback to plain strip
        stats = {"replaced": 0, "arrow_split": 0, "canonicalized": 0, "salvaged": 0, "total": 0}
        # total lines for progress bar
        try:
            total = sum(1 for _ in open(src_path, "r"))
        except Exception:
            total = None
        # tqdm wrapper (no-op if tqdm missing)
        def _tqdm(iterable, total=None, desc: str = ""):
            try:
                from tqdm import tqdm  # type: ignore
                return tqdm(iterable, total=total, unit="lines", desc=desc, leave=False)
            except Exception:
                return iterable

        with open(src_path, "r") as fin, open(dst_path, "w") as fout:
            for line in _tqdm(fin, total=total, desc=f"sanitize:{os.path.basename(dst_path)}"):
                original = line.rstrip("\n")
                before = original.replace(" ", "").strip()
                s = _sanitize_one(before, is_src=is_src, stats=stats)
                if s == "CC" and (before and before != "CC"):
                    stats["replaced"] += 1
                fout.write(s + "\n")
        stats["total"] = total or stats.get("total", 0)
        return stats

    stage_dir = os.path.join(args.out_dir, "_detok_stage")
    os.makedirs(stage_dir, exist_ok=True)

    def _maybe_detok(path: str, name: str, *, is_src: bool) -> str:
        if _needs_detok(path):
            dst = os.path.join(stage_dir, name)
            stats = _detok_file(path, dst, is_src=is_src)
            print(
                f"[bridge] sanitized {name}: total={stats.get('total', -1)} "
                f"arrow_split={stats.get('arrow_split', 0)} canonicalized={stats.get('canonicalized', 0)} "
                f"salvaged={stats.get('salvaged', 0)} replaced={stats.get('replaced', 0)}"
            )
            return dst
        return path

    # Build the argument list for Graph2SMILES preprocess
    # Optionally detokenize to temporary files
    train_src_p = _maybe_detok(train_src, "train.src", is_src=True)
    train_tgt_p = _maybe_detok(train_tgt, "train.tgt", is_src=False)
    val_src_p = _maybe_detok(val_src, "val.src", is_src=True)
    val_tgt_p = _maybe_detok(val_tgt, "val.tgt", is_src=False)
    test_src_p = _maybe_detok(test_src, "test.src", is_src=True)
    test_tgt_p = _maybe_detok(test_tgt, "test.tgt", is_src=False)

    g2s_cli = [
        "--model", args.model,
        "--task", "reaction_prediction",
        "--representation_start", "smiles",
        "--representation_end", args.repr,
        "--max_src_len", str(args.max_src_len),
        "--max_tgt_len", str(args.max_tgt_len),
        "--num_workers", str(args.workers),
        "--do_tokenize",
        "--train_src", train_src_p,
        "--train_tgt", train_tgt_p,
        "--val_src", val_src_p,
        "--val_tgt", val_tgt_p,
        "--test_src", test_src_p,
        "--test_tgt", test_tgt_p,
        "--preprocess_output_path", args.out_dir,
    ]

    parser = g2s_pre.get_preprocess_parser()
    g2s_args = parser.parse_args(g2s_cli)

    # Run preprocessing
    g2s_pre.preprocess_main(g2s_args)


if __name__ == "__main__":  # pragma: no cover
    main()


