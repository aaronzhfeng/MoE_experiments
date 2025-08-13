from __future__ import annotations

import argparse
from typing import Any, Dict


def load_yaml(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


def apply_overrides(args: argparse.Namespace, cfg: Dict[str, Any]) -> None:
    for k, v in cfg.items():
        if hasattr(args, k):
            setattr(args, k, v)


