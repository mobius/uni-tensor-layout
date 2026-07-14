#!/usr/bin/env python3
"""Print custom Host/Phi/VE atoms."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uni_cute_tensor.atoms import list_atoms


def main() -> None:
    for name, atom in list_atoms().items():
        print(f"{name}: shape_mnk={atom.shape_mnk} ptx={atom.ptx}")
        print(f"  c_layout={atom.c_layout}")


if __name__ == "__main__":
    main()
