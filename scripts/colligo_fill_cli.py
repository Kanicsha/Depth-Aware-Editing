#!/usr/bin/env python3
"""CLI smoke test for Colligo v2/images/fill clean-plate (Gate 1)."""
from __future__ import annotations

import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.colligo.fill_client import run_colligo_fill


def main() -> None:
    ap = argparse.ArgumentParser(description="Colligo clean-plate smoke test")
    ap.add_argument("source", help="RGB/BGR image path")
    ap.add_argument("mask", help="Grayscale removal mask path")
    ap.add_argument("out", help="Output image path")
    ap.add_argument("--preserve", default=None, help="Optional preserve-original path")
    ap.add_argument("--host", default=os.environ.get("COLLIGO_HOST", ""))
    ap.add_argument("--token", default=os.environ.get("COLLIGO_TOKEN", ""))
    args = ap.parse_args()

    src_bgr = cv2.imread(args.source)
    mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    if src_bgr is None or mask is None:
        raise SystemExit("Could not read source/mask")

    preserve_bgr = cv2.imread(args.preserve) if args.preserve else None
    out_bgr, method = run_colligo_fill(
        src_bgr,
        mask,
        host=args.host,
        token=args.token,
        preserve_bgr=preserve_bgr,
    )
    cv2.imwrite(args.out, out_bgr)
    print(f"Wrote {args.out} ({method})")


if __name__ == "__main__":
    main()
