"""Fetch the paper's Zenodo data and keep only the two flight evaluation files (~2 MB)."""

from __future__ import annotations

import argparse
import hashlib
import io
import zipfile
from pathlib import Path

import httpx

URL = "https://zenodo.org/records/17677329/files/data.zip?download=1"
KEEP = ("data/eval/interaction/flight.jsonl", "data/eval/heldout/flight.json")
ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "paper")
    ap.add_argument("--zip", type=Path, help="use an already downloaded data.zip")
    args = ap.parse_args()

    if all((args.out / k).exists() for k in KEEP):
        print("already present:", *(args.out / k for k in KEEP), sep="\n  ")
        return
    if args.zip:
        blob = args.zip.read_bytes()
    else:
        print("downloading", URL)
        blob = httpx.get(URL, follow_redirects=True, timeout=300).raise_for_status().content
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        for name in KEEP:
            target = args.out / name
            target.parent.mkdir(parents=True, exist_ok=True)
            data = zf.read(name)
            target.write_bytes(data)
            print(f"{target}  {len(data):,} bytes  sha256 {hashlib.sha256(data).hexdigest()[:16]}")


if __name__ == "__main__":
    main()
