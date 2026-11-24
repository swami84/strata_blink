#!/usr/bin/env python3
"""Exactness check: fixed prompts (no nonce), greedy, through the bench's own client; saves the generated text so two
engines/builds can be compared token for token (identical greedy tokens <=> identical text).

    exactness.py run  --out results/exact/<label>.json [--url http://127.0.0.1:8080] [--max-tokens 512]
    exactness.py diff results/exact/A.json results/exact/B.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/swamai/Work/experiments/basalt_vs_strata/bench")
from speed_bench import CODE, JSONGEN, MATH, PROSE, Client  # noqa: E402

PROMPTS = {
    "prose": PROSE,
    "code": CODE,
    "json": JSONGEN,
    "math": MATH,
    "edit": "Rewrite this function with type hints and a docstring, keeping the logic identical:\n\n"
            "def merge(a, b):\n    out = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n"
            "        if a[i] <= b[j]:\n            out.append(a[i]); i += 1\n        else:\n"
            "            out.append(b[j]); j += 1\n    return out + a[i:] + b[j:]\n",
}


def run(args):
    cli = Client(args.url, "default", 1800, "reasoning_content")
    out = {}
    for name, prompt in PROMPTS.items():
        t = time.perf_counter()
        r = cli.chat([{"role": "user", "content": prompt}], max_tokens=args.max_tokens)
        dt = r["t_last"] - r["t_first"] if r["t_first"] and r["t_last"] else 0
        n = r["usage"].get("completion_tokens", 0)
        out[name] = {"content": r["content"], "completion_tokens": n, "finish": r["finish"],
                     "decode_tps": (n - 1) / dt if dt > 0 else None}
        print(f"{name:6s} {n:4d} tokens  {out[name]['decode_tps'] or 0:6.1f} tok/s  ({time.perf_counter() - t:.1f} s)")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))


def diff(a_path, b_path):
    a, b = json.loads(Path(a_path).read_text()), json.loads(Path(b_path).read_text())
    bad = 0
    for k in a:
        x, y = a[k]["content"], b.get(k, {}).get("content")
        if x == y:
            print(f"{k:6s} identical ({a[k]['completion_tokens']} tokens)")
        else:
            bad += 1
            i = next((i for i, (c, d) in enumerate(zip(x, y or "")) if c != d), min(len(x), len(y or "")))
            print(f"{k:6s} DIFFERS at char {i}: {x[max(0, i - 40):i + 40]!r}\n{'':6s}            vs: {(y or '')[max(0, i - 40):i + 40]!r}")
    print("EXACT" if bad == 0 else f"{bad} of {len(a)} differ")
    return bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--url", default="http://127.0.0.1:8080")
    r.add_argument("--max-tokens", type=int, default=512)
    d = sub.add_parser("diff")
    d.add_argument("a")
    d.add_argument("b")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a)
    else:
        sys.exit(1 if diff(a.a, a.b) else 0)
