#!/usr/bin/env python3
"""Robustness checks for the Blink decode loop against the stock loop (run once per engine, then diff):

  multiturn  three chat turns, each continuing the last; greedy.  The engine's conversation cache reuses the state of the
             previous turn, which needs the host-side bookkeeping the GPU loop hands back (the tokens fed, the position,
             the PLE history) to be exactly the host loop's.  Saves every turn's text and the server's cached-token count.
  cancel     a long generation cut off by the client after ~1 s (the server sends STOP), then a normal request: the engine
             must still answer, exactly.

    robustness.py run  --out results/robust/<label>.json
    robustness.py diff results/robust/A.json results/robust/B.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, "/home/swamai/Work/experiments/basalt_vs_strata/bench")
from speed_bench import Client  # noqa: E402

TURNS = [
    "Explain in about 150 words why the sky is blue.",
    "Now explain the same thing to a five-year-old, in about 80 words.",
    "Write a haiku that captures both explanations.",
]
AFTER_CANCEL = "List the first ten prime numbers, comma separated, nothing else."


def run(args):
    cli = Client(args.url, "default", 600, "reasoning_content")
    out = {"turns": [], "cached": [], "after_cancel": None}
    msgs = []
    for q in TURNS:
        msgs.append({"role": "user", "content": q})
        r = cli.chat(msgs, max_tokens=200)
        out["turns"].append(r["content"])
        out["cached"].append(((r["usage"] or {}).get("prompt_tokens_details") or {}).get("cached_tokens"))
        msgs.append({"role": "assistant", "content": r["content"]})
        print(f"turn {len(out['turns'])}: {r['usage'].get('completion_tokens')} tokens, cached {out['cached'][-1]}")
    # cancel: stream a long answer, drop the connection after ~1 s
    body = cli.body([{"role": "user", "content": "Write a 3000-word story about a lighthouse keeper."}], 3000, False)
    t0 = time.time()
    got = 0
    with requests.post(f"{args.url}/v1/chat/completions", json=body, stream=True, timeout=600) as r:
        for raw in r.iter_lines():
            if raw and raw.startswith(b"data:"):
                got += 1
            if time.time() - t0 > 1.0:
                break
    print(f"cancel: dropped the stream after {time.time() - t0:.1f} s and {got} chunks")
    time.sleep(2.0)
    r = cli.chat([{"role": "user", "content": AFTER_CANCEL}], max_tokens=64)
    out["after_cancel"] = r["content"]
    print(f"after cancel: {r['content']!r}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))


def diff(a_path, b_path):
    a, b = json.loads(Path(a_path).read_text()), json.loads(Path(b_path).read_text())
    bad = 0
    for i, (x, y) in enumerate(zip(a["turns"], b["turns"])):
        same = x == y
        bad += not same
        print(f"turn {i + 1}: {'identical' if same else 'DIFFERS'}  (cached tokens {a['cached'][i]} vs {b['cached'][i]})")
    same = a["after_cancel"] == b["after_cancel"]
    bad += not same
    print(f"after cancel: {'identical' if same else 'DIFFERS'}")
    print("EXACT" if bad == 0 else f"{bad} differ")
    return bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--url", default="http://127.0.0.1:8080")
    d = sub.add_parser("diff")
    d.add_argument("a")
    d.add_argument("b")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a)
    else:
        sys.exit(1 if diff(a.a, a.b) else 0)
