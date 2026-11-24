#!/usr/bin/env python3
"""Summarize run_strata_timing.sh output: per model, isolated vs stress.

Reads logs/engine-strata-exact-<model>-timing.log (STRATA_DECODE_TIMING / STRATA_VERIFY_PROFILE lines, split by the
"=== <model> / <cond>" markers the run script writes) and results/strata/<model>-<cond>.jsonl (client decode tok/s).
Warm-up requests (<= 50 windows) are excluded.

    summarize_timing.py [model ...]        default: every model with a timing log
"""
from __future__ import annotations

import collections
import json
import re
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TIMING = re.compile(
    r"strata decode timing: (\d+) windows, avg T ([\d.]+), ([\d.]+) tokens/window, ([\d.]+) ms/window = adapt wait ([\d.]+)"
    r" \+ verify ([\d.]+) \(GPU-reach wait ([\d.]+) \+ per-layer host ([\d.]+) \[plan [\d.]+ actq [\d.]+ jobs [\d.]+ CPU ([\d.]+)\]"
    r" \+ stage ([\d.]+)\) \+ commit/emit ([\d.]+) \+ draft ([\d.]+) \+ adapt join ([\d.]+) \+ other ([\d.]+); per layer-window:"
    r" CPU experts ([\d.]+)")
KEYS = ["win", "T", "tok_per_win", "ms_win", "adapt_wait", "verify", "gpu_reach_wait", "layer_host", "cpu", "stage", "commit",
        "draft", "adapt_join", "other", "cpu_experts"]
ROWS = [("ms_win", "ms / window"), ("tok_per_win", "tokens / window"), ("verify", "  verify"),
        ("layer_host", "    per-layer host"), ("stage", "    stage hand-off"), ("commit", "  commit/emit"),
        ("draft", "  MTP draft"), ("other", "  other"), ("cpu_experts", "CPU experts/layer-win")]


def parse(model: str):
    rows = collections.defaultdict(list)       # cond -> [timing dict]
    waits = collections.defaultdict(lambda: collections.defaultdict(list))   # cond -> stage/section/name -> [ms]
    modes = collections.Counter()
    cur, nwin = None, 0
    for line in open(HERE / f"logs/engine-strata-exact-{model}-timing.log", errors="replace"):
        if line.startswith("=== "):
            cur = line.split("/")[-1].strip()
            continue
        if "zero-doorbell graph" in line or "doorbell graph runs the windows" in line:
            modes[line.strip()[:160]] += 1
        m = TIMING.search(line)
        if m:
            nwin = int(m.group(1))
            if cur and nwin > 50:
                rows[cur].append(dict(zip(KEYS, map(float, m.groups()))))
        if "decode GPU stages" in line and cur and nwin > 50:
            stage = "s1" if "stage 1" in line else "s0"
            body = line.split("):", 1)[1].split("| total")[0]
            for part, sect in zip(body.split("|"), ("GDN", "QSA")):
                tot = collections.Counter()
                for name, v in re.findall(r"([A-Za-z0-9+/()\- ]+?) ([\d.]+)", part):
                    if re.search(r"wait|gap", name):
                        tot[name.strip()] += float(v)
                for k, v in tot.items():
                    waits[cur][f"{stage} {sect} {k}"].append(v)
    return rows, waits, modes


def tps(model: str, cond: str) -> list[float]:
    p = HERE / f"results/strata/{model}-{cond}.jsonl"
    return [json.loads(l)["decode_tps_cli"] for l in open(p)] if p.exists() else []


def mean(xs):
    xs = list(xs)
    return st.mean(xs) if xs else float("nan")


def main() -> None:
    models = sys.argv[1:] or sorted(p.name[len("engine-strata-exact-"):-len("-timing.log")]
                                    for p in (HERE / "logs").glob("engine-strata-exact-*-timing.log"))
    print("Strata 0.2.15, QFN, 2x RTX 3090 @300 W; decode suite (prose/code/json x reps, 1024 tokens); warm-ups excluded.")
    print("stress: 16 unpinned threads (probe/stress); *-pinned: engine on cores 0-3, 8 stress threads pinned to cores 4-7.\n")
    for model in models:
        rows, waits, modes = parse(model)
        conds = [c for c in ("isolated", "stress") if rows.get(c)]
        if not conds:
            continue
        print(f"== {model}")
        for line, n in modes.items():
            print(f"   graph: {line}  (x{n})")
        print(f"{'':24s}" + "".join(f"{c:>11s}" for c in conds) + ("      ratio" if len(conds) == 2 else ""))

        def out(label, vals, fmt="{:11.2f}"):
            s = f"{label:24s}" + "".join(fmt.format(v) for v in vals)
            if len(vals) == 2 and vals[0]:
                s += f"{vals[1] / vals[0]:10.2f}x"
            print(s)

        out("decode tok/s (client)", [mean(tps(model, c)) for c in conds], "{:11.1f}")
        for k, label in ROWS:
            out(label, [mean(r[k] for r in rows[c]) for c in conds])
        names = sorted(set().union(*(waits[c].keys() for c in conds)))
        tot = [sum(mean(waits[c][n]) for n in names if waits[c][n]) for c in conds]
        out("GPU waits on host flags", tot)
        for n in names:
            vals = [mean(waits[c][n]) if waits[c][n] else 0.0 for c in conds]
            if max(vals) >= 0.05:
                out(f"    {n}", vals)
        for c in conds:
            print(f"   per-request tok/s {c:9s}: {[round(x) for x in tps(model, c)]}")
        print()


if __name__ == "__main__":
    main()
