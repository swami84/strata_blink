#!/usr/bin/env bash
# QFN IQ2_XS (ISTA-DASLab GSQ-RCO) at the same pinned revision as basalt_vs_strata's IQ3_S, plus its Strata pack.
# Shard 2 (the PLE n-gram table) is byte-identical to IQ3_S's (same SHA-256) -> symlinked, not downloaded.
set -euo pipefail
BVS=/home/swamai/Work/experiments/basalt_vs_strata
V=/home/swamai/Work/experiments/blink/vendor/strata-0.2.15
REV=ed59f92082b1e93c0e96d60a8b11aab089b52f09
DST=$BVS/models/chat-IQ2_XS
S1=$DST/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf
S1_SHA=92cee27ae5bbadcd732416a0f7a7f0acc092399dbbe8f5a5efa707c2ec0a49d7

"$BVS/.venv/bin/hf" download ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF --revision "$REV" \
    --include "IQ2_XS/*00001-of-00002.gguf" --local-dir "$DST"
ln -sf ../../chat-IQ3_S/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf \
    "$DST/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf"
echo "verifying sha256 of shard 1 ..."
echo "$S1_SHA  $S1" | sha256sum -c -
echo DOWNLOAD_DONE

# the Strata 0.2.15 pack (dense weights + tokenizer; experts are served natively from the GGUF), as setup built iq3_s
"$BVS/engines/Strata/terminal/.venv/bin/python" "$V/tools/iq_pack.py" --gguf "$S1" --out "$BVS/data/strata-data/packs/iq2_xs"
echo PACK_DONE
