#!/usr/bin/env bash
# Stock vs Blink (M3) on QFN IQ2_XS, same flags (--suffix-draft 0 --mmap-experts --ple-io ram), isolated vs with the CPU interferer
set -u
cd "$(dirname "$0")"
./run_strata_timing.sh iq2xs-ram iq2xs-blink3
./run_strata_timing.sh iq2xs-ram-pinned iq2xs-blink3-pinned
PINNED_STRESS_CPUS=1-7,9-15 PINNED_STRESS_THREADS=14 ./run_strata_timing.sh iq2xs-ram-1core-pinned iq2xs-blink3-1core-pinned
echo MATRIX_DONE
