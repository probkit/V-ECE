#!/usr/bin/env bash
# Run the whole pipeline of the paper, from the TabRepo predictions in data/ to the raw
# results in results/, the figures in paper_figures/ and the tables in paper_tables/.
# Set PYTHON to the interpreter of the environment described in README.md (the ground
# truth needs `tabicl`; pass DEVICE=cuda or DEVICE=mps to speed it up).
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cpu}"

"$PYTHON" -m experiments.select_experiments

# Experiment 1: which recalibration function recovers the most calibration error?
"$PYTHON" -m experiments.benchmark_calibration_errors
"$PYTHON" -m plotting.plot_calibration_errors

# Experiment 2: estimators against a semi-synthetic ground truth.
"$PYTHON" -m experiments.make_ground_truth --device "$DEVICE"
"$PYTHON" -m experiments.benchmark_ce_estimators
"$PYTHON" -m plotting.plot_ce_estimators
"$PYTHON" -m plotting.plot_ce_estimator_ranking

"$PYTHON" -m plotting.make_tables

# Figures and tables used by the paper.
if [ -d paper ]; then
    mkdir -p paper/figs paper/tables
    cp paper_figures/*.pdf paper/figs/
    cp paper_tables/*.tex paper/tables/
fi
