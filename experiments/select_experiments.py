"""Select the TabRepo experiments used in the paper.

``data/tabrepo-{binary,multiclass}-experiments.csv`` list every (dataset, model) pair of
the TabRepo-binary and TabRepo-multiclass benchmarks of CalArena (Berta et al., 2026;
https://huggingface.co/datasets/probkit/CalArena), with the sizes of its calibration
split (out-of-fold predictions on the training data) and test split.  We keep the
datasets whose calibration split holds at least ``MIN_CAL_SIZE`` samples: the
semi-synthetic ground truth of ``make_ground_truth.py`` is fitted on that split, and a
large one pins down ``E[Y | f(X)]`` accurately.  This keeps
22 binary and 21 multiclass datasets, each with the same eight models.

Outputs ``data/experiments_{binary,multiclass}.csv``, read by every other script.

Usage
-----
    python -m experiments.select_experiments
"""

import pandas as pd

MIN_CAL_SIZE = 8000

for modality in ("binary", "multiclass"):
    experiments = pd.read_csv(f"data/tabrepo-{modality}-experiments.csv")
    experiments = experiments[experiments.cal_size >= MIN_CAL_SIZE].reset_index(drop=True)
    experiments.to_csv(f"data/experiments_{modality}.csv")
    print(
        f"{modality}: kept {experiments.dataset.nunique()} datasets, "
        f"{len(experiments)} experiments"
    )
