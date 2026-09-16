"""Chronological train/val/test splits, shared by every model.

Time-based (not random) splitting is required for forecasting: shuffling
would let the model "see the future" via leaked adjacent days.
"""
import numpy as np
import pandas as pd

from . import config


def chronological_split(df):
    train = df[(df["date"] >= config.TRAIN_START) & (df["date"] <= config.TRAIN_END)]
    val = df[(df["date"] >= config.VAL_START) & (df["date"] <= config.VAL_END)]
    test = df[(df["date"] >= config.TEST_START) & (df["date"] <= config.TEST_END)]
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def gpr_subsample(train_df, n=config.GPR_TRAIN_SUBSAMPLE, seed=config.RANDOM_SEED):
    """Random (seeded, reproducible) subsample of the training window,
    used only by the GPR-family models because exact GPR training is
    O(n^3) and PSO needs hundreds of refits.
    """
    if len(train_df) <= n:
        return train_df.reset_index(drop=True)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(train_df), size=n, replace=False)
    idx.sort()
    return train_df.iloc[idx].reset_index(drop=True)


def rolling_origin_splits(df, n_folds=3, val_days=30, test_days=30):
    """Generate n_folds chronological (train, val, test) splits by
    sliding the val/test window backward in fixed-size steps, keeping
    train as "everything before val" each time. Unlike the single fixed
    split in chronological_split(), this gives several genuine train/PSO
    refit cycles so the full method's improvement over baselines can be
    checked for stability across different train/test boundaries, not
    just one arbitrary cut -- used by walk_forward.py.
    """
    dates = df["date"]
    max_date = dates.max()
    fold_span = pd.Timedelta(days=val_days + test_days)
    splits_out = []
    for k in range(n_folds):
        test_end = max_date - k * fold_span
        test_start = test_end - pd.Timedelta(days=test_days)
        val_end = test_start
        val_start = val_end - pd.Timedelta(days=val_days)
        train_end = val_start

        train = df[df["date"] < train_end].reset_index(drop=True)
        val = df[(df["date"] >= val_start) & (df["date"] < val_end)].reset_index(drop=True)
        test = df[(df["date"] >= test_start) & (df["date"] < test_end)].reset_index(drop=True)
        if len(train) == 0 or len(val) == 0 or len(test) == 0:
            continue
        splits_out.append((train, val, test))
    return list(reversed(splits_out))  # chronological order, oldest fold first


def expanding_window_splits(df, n_folds=3, test_days=30):
    """Expanding-window folds: each fold's train set is everything before
    that fold's test window (not a fixed-size lookback), and test windows
    step backward from the end of the series. Used by the Design-2
    walk-forward check (walk_forward_expanding.py), which freezes the
    kernel hyperparameters already found by the main PSO run and only
    re-fits the GPR mean per fold -- unlike rolling_origin_splits above
    (no separate val window is needed here since no PSO search happens
    per fold). Training data grows with each fold rather than staying a
    fixed size, matching how the model would actually be retrained in
    deployment as data accumulates.
    """
    dates = df["date"]
    max_date = dates.max()
    test_span = pd.Timedelta(days=test_days)
    folds = []
    for k in range(n_folds):
        test_end = max_date - k * test_span
        test_start = test_end - test_span
        train_end = test_start
        train = df[df["date"] < train_end].reset_index(drop=True)
        test = df[(df["date"] >= test_start) & (df["date"] < test_end)].reset_index(drop=True)
        if len(train) == 0 or len(test) == 0:
            continue
        folds.append((train, test))
    return list(reversed(folds))
