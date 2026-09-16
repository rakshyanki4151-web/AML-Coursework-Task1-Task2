"""Re-run ONLY the two walk-forward validation designs.

The main pipeline (``python run.py``) writes outputs/metrics.json first and
then spends roughly three times as long again on walk-forward validation.
When only the walk-forward needs redoing -- because its seed budget changed,
or because it was interrupted -- re-running the whole pipeline just to reach
it wastes the main split's results, which are already on disk and unchanged.

This module reads the existing metrics.json for Design 2's frozen kernel
positions and regenerates both walk-forward JSONs in place.

Run with:  python -m src.run_walk_forward_only
"""
import json
import time

from . import config, data_prep, features, walk_forward, walk_forward_expanding


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    t0 = time.time()

    metrics_path = config.OUTPUTS_DIR / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"{metrics_path} not found. Design 2 reuses the kernel hyperparameters "
            f"PSO found on the main split, so run `python run.py` at least once first."
        )
    with open(metrics_path) as f:
        main_results = json.load(f)

    _log("Rebuilding the feature frame (data -> clean -> features)")
    raw_df, _ = data_prep.load_clean()
    feat_df = features.build_feature_frame(raw_df)
    _log(f"{len(feat_df)} rows, {len(features.FEATURE_COLUMNS)} features")
    _log(f"Walk-forward seed budget: {config.WALK_FORWARD_PSO_SEEDS} "
         f"({len(config.WALK_FORWARD_PSO_SEEDS)} seeds/fold vs {len(config.PSO_SEEDS)} on the main split)")

    # --- Design 1: rolling-origin, full PSO sweep per fold ---------------
    _log("Design 1 (rolling-origin, full PSO sweep per fold)...")
    design1 = {}
    for target in config.TARGETS:
        design1[target] = walk_forward.run_walk_forward(feat_df, target, n_folds=3)
    with open(config.OUTPUTS_DIR / "walk_forward_design1_results.json", "w") as f:
        json.dump(design1, f, indent=2)
    _log(f"Design 1 written ({(time.time()-t0)/60:.1f} min elapsed)")

    # --- Design 2: expanding-window, kernel frozen from the main split ---
    _log("Design 2 (expanding-window, frozen kernel)...")
    design2 = {}
    for target in config.TARGETS:
        kernel_positions = {
            k: tuple(main_results[target]["models"][f"gpr_{k}_pso"]["best_position"])
            for k in ("rbf", "matern", "periodic")
        }
        design2[target] = walk_forward_expanding.run_expanding_window(
            feat_df, target, kernel_positions, n_folds=3
        )
    with open(config.OUTPUTS_DIR / "walk_forward_design2_results.json", "w") as f:
        json.dump(design2, f, indent=2)

    _log(f"Both designs written. Total {(time.time()-t0)/60:.1f} min.")
    return design1, design2


if __name__ == "__main__":
    main()
