"""Edge-hardware feasibility check: measures ACTUAL peak memory and
per-sample inference latency of the saved PSO-tuned GPR models, so the
report's hardware-compatibility table cites real numbers instead of
plausible-sounding guesses.

Run standalone after the main pipeline has produced outputs/models/*.joblib:
    python -m src.edge_feasibility

Memory: measured via tracemalloc, which tracks actual Python heap
allocations during inference (not RSS, which includes the whole process
and everything already loaded e.g. numpy/sklearn itself). tracemalloc's
number is the fairer one for "how much does calling predict() cost you,"
which is what matters for e.g. deciding if a Raspberry Pi 4 (4GB) has
headroom to run this inside a larger control-loop process, not just in
isolation.

Latency: GPR predict() is timed for single-sample calls (the deployment
scenario: one new hour of data arrives, one forecast is produced) using
perf_counter, repeated and averaged to smooth out OS jitter, with
warm-up calls excluded (first call pays one-off JIT/cache costs that
don't recur in steady-state operation).
"""
import time
import tracemalloc

import joblib
import numpy as np

from . import config, features


N_TIMING_REPEATS = 200
N_WARMUP = 10


def _time_single_sample_predict(gpr, X_one_row):
    # Warm-up: exclude first-call overhead (BLAS thread pool spin-up etc.)
    for _ in range(N_WARMUP):
        gpr.predict(X_one_row)

    times_ms = []
    for _ in range(N_TIMING_REPEATS):
        t0 = time.perf_counter()
        gpr.predict(X_one_row)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    return {
        "mean_ms": float(np.mean(times_ms)),
        "p95_ms": float(np.percentile(times_ms, 95)),
        "max_ms": float(np.max(times_ms)),
    }


def _measure_target(short, target):
    model_path = config.MODELS_DIR / f"{short}_models.joblib"
    if not model_path.exists():
        print(f"Skipping {target}: {model_path} not found. Run the main pipeline first.")
        return None

    saved = joblib.load(model_path)
    scaler, pca = saved["scaler"], saved["pca"]
    gpr = saved["gpr_rbf_pso"]  # headline model
    transformer = saved["transformer_pso"]

    n_features = len(saved["feature_columns"])
    # One representative row of raw feature values (zeros are fine here --
    # timing/memory of predict() does not depend on the actual feature
    # values, only on array shape and the fitted kernel/training-set size).
    X_raw_one = np.zeros((1, n_features))

    def _full_inference_step():
        Xs = scaler.transform(X_raw_one)
        Xp = pca.transform(Xs)
        mean_t, std_t = gpr.predict(Xp, return_std=True)
        mean = transformer.inverse_transform(mean_t.reshape(-1, 1)).ravel()
        return mean, std_t

    # --- Memory: peak heap allocation during one full inference step ---
    tracemalloc.start()
    _full_inference_step()
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak_bytes / (1024.0 * 1024.0)

    # --- Latency: time the GPR predict() call specifically (the
    # dominant cost -- scaler/PCA are cheap linear ops by comparison) ---
    Xs_one = scaler.transform(X_raw_one)
    Xp_one = pca.transform(Xs_one)
    timing = _time_single_sample_predict(gpr, Xp_one)

    n_train_points = gpr.X_train_.shape[0]  # GPR cost scales with this

    return {
        "target": target,
        "n_gpr_training_points": int(n_train_points),
        "n_pca_components": int(pca.n_components_),
        "peak_memory_mb_single_predict": round(peak_mb, 2),
        "inference_latency_ms": timing,
    }


def main():
    print("--- Edge-Hardware Feasibility Measurement ---")
    print(f"(single-sample predict, {N_TIMING_REPEATS} repeats after {N_WARMUP} warm-up calls)\n")

    rows = []
    for target in config.TARGETS:
        short = "solar" if target == "solar_irradiance" else "wind"
        result = _measure_target(short, target)
        if result is None:
            continue
        rows.append(result)
        print(f"[{target}] GPR trained on {result['n_gpr_training_points']} points, "
              f"{result['n_pca_components']} PCA components")
        print(f"  Peak memory (single predict): {result['peak_memory_mb_single_predict']} MB")
        print(f"  Latency: mean={result['inference_latency_ms']['mean_ms']:.2f} ms, "
              f"p95={result['inference_latency_ms']['p95_ms']:.2f} ms, "
              f"max={result['inference_latency_ms']['max_ms']:.2f} ms\n")

    if not rows:
        print("No models found -- run the main pipeline first (python -m src.pipeline).")
        return

    import pandas as pd
    flat_rows = []
    for r in rows:
        flat_rows.append({
            "target": r["target"],
            "n_gpr_training_points": r["n_gpr_training_points"],
            "n_pca_components": r["n_pca_components"],
            "peak_memory_mb": r["peak_memory_mb_single_predict"],
            "latency_mean_ms": round(r["inference_latency_ms"]["mean_ms"], 3),
            "latency_p95_ms": round(r["inference_latency_ms"]["p95_ms"], 3),
            "latency_max_ms": round(r["inference_latency_ms"]["max_ms"], 3),
        })
    out_df = pd.DataFrame(flat_rows)
    out_path = config.OUTPUTS_DIR / "edge_feasibility.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved table to {out_path}")
    print("\nNOTE for the report: these numbers describe INFERENCE only "
          "(predict on an already-fitted model). They say nothing about "
          "whether TRAINING/PSO search could run on a Pi 4 -- that is far "
          "heavier and is not, and should not be, claimed to be edge-deployable. "
          "State this scope explicitly in the report.")


if __name__ == "__main__":
    main()
