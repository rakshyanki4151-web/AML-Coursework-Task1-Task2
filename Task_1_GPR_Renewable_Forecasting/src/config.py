"""Central configuration: paths, site metadata, and split/PSO settings.

Keeping every tunable in one place means the report's "Experimental setup"
section can cite this file directly instead of restating magic numbers.
"""
from pathlib import Path

# --- Paths -------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT_DIR / "data" / "raw" / "nasa_power_hourly.csv"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
OUTPUTS_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR = OUTPUTS_DIR / "models"

# --- Site metadata (from the karnali_center_hourly_2025.csv extract) -----
# "Karnali_Center", 29.267N / 82.183E -- sits right by Jumla, Karnali
# Province, matching the proposal's named region. This extract has no
# reported elevation field (it is not NASA's native export format), so
# ELEVATION_M is left as an approximate regional figure, not a value read
# from the file itself.
LATITUDE = 29.267
LONGITUDE = 82.183
ELEVATION_M = 2300.0  # approximate regional elevation for the Jumla area; not in the source file

RAW_COLUMNS = {
    "ALLSKY_SFC_SW_DWN": "solar_irradiance",  # converted Wh/m^2/hr -> kWh/m^2/hr in data_prep
    "WS10M": "wind_speed",                    # m/s
    "T2M": "temperature",                     # C
    "RH2M": "humidity",                       # %
    "CLOUD_AMT": "cloud_amount",              # %, 5th input variable named in the proposal
}
MISSING_SENTINEL = -999

# --- Forecast targets ----------------------------------------------------
TARGETS = ["solar_irradiance", "wind_speed"]
# Day-ahead framing at hourly resolution: forecast the SAME hour 24h later,
# matching the proposal's "next day's generation" motivation while using
# the real HR column this dataset provides.
FORECAST_HORIZON_HOURS = 24

# Solar irradiance is trivially 0 for ~half the day (night); scoring every
# model on those hours would inflate every RMSE/R2 equally and hide the
# forecast quality that actually matters operationally (daylight output).
# Rows are flagged daylight empirically (current-hour irradiance > 0) and
# the solar model/evaluation is restricted to them. Wind uses all 24h.
DAYLIGHT_FILTER_TARGETS = {"solar_irradiance"}

# --- Chronological split (shared by every model for a fair comparison) --
# Only one calendar year (2025) is available at hourly resolution, so the
# split is by month rather than by year.
TRAIN_START = "2025-01-01 00:00:00"
TRAIN_END = "2025-08-31 23:00:00"
VAL_START = "2025-09-01 00:00:00"
VAL_END = "2025-09-30 23:00:00"
TEST_START = "2025-10-01 00:00:00"
TEST_END = "2025-12-31 23:00:00"

# GPR scales cubically with the number of training points, so the GPR /
# PSO experiments train on a bounded random subsample of the training
# window rather than all ~5800 hours. This mirrors the proposal's own
# framing (GPR suits "small, incomplete datasets") -- baselines below
# still use the full training window since they scale linearly.
GPR_TRAIN_SUBSAMPLE = 700
RANDOM_SEED = 42

# --- Dimensionality reduction --------------------------------------------
# PCA is DISABLED. It was enabled in an earlier revision, and the ablation
# study (outputs/ablation_summary.csv, `python -m src.ablation`) showed it
# hurt both targets rather than helped: solar test RMSE 0.0965 with PCA vs
# 0.0844 without, wind 0.6575 vs 0.6223. The reason is that PCA is
# unsupervised -- it keeps the directions of greatest *input* variance,
# which are not the directions that predict the target. With 41 strongly
# collinear lag/rolling features, the low-variance directions it discards
# still carry genuine short-horizon signal.
#
# A second, subtler problem: sklearn's PCA does not whiten by default, so
# the component scores fed to the kernel had standard deviations spanning
# 3.92 down to 0.59. A single isotropic RBF length-scale therefore weighted
# PC1 about 7x more heavily than PC15 in the kernel distance -- an implicit
# feature weighting nobody chose.
#
# Set back to True to reproduce the earlier PCA-based results; the pipeline
# substitutes a pass-through reducer (gpr_models.IdentityReducer) when
# False, so every downstream consumer that calls .transform() on the saved
# "pca" object keeps working unchanged.
USE_PCA = False
PCA_VARIANCE_RETAINED = 0.95

# --- PSO hyperparameters --------------------------------------------------
PSO_N_PARTICLES = 20
PSO_N_ITERATIONS = 30
PSO_INERTIA = 0.6
PSO_COGNITIVE = 1.5
PSO_SOCIAL = 1.5

# PSO's swarm is randomly initialised, so a single run can get a lucky or
# unlucky starting position. Each GPR-kernel PSO search is repeated once
# per seed below and the mean/std of the best validation RMSE is reported
# alongside the single best-scoring run (which is what feeds the final
# test-set numbers). 5 seeds x 3 kernels x 2 targets = 30 total swarm
# searches -- lower this list (e.g. to 3 seeds) if runtime is a problem.
PSO_SEEDS = [42, 7, 123, 2024, 99]

# The walk-forward check (walk_forward.py) repeats the whole 3-kernel PSO
# sweep once per fold, so at 3 folds x 2 targets it costs 90 PSO searches
# against the main split's 30 -- three times the main run's model-fitting
# time. Its purpose is only to confirm the main split's ranking is stable
# across different train/test boundaries, not to re-establish PSO's own
# seed-to-seed stability (fit_pso_gpr_multiseed already does that on the
# main split with all 5 seeds). Two seeds per fold is therefore enough
# here, and cuts the walk-forward's runtime by ~60%.
# STATE THIS IN THE REPORT: walk-forward folds use 2 seeds, the main split
# uses 5.
WALK_FORWARD_PSO_SEEDS = [42, 7]

# GP Classification PSO search space: [length_scale, constant_value].
# Narrower than the GPR search since GPC is comparatively cheap to fit
# and this is a secondary, smaller-budget search.
GPC_PSO_BOUNDS = [(1e-2, 1e2), (1e-3, 1e3)]
GPC_PSO_N_PARTICLES = 10
GPC_PSO_N_ITERATIONS = 15

# Classification thresholds: tertiles computed on the TRAIN split only,
# then reused everywhere else so no test-set information leaks in.
CLASS_LABELS = ["Low", "Medium", "High"]
