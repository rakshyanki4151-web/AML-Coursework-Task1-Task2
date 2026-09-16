"""Gaussian Process Regression: default (sklearn-optimised) kernels vs a
PSO-tuned kernel, evaluated on the identical held-out test set.

Inputs are standardised first so that a single isotropic length-scale is
meaningful across features measured in different units (deg C, %, m/s,
kWh/m^2/day, degrees, ...).
"""
import warnings

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel, WhiteKernel, ExpSineSquared
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.metrics import mean_squared_error

from . import pso as pso_module
from . import config

# Search space for PSO: [length_scale, signal_variance, noise_alpha].
# Bounds are shared for consistency.
PSO_BOUNDS = [(1e-2, 100.0), (1e-2, 100.0), (1e-4, 1.0)]
FAILED_FIT_PENALTY = 1e6


def _make_kernel(base_kernel_cls, length_scale=1.0, signal_variance=1.0, noise=1e-2, **base_kwargs):
    if base_kernel_cls == ExpSineSquared:
        # Hybrid Periodic-RBF kernel: captures 24-hour repeating cycles modulated by RBF decay
        periodic_part = ExpSineSquared(length_scale=length_scale, periodicity=24.0, periodicity_bounds="fixed")
        # Use a fixed broad RBF envelope to allow day-to-day weather variance
        rbf_envelope = RBF(length_scale=48.0, length_scale_bounds="fixed")
        base_kernel = periodic_part * rbf_envelope
    else:
        base_kernel = base_kernel_cls(length_scale=length_scale, length_scale_bounds="fixed", **base_kwargs)
        
    return (
        ConstantKernel(constant_value=signal_variance, constant_value_bounds="fixed")
        * base_kernel
        + WhiteKernel(noise_level=noise, noise_level_bounds="fixed")
    )


def compute_solar_baseline(df):
    """Clear-sky solar baseline based on Lambert's Cosine Law and atmospheric attenuation.
    Ic = I0 * cos(zenith)
    """
    cos_zenith = np.cos(np.deg2rad(df["solar_zenith"]))
    return 0.85 * np.maximum(0.0, cos_zenith)


def compute_wind_baseline(df):
    """Physical wind baseline: current-hour momentum persistence, modulated
    by the local diurnal thermal gradient.

        wind_baseline = wind_speed + 0.05 * (temp - temp_lag24)

    The persistence term is the wind speed measured *now* (time t), which is
    the most recent observation available when forecasting t+24. An earlier
    revision used ``wind_speed_lag24`` (the value at t-24) here, which threw
    away a full day of information and made the physical baseline strictly
    worse than the naive persistence model the GPR is benchmarked against --
    the residual model was being handed a 48-hour-old anchor to correct.
    Using the current value makes the baseline exactly the persistence
    forecast, so the GPR's job is the well-posed one of learning the
    *correction* to persistence rather than re-deriving persistence itself.

    The thermal term is unchanged: the 24-hour temperature change is a
    cheap proxy for the local pressure-gradient forcing that drives
    day-to-day wind variability in mountainous terrain.
    """
    t_diff = df["temperature"] - df["temperature_lag24"]
    return df["wind_speed"] + 0.05 * t_diff


def fit_default_gpr(X_train, y_train_residual, kernel_kind="rbf"):
    """sklearn's own internal optimiser (L-BFGS-B with restarts) tunes the
    kernel hyperparameters -- this is the "default/manually-tuned" baseline
    the proposal benchmarks PSO against. Fits on residuals.
    """
    if kernel_kind == "periodic":
        # ExpSineSquared has no direct sklearn-optimisable equivalent of
        # "length_scale_bounds" behaving the same way once wrapped in the
        # periodic*RBF product used by _make_kernel/fit_pso_gpr, so the
        # "default" variant here optimises only the periodic length_scale
        # via sklearn's optimiser, keeping periodicity fixed at 24h and the
        # RBF envelope fixed at 48h -- the same structure PSO tunes below,
        # so the default-vs-PSO comparison stays apples-to-apples.
        base_cls, base_kwargs = ExpSineSquared, {}
        periodic_part = ExpSineSquared(length_scale=1.0, length_scale_bounds=(1e-2, 1e2),
                                        periodicity=24.0, periodicity_bounds="fixed")
        rbf_envelope = RBF(length_scale=48.0, length_scale_bounds="fixed")
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3)) * (periodic_part * rbf_envelope)
            + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1.0))
        )
    else:
        base_cls = RBF if kernel_kind == "rbf" else Matern
        base_kwargs = {} if kernel_kind == "rbf" else {"nu": 1.5}
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * base_cls(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), **base_kwargs)
            + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1.0))
        )
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=5,
                                    random_state=config.RANDOM_SEED)
    
    # Fit target transformer on residuals
    transformer = PowerTransformer(method="yeo-johnson")
    y_train_t = transformer.fit_transform(y_train_residual.reshape(-1, 1)).ravel()
    
    gpr.fit(X_train, y_train_t)
    return gpr, transformer


def fit_pso_gpr(X_train, y_train_residual, X_val, y_val_residual, y_val_actual, y_val_baseline, kernel_kind="rbf",
                 n_particles=config.PSO_N_PARTICLES, n_iterations=config.PSO_N_ITERATIONS, seed=config.RANDOM_SEED):
    """PSO searches (length_scale, signal_variance, noise) to minimise
    validation RMSE of the final predictions (baseline + GPR forecast),
    with sklearn's internal optimiser switched OFF (optimizer=None).

    ``seed`` controls only the PSO swarm's random initialisation/velocity
    draws (passed through to pso_module.optimise); the GPR's own
    random_state stays fixed at config.RANDOM_SEED throughout so that,
    for a given kernel position, the fit itself is reproducible -- only
    the search trajectory through hyperparameter space varies across
    seeds. This split is what makes the multi-seed run below a fair test
    of PSO's stability rather than of fit-to-fit noise.
    """
    base_cls = RBF if kernel_kind == "rbf" else (Matern if kernel_kind == "matern" else ExpSineSquared)
    base_kwargs = {} if kernel_kind != "matern" else {"nu": 1.5}

    # Fit target transformer on residuals
    transformer = PowerTransformer(method="yeo-johnson")
    y_train_t = transformer.fit_transform(y_train_residual.reshape(-1, 1)).ravel()

    def fitness(position):
        length_scale, signal_variance, noise = position
        kernel = _make_kernel(base_cls, length_scale, signal_variance, noise, **base_kwargs)
        gpr = GaussianProcessRegressor(kernel=kernel, optimizer=None, normalize_y=True,
                                        random_state=config.RANDOM_SEED)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                gpr.fit(X_train, y_train_t)
                preds_t = gpr.predict(X_val)
                # Inverse transform GPR residual predictions
                preds_residual = transformer.inverse_transform(preds_t.reshape(-1, 1)).ravel()
                # Final prediction = baseline + predicted residual
                preds = y_val_baseline + preds_residual
            except Exception:
                return FAILED_FIT_PENALTY
        if not np.all(np.isfinite(preds)):
            return FAILED_FIT_PENALTY
        return float(np.sqrt(mean_squared_error(y_val_actual, preds)))

    result = pso_module.optimise(
        fitness, PSO_BOUNDS, n_particles=n_particles, n_iterations=n_iterations,
        inertia=config.PSO_INERTIA, cognitive=config.PSO_COGNITIVE, social=config.PSO_SOCIAL,
        seed=seed,
    )

    length_scale, signal_variance, noise = result.best_position
    best_kernel = _make_kernel(base_cls, length_scale, signal_variance, noise, **base_kwargs)
    final_gpr = GaussianProcessRegressor(kernel=best_kernel, optimizer=None, normalize_y=True,
                                          random_state=config.RANDOM_SEED)
    # Refit residual on train+val
    X_full = np.vstack([X_train, X_val])
    y_full_residual = np.concatenate([y_train_residual, y_val_residual])
    y_full_t = transformer.fit_transform(y_full_residual.reshape(-1, 1)).ravel()
    final_gpr.fit(X_full, y_full_t)
    return final_gpr, result, transformer


def fit_pso_gpr_multiseed(X_train, y_train_residual, X_val, y_val_residual, y_val_actual, y_val_baseline,
                           kernel_kind="rbf", n_particles=config.PSO_N_PARTICLES,
                           n_iterations=config.PSO_N_ITERATIONS, seeds=config.PSO_SEEDS):
    """Run PSO independently once per seed in ``seeds`` and keep the
    single best-scoring run's fitted model (used for downstream test-set
    predictions), while also reporting the mean/std of the best
    validation RMSE across all runs. This is the "full marks" version of
    the single-seed fit_pso_gpr: it answers whether PSO's improvement is
    a stable property of the search or an artefact of one lucky swarm
    initialisation.

    Returns
    -------
    best_gpr, best_result, best_transformer : the winning run's outputs,
        same shape as fit_pso_gpr's return value.
    summary : dict with per-seed best RMSE, mean, std, and which seed won.
    """
    per_seed_results = []
    best = None  # (val_rmse, gpr, result, transformer, seed)

    for seed in seeds:
        gpr, result, transformer = fit_pso_gpr(
            X_train, y_train_residual, X_val, y_val_residual, y_val_actual, y_val_baseline,
            kernel_kind=kernel_kind, n_particles=n_particles, n_iterations=n_iterations, seed=seed,
        )
        val_rmse = float(result.best_score)
        per_seed_results.append({"seed": seed, "best_val_rmse": val_rmse, "best_position": [float(p) for p in result.best_position]})
        if best is None or val_rmse < best[0]:
            best = (val_rmse, gpr, result, transformer, seed)

    val_rmses = np.array([r["best_val_rmse"] for r in per_seed_results])
    summary = {
        "kernel_kind": kernel_kind,
        "seeds_used": list(seeds),
        "per_seed": per_seed_results,
        "mean_val_rmse": float(np.mean(val_rmses)),
        "std_val_rmse": float(np.std(val_rmses)),
        "best_seed": best[4],
        "best_val_rmse": best[0],
    }
    best_gpr, best_result, best_transformer = best[1], best[2], best[3]
    return best_gpr, best_result, best_transformer, summary


class IdentityReducer:
    """Pass-through stand-in for sklearn's PCA, used when config.USE_PCA is
    False.

    Exposing the same small slice of PCA's API that this project actually
    calls (``fit_transform``, ``transform``, ``n_components_``) means
    disabling PCA is a one-line config change: the saved model bundle still
    carries a "pca" object, so the Streamlit app, feature_importance.py and
    edge_feasibility.py keep working without modification, and switching
    PCA back on is symmetric. Defined at module level (not as a closure or
    lambda) so joblib can pickle it into the saved bundles.
    """

    def fit(self, X, y=None):
        self.n_components_ = X.shape[1]
        return self

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def transform(self, X):
        return X


def build_scaler(X_train):
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def predict_with_uncertainty(gpr, X):
    mean, std = gpr.predict(X, return_std=True)
    return mean, std
