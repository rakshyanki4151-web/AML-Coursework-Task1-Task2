"""Threshold the regression target into Low/Medium/High generation bands
and evaluate Gaussian Process Classification -- objective 3 of the
proposal, mirroring how a microgrid operator plans battery/diesel use.

Thresholds (tertiles) are computed on the TRAIN split's target_next
values only, then applied unchanged to val/test so no test information
leaks into the class boundaries.
"""
import warnings

import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.metrics import f1_score

from . import config
from . import pso as pso_module


def compute_thresholds(train_target_next):
    q1, q2 = np.quantile(train_target_next, [1 / 3, 2 / 3])
    return q1, q2


def to_classes(values, q1, q2):
    return np.select(
        [values <= q1, values <= q2],
        [0, 1],
        default=2,
    )  # 0=Low, 1=Medium, 2=High


def fit_gpc(X_train, y_class_train):
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    gpc = GaussianProcessClassifier(kernel=kernel, random_state=config.RANDOM_SEED, n_restarts_optimizer=3)
    gpc.fit(X_train, y_class_train)
    return gpc


def fit_gpc_pso(X_train, y_class_train, X_val, y_class_val,
                 n_particles=config.GPC_PSO_N_PARTICLES, n_iterations=config.GPC_PSO_N_ITERATIONS,
                 seed=config.RANDOM_SEED):
    """PSO-tuned counterpart to fit_gpc(): searches (length_scale,
    constant_value) of the RBF kernel to maximise validation macro-F1,
    instead of relying on sklearn's own restart optimiser. Added so that
    classification -- objective 3 -- is tuned by the same PSO method used
    for the regression stage, rather than being the one stage PSO never
    touches. fit_gpc() (untuned) is kept as-is above for comparison; the
    report should show both.
    """
    def fitness(position):
        length_scale, constant_value = position
        kernel = ConstantKernel(constant_value, "fixed") * RBF(length_scale=length_scale, length_scale_bounds="fixed")
        gpc = GaussianProcessClassifier(kernel=kernel, random_state=config.RANDOM_SEED, optimizer=None)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                gpc.fit(X_train, y_class_train)
                pred_val = gpc.predict(X_val)
            except Exception:
                return 1.0  # worst possible score (1 - f1, f1=0)
        # PSO minimises, so score = 1 - macro F1 (maximise F1)
        return 1.0 - f1_score(y_class_val, pred_val, average="macro")

    result = pso_module.optimise(
        fitness, config.GPC_PSO_BOUNDS, n_particles=n_particles, n_iterations=n_iterations,
        inertia=config.PSO_INERTIA, cognitive=config.PSO_COGNITIVE, social=config.PSO_SOCIAL,
        seed=seed,
    )
    length_scale, constant_value = result.best_position
    best_kernel = ConstantKernel(constant_value, "fixed") * RBF(length_scale=length_scale, length_scale_bounds="fixed")
    final_gpc = GaussianProcessClassifier(kernel=best_kernel, random_state=config.RANDOM_SEED, optimizer=None)
    # Refit on train+val, same convention as fit_pso_gpr's final refit
    X_full = np.vstack([X_train, X_val])
    y_full = np.concatenate([y_class_train, y_class_val])
    final_gpc.fit(X_full, y_full)
    return final_gpc, result
