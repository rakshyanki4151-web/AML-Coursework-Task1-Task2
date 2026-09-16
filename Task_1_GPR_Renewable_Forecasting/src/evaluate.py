"""Metrics for the regression benchmark and the classification stage."""
import numpy as np
from scipy import stats
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, confusion_matrix,
)


def regression_metrics(y_true, y_pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def nlpd(y_true, mean_pred, std_pred):
    """Negative log predictive density -- rewards a GPR model whose
    uncertainty band is well-calibrated, not just whose mean is accurate.
    """
    std_pred = np.clip(std_pred, 1e-6, None)
    return float(np.mean(
        0.5 * np.log(2 * np.pi * std_pred ** 2) + 0.5 * ((y_true - mean_pred) ** 2) / (std_pred ** 2)
    ))


def paired_error_test(y_true, pred_a, pred_b):
    """Paired significance test on per-hour squared error, answering
    "is model A's RMSE improvement over model B real, or within noise?"
    Both models are scored on the exact same test hours (same y_true), so
    a paired test is appropriate here (stronger than comparing RMSE point
    estimates alone). Uses a paired t-test on squared errors; falls back
    to the non-parametric Wilcoxon signed-rank test as a robustness check
    since squared-error differences are often skewed/non-normal.
    """
    y_true = np.asarray(y_true)
    err_a = (y_true - np.asarray(pred_a)) ** 2
    err_b = (y_true - np.asarray(pred_b)) ** 2
    diff = err_a - err_b  # negative => model A had lower squared error

    t_stat, t_pvalue = stats.ttest_rel(err_a, err_b)
    try:
        w_stat, w_pvalue = stats.wilcoxon(err_a, err_b)
    except ValueError:
        # Wilcoxon fails if all differences are exactly zero
        w_stat, w_pvalue = float("nan"), float("nan")

    return {
        "mean_sq_error_diff_A_minus_B": float(np.mean(diff)),
        "paired_ttest_statistic": float(t_stat),
        "paired_ttest_pvalue": float(t_pvalue),
        "wilcoxon_statistic": float(w_stat),
        "wilcoxon_pvalue": float(w_pvalue),
        "A_significantly_better_at_0.05": bool(t_pvalue < 0.05 and np.mean(diff) < 0),
    }


def diebold_mariano_test(y_true, pred_a, pred_b, h=1, power=2):
    """Diebold-Mariano test (Diebold & Mariano, 1995), the standard test
    for comparing forecast accuracy between two models on time-series
    data -- used here as the primary companion to paired_error_test()
    above, because paired_error_test's t-test/Wilcoxon both assume the
    per-hour errors are independent, which does not strictly hold for
    hourly forecasts (an error at 2pm is correlated with the error at
    3pm). DM does not make that assumption: it builds a Newey-West
    (HAC) standard error for the mean loss-differential that explicitly
    accounts for up to ``h-1`` lags of autocorrelation.

    Parameters
    ----------
    h : forecast horizon in steps. Loss differentials are assumed to be
        autocorrelated up to lag h-1 under the standard DM assumptions;
        this project forecasts 24h ahead on hourly data, but the day-to-day
        (not hour-to-hour) autocorrelation structure of the *error series*
        is what matters here, so h=1 is used as a conservative default
        (equivalent to only correcting for whatever residual
        autocorrelation actually shows up in the loss differential itself,
        via the Newey-West truncation lag below) unless you have a
        specific reason to test a stronger seasonal correlation assumption.
    power : 1 = absolute-error loss, 2 = squared-error loss (default,
        matches RMSE-based comparisons used elsewhere in this project).

    Returns a dict with the DM statistic, its p-value (two-sided, using a
    Student-t reference distribution as recommended by Harvey, Leybourne
    & Newbold (1997) for finite-sample correction), and which model the
    test favours.
    """
    y_true = np.asarray(y_true, dtype=float)
    pred_a = np.asarray(pred_a, dtype=float)
    pred_b = np.asarray(pred_b, dtype=float)
    n = len(y_true)

    e_a = y_true - pred_a
    e_b = y_true - pred_b
    loss_a = np.abs(e_a) ** power
    loss_b = np.abs(e_b) ** power
    d = loss_a - loss_b  # loss differential; d < 0 on average => A better

    d_bar = np.mean(d)

    # Newey-West-style long-run variance of d, truncated at lag h-1 (the
    # standard DM autocovariance truncation), which is what lets this test
    # remain valid even though consecutive hourly errors are correlated.
    max_lag = max(h - 1, 0)
    gamma0 = np.var(d, ddof=0)
    var_d = gamma0
    for lag in range(1, max_lag + 1):
        cov = np.cov(d[lag:], d[:-lag])[0, 1] if n > lag else 0.0
        var_d += 2 * cov
    var_d = max(var_d, 1e-12)  # guard against numerical negative/zero variance

    dm_stat = d_bar / np.sqrt(var_d / n)

    # Harvey, Leybourne & Newbold (1997) small-sample correction factor,
    # then reference against Student-t(n-1) rather than Normal(0,1) --
    # more conservative (appropriately so) for finite test-set sizes.
    hln_correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_stat_corrected = dm_stat * hln_correction
    p_value = 2 * (1 - stats.t.cdf(np.abs(dm_stat_corrected), df=n - 1))

    return {
        "dm_statistic": float(dm_stat_corrected),
        "dm_pvalue": float(p_value),
        "mean_loss_diff_A_minus_B": float(d_bar),
        "n_observations": int(n),
        "A_significantly_better_at_0.05": bool(p_value < 0.05 and d_bar < 0),
        "B_significantly_better_at_0.05": bool(p_value < 0.05 and d_bar > 0),
    }


def coverage_at_z(y_true, mean_pred, std_pred, z=1.96):
    """Empirical coverage: fraction of true values falling inside
    mean_pred +/- z*std_pred. For a well-calibrated model at z=1.96
    this should be close to 0.95 -- a direct, cheap check of whether the
    GPR's uncertainty band means what it claims to mean, complementing
    NLPD (which scores calibration but isn't as directly interpretable
    as "should be about 95%").
    """
    y_true, mean_pred, std_pred = map(np.asarray, (y_true, mean_pred, std_pred))
    lower = mean_pred - z * std_pred
    upper = mean_pred + z * std_pred
    inside = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(inside))


def classification_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }
