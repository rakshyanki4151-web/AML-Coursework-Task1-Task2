"""Three baselines, from weakest to strongest, that PSO-tuned GPR must beat:

1. Persistence  -- "same hour next day" = "same hour today" (the naive
   forecast an operator makes with no model at all).
2. Linear Regression -- same feature set as the GPR models, full training
   window (linear models scale fine, so no subsampling needed).
3. SARIMAX -- classical time-series baseline with an explicit 24-hour
   seasonal term (plain ARIMA has no way to represent the diurnal solar/
   wind cycle), updated one step at a time so each forecast only ever
   sees real past observations.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from statsmodels.tsa.statespace.sarimax import SARIMAX

from . import config

# Non-seasonal: a full 24h-seasonal term was tried first but each
# append+forecast(24) walk-forward step became too expensive over
# ~5,800 evaluation hours (>10 minutes and still running). The feature-
# rich GPR/LR models already see the diurnal cycle via hour_sin/hour_cos
# and the lag24 features, so SARIMAX here is deliberately the "naive
# time-series-only" baseline, not a fully-tuned competitor.
SARIMAX_ORDER = (2, 0, 2)
SARIMAX_SEASONAL_ORDER = (0, 0, 0, 0)


def persistence_predict(df_split, target):
    """Forecast horizon-hours-ahead value as today's actual value (already a column)."""
    return df_split[target].to_numpy()


def fit_linear_regression(X_train, y_train):
    # Ridge (light L2, alpha=1.0), not plain OLS: the lag/rolling feature
    # set is strongly autocorrelated at hourly resolution (e.g. lag1 and
    # lag2 are nearly identical), which makes the unregularised normal
    # equations ill-conditioned enough to blow coefficients up to +-inf.
    # Ridge keeps this a "linear regression" baseline while staying
    # numerically stable.
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    return model


def rolling_forecast_sarimax(train_series, eval_today_series, horizon=config.FORECAST_HORIZON_HOURS,
                               order=SARIMAX_ORDER, seasonal_order=SARIMAX_SEASONAL_ORDER):
    """Fit SARIMAX on the continuous training series, then walk forward
    through the continuous val+test hours: reveal this hour's true value
    to the model state first, THEN forecast ``horizon`` steps ahead and
    keep only the final step -- so ``preds[i]`` lines up with the same
    ``target_next`` (t+horizon) label every other model is scored against.
    Uses ``append(..., refit=False)`` (a Kalman-filter state update)
    instead of a full refit at every step.
    """
    model = SARIMAX(train_series.to_numpy(), order=order, seasonal_order=seasonal_order,
                     enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)

    preds = np.empty(len(eval_today_series))
    for i, today_true in enumerate(eval_today_series.to_numpy()):
        res = res.append([today_true], refit=False)
        preds[i] = res.forecast(steps=horizon)[-1]
    return preds
