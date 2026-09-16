"""Feature engineering for the hourly dataset.

Unlike the daily resolution, this data has a real HR column, so "time of
day" -- one of the six input parameters named in the proposal -- is used
directly (as a cyclic hour-of-day encoding) instead of being approximated.
Solar zenith angle is now computed per-hour (using an hour angle term),
not just at solar noon.
"""
import numpy as np

from . import config

LAGS = (1, 2, 3, 6, 12, 24)       # hours
# NOTE: a window of 3 is deliberately excluded here -- rolling(3) of
# shift(1) would be the exact arithmetic mean of lag1+lag2+lag3, an exact
# linear combination of features already in LAGS. Including it made the
# design matrix exactly rank-deficient (confirmed: rank 41 of 45 columns,
# condition number ~1e16) and blew up Linear Regression's coefficients.
ROLL_WINDOWS = (6, 24)            # hours
# cloud_amount added as the 5th input variable named in the proposal
# (CLOUD_AMT from NASA POWER). Requires the raw CSV to actually contain
# a CLOUD_AMT column -- see the assertion in build_feature_frame() below,
# which fails fast with a clear message if it's missing rather than
# letting a silent KeyError surface deep inside feature engineering.
BASE_COLUMNS = ("solar_irradiance", "wind_speed", "temperature", "humidity", "cloud_amount")


def _solar_zenith_deg(day_of_year, hour_of_day, latitude_deg):
    """Solar zenith angle (degrees) for a given day-of-year and clock hour.

    Simplification: clock hour is treated as local solar hour (the data is
    already in LST per the NASA header), i.e. no equation-of-time /
    longitude-standard-meridian correction is applied. Adequate for a
    smooth continuous input feature; not adequate for precision solar
    engineering.
    """
    declination = 23.45 * np.sin(np.deg2rad(360.0 / 365.0 * (day_of_year - 81)))
    hour_angle = 15.0 * ((hour_of_day + 0.5) - 12.0)  # +0.5: hour bucket midpoint
    lat_rad = np.deg2rad(latitude_deg)
    dec_rad = np.deg2rad(declination)
    ha_rad = np.deg2rad(hour_angle)
    cos_zenith = (
        np.sin(lat_rad) * np.sin(dec_rad)
        + np.cos(lat_rad) * np.cos(dec_rad) * np.cos(ha_rad)
    )
    cos_zenith = np.clip(cos_zenith, -1.0, 1.0)
    return np.rad2deg(np.arccos(cos_zenith))


def add_calendar_features(df):
    df = df.copy()
    doy = df["date"].dt.dayofyear
    hour = df["date"].dt.hour
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["solar_zenith"] = _solar_zenith_deg(doy, hour, config.LATITUDE)
    
    # Air Mass calculation clipped to prevent division by zero near sunset
    cos_zenith = np.cos(np.deg2rad(df["solar_zenith"]))
    df["air_mass"] = np.where(df["solar_zenith"] < 89.0, 1.0 / np.maximum(cos_zenith, 0.01), 38.0)
    
    # Wind Speed Cubed (proportional to wind power kinetic energy)
    df["wind_speed_cubed"] = df["wind_speed"] ** 3
    
    # Empirical daylight flag (measured irradiance > 0), used to filter the
    # solar target down to hours that matter -- not a modelling input, so
    # it is not part of FEATURE_COLUMNS.
    df["is_daylight"] = df["solar_irradiance"] > 0
    return df


def add_lag_and_rolling_features(df, columns=BASE_COLUMNS):
    df = df.copy()
    for col in columns:
        for lag in LAGS:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
        for win in ROLL_WINDOWS:
            df[f"{col}_roll{win}"] = df[col].shift(1).rolling(win).mean()
    return df


def add_horizon_targets(df, targets=config.TARGETS, horizon=config.FORECAST_HORIZON_HOURS):
    """Day-ahead framing: predict the SAME hour ``horizon`` hours later
    (24h = tomorrow, same hour) from today's weather + recent history.
    """
    df = df.copy()
    for t in targets:
        df[f"{t}_next"] = df[t].shift(-horizon)
    return df


FEATURE_COLUMNS = (
    list(BASE_COLUMNS)
    + ["doy_sin", "doy_cos", "hour_sin", "hour_cos", "solar_zenith", "air_mass", "wind_speed_cubed"]
    + [f"{c}_lag{l}" for c in BASE_COLUMNS for l in LAGS]
    + [f"{c}_roll{w}" for c in BASE_COLUMNS for w in ROLL_WINDOWS]
)


def build_feature_frame(df):
    """Full pipeline: calendar + lag/rolling features + horizon targets,
    then drop rows with NaNs introduced by lagging/shifting at the edges.
    """
    missing_base_cols = [c for c in BASE_COLUMNS if c not in df.columns]
    if missing_base_cols:
        raise ValueError(
            f"Raw data is missing required column(s): {missing_base_cols}. "
            f"If 'cloud_amount' is missing, the source CSV was not pulled with the "
            f"CLOUD_AMT parameter -- re-fetch NASA POWER data including CLOUD_AMT "
            f"(config.RAW_COLUMNS maps NASA's 'CLOUD_AMT' -> 'cloud_amount'), or "
            f"remove 'cloud_amount' from features.BASE_COLUMNS if it is intentionally unavailable."
        )
    out = add_calendar_features(df)
    out = add_lag_and_rolling_features(out)
    out = add_horizon_targets(out)
    required = FEATURE_COLUMNS + [f"{t}_next" for t in config.TARGETS] + ["date", "is_daylight"]
    out = out.dropna(subset=required).reset_index(drop=True)
    return out


if __name__ == "__main__":
    from . import data_prep

    df, _ = data_prep.load_clean()
    feat = build_feature_frame(df)
    print(f"Feature frame: {feat.shape[0]} rows x {len(FEATURE_COLUMNS)} input features")
    print(f"Daylight hours: {feat['is_daylight'].sum()} / {len(feat)}")
    print("Inputs used for GPR (>=4 required):", FEATURE_COLUMNS)
