"""Load and clean the NASA POWER hourly CSV.

Two source formats are supported, auto-detected from the first line:

1. NASA POWER's native Data Access Viewer export: a variable-length
   free-text header terminated by the literal line "-END HEADER-",
   followed by YEAR,MO,DY,HR,<params> columns.
2. A flat, pre-header-stripped export (e.g. from the POWER point API,
   possibly reformatted by a third-party script): a plain CSV starting
   with a "Location,Latitude,Longitude,..." header row and its own
   Year/Month/Day/Hour columns.

Both report hourly irradiance in Wh/m^2; both are converted to kWh/m^2
here so the scale matches the "kWh" language used throughout the proposal.
"""
import numpy as np
import pandas as pd

from . import config


def _find_header_end(path):
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if line.strip() == "-END HEADER-":
                return i
    raise ValueError(f"Could not find '-END HEADER-' marker in {path}")


def _peek_first_line(path):
    with open(path, "r") as f:
        return f.readline().strip()


def _load_native_format(path):
    skip = _find_header_end(path) + 1
    df = pd.read_csv(path, skiprows=skip)
    df = df.rename(columns=config.RAW_COLUMNS)
    df["date"] = pd.to_datetime(
        dict(year=df["YEAR"], month=df["MO"], day=df["DY"], hour=df["HR"])
    )
    return df.drop(columns=["YEAR", "MO", "DY", "HR"])


def _load_flat_format(path):
    df = pd.read_csv(path)
    df = df.rename(columns=config.RAW_COLUMNS)
    df["date"] = pd.to_datetime(
        dict(year=df["Year"], month=df["Month"], day=df["Day"], hour=df["Hour"])
    )
    keep = ["date"] + list(config.RAW_COLUMNS.values())
    return df[keep]


def load_raw(path=config.RAW_CSV):
    first_line = _peek_first_line(path)
    if first_line == "-BEGIN HEADER-":
        df = _load_native_format(path)
    elif first_line.startswith("Location,"):
        df = _load_flat_format(path)
    else:
        raise ValueError(f"Unrecognised CSV format in {path} (first line: {first_line!r})")

    df = df.sort_values("date").reset_index(drop=True)
    # NASA reports hourly irradiance in Wh/m^2; convert to kWh/m^2 so the
    # scale matches the "kWh" language used throughout the proposal.
    df["solar_irradiance"] = df["solar_irradiance"] / 1000.0
    return df


def clean(df):
    df = df.copy()
    value_cols = list(config.RAW_COLUMNS.values())
    # wind_speed/temperature/humidity keep the raw -999 sentinel; solar_irradiance
    # was already divided by 1000 in load_raw(), so its sentinel is -0.999 too.
    df["solar_irradiance"] = df["solar_irradiance"].replace(config.MISSING_SENTINEL / 1000.0, np.nan)
    other_cols = [c for c in value_cols if c != "solar_irradiance"]
    df[other_cols] = df[other_cols].replace(config.MISSING_SENTINEL, np.nan)

    n_missing = df[value_cols].isna().sum().sum()
    if n_missing:
        df = df.set_index("date")
        df[value_cols] = df[value_cols].interpolate(method="time", limit_direction="both")
        df = df.reset_index()

    # Physical bounds: irradiance and wind speed cannot be negative.
    df["solar_irradiance"] = df["solar_irradiance"].clip(lower=0)
    df["wind_speed"] = df["wind_speed"].clip(lower=0)
    return df, int(n_missing)


def load_clean(path=config.RAW_CSV):
    """Convenience wrapper: load + clean in one call, returns (df, n_missing_filled)."""
    return clean(load_raw(path))


if __name__ == "__main__":
    df, n_missing = load_clean()
    print(f"Loaded {len(df)} rows spanning {df['date'].min()} to {df['date'].max()}")
    print(f"Filled {n_missing} missing (-999) cells via time-interpolation")
    print(df.describe())
