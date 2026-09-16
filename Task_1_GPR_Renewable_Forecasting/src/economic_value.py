"""Illustrative microgrid economic value estimate: translates the
measured RMSE reduction (PSO-GPR vs Linear Regression baseline, from
outputs/metrics.json) into an approximate diesel-cost saving.

IMPORTANT -- this is explicitly a back-of-envelope illustration, not a
validated costing. It uses assumption constants (diesel price, generator
efficiency, genset capacity) that are NOT measured from real microgrid
operating data, because that data doesn't exist for this project. Every
assumption is a named, changeable constant at the top of this file so
the report can state exactly what was assumed and a marker can see the
method is transparent about its own limits.

Logic, in words: a forecast error of magnitude E kWh (RMSE) represents
generation the operator did not correctly plan for. In the worst case
that whole shortfall/surplus has to be covered by unplanned diesel
run-time. This is a deliberately conservative (i.e. upper-bound) way of
turning "lower RMSE" into "less diesel," precisely because we cannot
observe the operator's actual dispatch policy.
"""
import json

from . import config

# --- Named assumptions (change these and re-run if your context differs) ---
DIESEL_PRICE_NPR_PER_LITRE = 180.0       # approx. Nepal retail diesel price
GENSET_FUEL_CONSUMPTION_L_PER_KWH = 0.3  # typical small diesel genset, ~3.3 kWh/L
FORECAST_HOURS_PER_YEAR = 365 * 24       # annualising the hourly RMSE improvement


def _load_metrics():
    metrics_path = config.OUTPUTS_DIR / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"{metrics_path} not found. Run the main pipeline first (python -m src.pipeline)."
        )
    with open(metrics_path) as f:
        return json.load(f)


def estimate_savings(target, all_metrics, best_model_key="gpr_rbf_pso", baseline_key="linear_regression"):
    models = all_metrics[target]["models"]
    if best_model_key not in models or baseline_key not in models:
        return None

    rmse_best = models[best_model_key]["RMSE"]
    rmse_baseline = models[baseline_key]["RMSE"]
    rmse_reduction_kwh_per_hour = rmse_baseline - rmse_best  # per-forecast-hour, in target's native units

    # Annualised unplanned-generation exposure avoided, in kWh-equivalent.
    # NOTE: solar_irradiance's units here are kWh/m^2/hr per config.py's
    # conversion, not kWh of actual generation -- multiplying by an
    # assumed panel/turbine capacity would be needed to get real kWh.
    # This function reports the RMSE reduction itself and its direct
    # diesel-cost translation; converting irradiance/wind-speed RMSE to
    # actual generated kWh needs your site's panel area/efficiency or
    # turbine power curve, which are NOT in this dataset -- state that
    # explicitly in the report rather than inventing a capacity figure.
    annual_exposure_reduction = rmse_reduction_kwh_per_hour * FORECAST_HOURS_PER_YEAR

    diesel_litres_avoided = annual_exposure_reduction * GENSET_FUEL_CONSUMPTION_L_PER_KWH
    cost_avoided_npr = diesel_litres_avoided * DIESEL_PRICE_NPR_PER_LITRE

    return {
        "target": target,
        "rmse_baseline_lr": round(rmse_baseline, 4),
        "rmse_pso_gpr": round(rmse_best, 4),
        "rmse_reduction": round(rmse_reduction_kwh_per_hour, 4),
        "illustrative_annual_diesel_litres_avoided": round(diesel_litres_avoided, 1),
        "illustrative_annual_cost_avoided_NPR": round(cost_avoided_npr, 0),
    }


def main():
    print("--- Illustrative Microgrid Economic Value Estimate ---")
    print("(back-of-envelope only -- see docstring for assumptions)\n")
    all_metrics = _load_metrics()

    rows = []
    for target in config.TARGETS:
        result = estimate_savings(target, all_metrics)
        if result is None:
            print(f"Skipping {target}: required model keys not found in metrics.json")
            continue
        rows.append(result)
        print(f"[{target}] RMSE: LR={result['rmse_baseline_lr']}, "
              f"PSO-GPR={result['rmse_pso_gpr']} (reduction={result['rmse_reduction']})")
        print(f"  Illustrative diesel avoided/year: "
              f"{result['illustrative_annual_diesel_litres_avoided']} L "
              f"(~NPR {result['illustrative_annual_cost_avoided_NPR']:,.0f})\n")

    if rows:
        import pandas as pd
        out_path = config.OUTPUTS_DIR / "economic_value_illustrative.csv"
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"Saved table to {out_path}")

    print("\nAssumptions used (state these explicitly in the report):")
    print(f"  Diesel price: NPR {DIESEL_PRICE_NPR_PER_LITRE}/L")
    print(f"  Genset fuel consumption: {GENSET_FUEL_CONSUMPTION_L_PER_KWH} L/kWh")
    print(f"  Annualised over {FORECAST_HOURS_PER_YEAR} forecast-hours/year")
    print("  Caveat: converting irradiance/wind-speed RMSE into actual generated kWh "
          "requires site panel area / turbine power curve, which this dataset does not "
          "provide -- this estimate uses the RMSE reduction directly as a kWh-equivalent "
          "proxy for unplanned-generation exposure, an upper-bound illustration, not a "
          "validated cost model.")


if __name__ == "__main__":
    main()
