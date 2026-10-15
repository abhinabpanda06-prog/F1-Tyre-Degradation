import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fastf1
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox

warnings.filterwarnings("ignore", message="adfuller currently returns")

YEAR = 2026
EVENT = "Singapore"
SESSION_TYPE = "R"
DRIVER = "LEC"
STINT_NUMBER = None
SHOW_DIAGNOSTICS = False

os.makedirs("f1_cache", exist_ok=True)
fastf1.Cache.enable_cache("f1_cache")

session = fastf1.get_session(YEAR, EVENT, SESSION_TYPE)
session.load(telemetry=False, weather=False)

driver_laps = session.laps.pick_drivers(DRIVER)
clean_all = driver_laps.pick_wo_box().pick_track_status("1").pick_accurate()

summary = clean_all.groupby("Stint").agg(
    compound=("Compound", "first"),
    clean_laps=("LapNumber", "count"),
    first_lap=("LapNumber", "min"),
    last_lap=("LapNumber", "max"),
)

if summary.empty:
    raise SystemExit(f"No clean laps found for {DRIVER} at {YEAR} {EVENT}.")

if STINT_NUMBER is None:
    STINT_NUMBER = int(summary["clean_laps"].idxmax())

stint = clean_all[clean_all["Stint"] == STINT_NUMBER]
stint = stint[stint["LapNumber"] > 1].pick_quicklaps().sort_values("LapNumber")

if len(stint) < 10:
    raise SystemExit("Fewer than 10 clean laps. Select another stint, driver, or race.")

lap_numbers = stint["LapNumber"].to_numpy()
ts = stint["LapTime"].dt.total_seconds().reset_index(drop=True)
compound = str(stint["Compound"].iloc[0])

n_test = max(3, len(ts) // 5)
train = ts.iloc[:-n_test]
test = ts.iloc[-n_test:]

slope, intercept = np.polyfit(np.arange(len(train)), train.to_numpy(), 1)
naive_fc = np.repeat(train.iloc[-1], n_test)
linear_fc = intercept + slope * np.arange(len(train), len(ts))

adf_p = adfuller(ts.diff().dropna())[1]

TREND = "t"
candidates = [(1, 1, 0), (0, 1, 1), (1, 1, 1), (2, 1, 1), (1, 1, 2)]
aic_rows = []

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for order in candidates:
        try:
            model = ARIMA(train, order=order, trend=TREND).fit()
            aic_rows.append((order, model.aic))
        except Exception:
            continue

if not aic_rows:
    raise SystemExit("None of the candidate ARIMA models could be fitted.")

aic_rows.sort(key=lambda item: item[1])
ORDER = aic_rows[0][0]

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fit = ARIMA(train, order=ORDER, trend=TREND).fit()
    fc_res = fit.get_forecast(steps=n_test)
    arima_fc = fc_res.predicted_mean.to_numpy()
    conf_int = fc_res.conf_int()

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        alt_fc = ARIMA(train, order=(1, 0, 0), trend="ct").fit().forecast(n_test).to_numpy()
except Exception:
    alt_fc = None

def score(name, predictions):
    errors = test.to_numpy() - np.asarray(predictions, dtype=float)
    return {
        "Model": name,
        "RMSE (s)": np.sqrt(np.mean(errors ** 2)),
        "MAE (s)": np.mean(np.abs(errors))
    }

rows = [
    score("Naive", naive_fc),
    score("Linear trend (OLS)", linear_fc),
    score(f"ARIMA{ORDER} + drift", arima_fc)
]

if alt_fc is not None:
    rows.append(score("ARIMA(1,0,0) + trend", alt_fc))

results = pd.DataFrame(rows).set_index("Model").round(3)
best_aic = aic_rows[0][1]

resid = fit.resid.iloc[ORDER[1]:]
model_df = ORDER[0] + ORDER[2]
lb_lag = min(5, max(1, len(resid) // 3))
lb_lag = max(model_df + 1, lb_lag)

try:
    lb_p = acorr_ljungbox(
        resid,
        lags=[lb_lag],
        model_df=model_df,
        return_df=True
    )["lb_pvalue"].iloc[0]
except Exception:
    lb_p = np.nan

print("\nF1 TYRE DEGRADATION FORECAST")
print("=" * 34)
print(f"Race: {YEAR} {EVENT} | Driver: {DRIVER}")
print(f"Stint: {STINT_NUMBER} | Compound: {compound} | Clean laps: {len(ts)}")
print(f"OLS net degradation slope: {slope:+.3f} s/lap")
print(f"ADF p-value after first differencing: {adf_p:.4g}")
print(f"Selected model: ARIMA{ORDER} + drift")
print(f"Selected model AIC: {best_aic:.3f}")
print(f"Forecast horizon: {n_test} held-out laps")

print("\nFORECAST PERFORMANCE")
print(results.to_string())

print("\nDIAGNOSTICS")
if np.isnan(lb_p):
    print("Ljung-Box test: unavailable for this sample/model.")
else:
    print(f"Ljung-Box p-value (lag {lb_lag}): {lb_p:.4f}")

best_rmse_model = results["RMSE (s)"].idxmin()
best_mae_model = results["MAE (s)"].idxmin()
print(f"Lowest test RMSE: {best_rmse_model}")
print(f"Lowest test MAE: {best_mae_model}")

if SHOW_DIAGNOSTICS:
    print("\nADF test on raw series:")
    print(f"p-value with constant: {adfuller(ts)[1]:.4f}")
    print(f"p-value with constant and trend: {adfuller(ts, regression='ct')[1]:.4f}")
    print("\nAIC by candidate ARIMA order:")
    for order, aic in aic_rows:
        print(f"ARIMA{order}: {aic:.3f}")
    print("\nFull model summary:")
    print(fit.summary())

plt.figure(figsize=(9, 5))
plt.plot(lap_numbers[:len(train)], train.to_numpy(), marker="o", label="Training laps")
plt.plot(lap_numbers[-n_test:], test.to_numpy(), marker="o", label="Actual test laps")
plt.plot(lap_numbers[-n_test:], arima_fc, marker="x", label=f"ARIMA{ORDER} forecast")
plt.plot(lap_numbers[-n_test:], linear_fc, marker="^", linestyle="--", label="Linear trend forecast")
plt.fill_between(
    lap_numbers[-n_test:],
    conf_int.iloc[:, 0].to_numpy(),
    conf_int.iloc[:, 1].to_numpy(),
    alpha=0.2,
    label="ARIMA 95% confidence interval"
)
plt.title(f"{DRIVER} {YEAR} {EVENT}: Actual vs Predicted Lap Times")
plt.xlabel("Lap number")
plt.ylabel("Lap time (seconds)")
plt.legend()
plt.tight_layout()
plt.savefig("forecast_vs_actual.png", dpi=150)
plt.show()

print("\nForecast plot saved as forecast_vs_actual.png")
