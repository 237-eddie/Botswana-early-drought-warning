import pandas as pd, numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix

LEAD = 3

# ---------- 1. BUILD FEATURES (same as Models.py — no ENSO) ----------
df = pd.read_csv("spi3_history.csv", parse_dates=["date"]).sort_values("date")
m = df.set_index("date").resample("ME").agg(
    spi=("spi", "last"), precip=("rainfall_mm", "sum"), temp=("temp_c", "mean")
).dropna()

X = pd.DataFrame(index=m.index)
X["spi_now"]  = m["spi"]
X["spi_lag1"] = m["spi"].shift(1)
X["spi_lag2"] = m["spi"].shift(2)
X["spi_roll3"]     = m["spi"].rolling(3).mean()
X["precip_roll3"]  = m["precip"].rolling(3).sum()
X["precip_roll6"]  = m["precip"].rolling(6).sum()
X["precip_roll12"] = m["precip"].rolling(12).sum()
X["temp_anom12"]   = m["temp"] - m["temp"].rolling(12, min_periods=6).mean()

y = (m["spi"].shift(-LEAD) <= -1).astype(int)
data = X.join(y.rename("target")).dropna()
X, y = data.drop(columns="target"), data["target"]

# ---------- 2. TRAIN ON EVERYTHING BEFORE 2024, TEST ON 2024-2026 ----------
tr = X.index < "2024-01-01"
te = (X.index >= "2024-01-01") & (X.index <= "2026-12-31")

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=1000))
])
model.fit(X[tr], y[tr])
pred = model.predict(X[te])
proba = model.predict_proba(X[te])[:, 1]

y_te = y[te]
f1 = f1_score(y_te, pred)
print(f"2024-2026 fold F1: {f1:.3f}\n")

# ---------- 3. MONTH-BY-MONTH BREAKDOWN ----------
report = pd.DataFrame({
    "actual_drought_in_3mo": y_te.values,
    "predicted_drought_in_3mo": pred,
    "model_confidence": proba.round(2),
}, index=y_te.index)

def label_row(row):
    if row["actual_drought_in_3mo"] == 1 and row["predicted_drought_in_3mo"] == 1:
        return "HIT"
    if row["actual_drought_in_3mo"] == 1 and row["predicted_drought_in_3mo"] == 0:
        return "MISS (missed a real drought)"
    if row["actual_drought_in_3mo"] == 0 and row["predicted_drought_in_3mo"] == 1:
        return "FALSE ALARM"
    return "correct no-drought"

report["result"] = report.apply(label_row, axis=1)
print(report.to_string())

# ---------- 4. ONSET vs ONGOING: are misses concentrated at drought START? ----------
actual = report["actual_drought_in_3mo"]
is_onset = (actual == 1) & (actual.shift(1, fill_value=0) == 0)   # first month of a new drought run
misses = report["result"] == "MISS (missed a real drought)"

onset_miss_rate = (misses & is_onset).sum() / max(is_onset.sum(), 1)
ongoing_miss_rate = (misses & ~is_onset & (actual == 1)).sum() / max(((actual == 1) & ~is_onset).sum(), 1)

print(f"\n--- ONSET vs ONGOING ---")
print(f"Onset months (start of a new drought): {is_onset.sum()}, missed: {(misses & is_onset).sum()} "
      f"({onset_miss_rate:.0%} miss rate)")
print(f"Ongoing drought months: {((actual==1) & ~is_onset).sum()}, missed: "
      f"{(misses & ~is_onset & (actual==1)).sum()} ({ongoing_miss_rate:.0%} miss rate)")

print("\n--- CONFUSION MATRIX ---")
tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
print(f"True positives (caught droughts): {tp}")
print(f"False negatives (missed droughts): {fn}")
print(f"False positives (false alarms): {fp}")
print(f"True negatives (correct no-drought): {tn}")

# ---------- 5. FALSE ALARM CLUSTERING ----------
# For each false alarm, find how many months away the nearest REAL drought month is.
# Close to a real drought = "near-miss" (defensible). Far from any drought = genuine noise.
actual_drought_months = actual[actual == 1].index
false_alarms = report[report["result"] == "FALSE ALARM"]

print(f"\n--- FALSE ALARM CLUSTERING ({len(false_alarms)} false alarms) ---")
if len(actual_drought_months) == 0:
    print("No actual drought months in this window to compare against.")
else:
    distances = []
    for fa_date in false_alarms.index:
        gaps_months = [abs((fa_date.to_period("M") - d.to_period("M")).n)
                       for d in actual_drought_months]
        nearest = min(gaps_months)
        distances.append(nearest)
        tag = "NEAR-MISS (within 1 month of a real drought)" if nearest <= 1 else \
              "NEARBY (2-3 months from a real drought)" if nearest <= 3 else \
              "ISOLATED (4+ months from any real drought)"
        print(f"  {fa_date.date()}: nearest real drought is {nearest} month(s) away -> {tag}")

    distances = np.array(distances)
    print(f"\nSummary: {(distances <= 1).sum()} near-miss, "
          f"{((distances > 1) & (distances <= 3)).sum()} nearby, "
          f"{(distances > 3).sum()} isolated")
    print(f"Median distance from false alarm to nearest real drought: {np.median(distances):.1f} months")

# ---------- 6. THRESHOLD SWEEP: does raising the cutoff trade false alarms for recall? ----------
print("\n--- THRESHOLD SWEEP (default is 0.5) ---")
print(f"{'threshold':>10} | {'F1':>6} | {'precision':>9} | {'recall':>7} | {'TP':>3} {'FP':>3} {'FN':>3} {'TN':>3}")
for thresh in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
    pred_t = (proba >= thresh).astype(int)
    tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_te, pred_t).ravel()
    precision_t = tp_t / max(tp_t + fp_t, 1)
    recall_t = tp_t / max(tp_t + fn_t, 1)
    f1_t = f1_score(y_te, pred_t, zero_division=0)
    print(f"{thresh:>10.2f} | {f1_t:>6.3f} | {precision_t:>9.3f} | {recall_t:>7.3f} | "
          f"{tp_t:>3} {fp_t:>3} {fn_t:>3} {tn_t:>3}")