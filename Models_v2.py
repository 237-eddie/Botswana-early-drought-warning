import pandas as pd, numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

LEAD = 3
FOLDS = [("2015","2017"), ("2018","2020"), ("2021","2023"), ("2024","2026")]

# ---------- 1. DATA + LOCAL FEATURES (same as before) ----------
df = pd.read_csv("spi3_history.csv", parse_dates=["date"]).sort_values("date")
m = df.set_index("date").resample("ME").agg(
    spi=("spi","last"), precip=("rainfall_mm","sum"), temp=("temp_c","mean")).dropna()

X = pd.DataFrame(index=m.index)
X["spi_now"]=m["spi"]; X["spi_lag1"]=m["spi"].shift(1); X["spi_lag2"]=m["spi"].shift(2)
X["spi_roll3"]=m["spi"].rolling(3).mean()
X["precip_roll3"]=m["precip"].rolling(3).sum()
X["precip_roll6"]=m["precip"].rolling(6).sum()
X["precip_roll12"]=m["precip"].rolling(12).sum()
X["temp_anom12"]=m["temp"]-m["temp"].rolling(12,min_periods=6).mean()

# ---------- 2. ADD ENSO (Nino 3.4 anomaly from NOAA) ----------
URL="https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/detrend.nino34.ascii.txt"
raw = pd.read_csv(URL, sep=r"\s+")
# FIX: the file's real columns are YR, MON, TOTAL, ClimAdjust, ANOM.
# There is no "NINO3.4" column, so the old cols.index("NINO3.4") lookup
# always raised ValueError. The anomaly value is just the "ANOM" column.
anom_col = "ANOM"
enso = pd.DataFrame({
    "per": pd.to_datetime(raw["YR"].astype(str) + "-" + raw["MON"].astype(str), format="%Y-%m").dt.to_period("M"),
    "nino34": raw[anom_col]
}).set_index("per").sort_index()

per = m.index.to_period("M")
X["enso_now"]   = pd.Series(enso["nino34"].reindex(per).values, index=m.index)
X["enso_lag1"]  = X["enso_now"].shift(1)
X["enso_lag2"]  = X["enso_now"].shift(2)
X["enso_roll3"] = X["enso_now"].rolling(3).mean()

y = (m["spi"].shift(-LEAD) <= -1).astype(int)
data = X.join(y.rename("target")).dropna()
X, y = data.drop(columns="target"), data["target"]
print(f"Dataset: {len(data)} months | features: {list(X.columns)}\n")

# ---------- 3. MODELS WITH WALK-FORWARD CV ----------
models = {
    "logistic_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000))]),
    "random_forest": RandomForestClassifier(
        n_estimators=300, max_depth=4, class_weight="balanced", random_state=42),
    "xgboost": None,  # built per-fold below (imbalance ratio differs per train set)
}

for name, model in models.items():
    f1s = []
    for start, end in FOLDS:
        tr = X.index < f"{start}-01-01"
        te = (X.index >= f"{start}-01-01") & (X.index <= f"{end}-12-31")
        if tr.sum() < 60 or te.sum() < 12: continue
        if name == "xgboost":
            spw = (y[tr]==0).sum() / max((y[tr]==1).sum(), 1)   # scale_pos_weight
            model = XGBClassifier(n_estimators=200, max_depth=2, learning_rate=0.05,
                                  subsample=0.8, scale_pos_weight=spw,
                                  random_state=42, eval_metric="logloss")
        model.fit(X[tr], y[tr])
        f1 = f1_score(y[te], model.predict(X[te]))
        f1s.append(f1)
        print(f"  {name} | {start}-{end}: F1 = {f1:.3f}")
    print(f"  >>> {name} average F1: {np.mean(f1s):.3f} (without ENSO was: LR 0.322 / RF 0.315)\n")

# ---------- 4. WHAT DRIVES PREDICTIONS? ----------
rf = RandomForestClassifier(n_estimators=300, max_depth=4,
                            class_weight="balanced", random_state=42).fit(X, y)
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
print("Random forest feature importance (top -> bottom):")
print(imp.round(3).to_string())