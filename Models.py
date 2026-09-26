import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

LEAD = 3   # start with the hardest, most valuable one

# ---------- 1. BUILD MONTHLY FEATURES ----------
df = pd.read_csv("spi3_history.csv", parse_dates=["date"]).sort_values("date")
m = df.set_index("date").resample("ME").agg(
    spi=("spi", "last"), precip=("rainfall_mm", "sum"), temp=("temp_c", "mean")
).dropna()

X = pd.DataFrame(index=m.index)
X["spi_now"]  = m["spi"]
X["spi_lag1"] = m["spi"].shift(1)
X["spi_lag2"] = m["spi"].shift(2)
X["spi_roll3"]    = m["spi"].rolling(3).mean()
X["precip_roll3"]  = m["precip"].rolling(3).sum()
X["precip_roll6"]  = m["precip"].rolling(6).sum()
X["precip_roll12"] = m["precip"].rolling(12).sum()
X["temp_anom12"]   = m["temp"] - m["temp"].rolling(12, min_periods=6).mean()

y = (m["spi"].shift(-LEAD) <= -1).astype(int)   # drought 'LEAD' months ahead

data = X.join(y.rename("target")).dropna()
X, y = data.drop(columns="target"), data["target"]
print(f"Dataset: {len(data)} months, {y.mean():.1%} drought rate")

# ---------- 2. WALK-FORWARD VALIDATION ----------
# Train on the past, test on the future. Last fold = the recent extreme drought years.
FOLDS = [("2015","2017"), ("2018","2020"), ("2021","2023"), ("2024","2026")]

models = {
    "logistic_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ]),
    "random_forest": RandomForestClassifier(
        n_estimators=300, max_depth=4, class_weight="balanced", random_state=42
    ),
}

for name, model in models.items():
    f1s = []
    for start, end in FOLDS:
        train_mask = X.index < f"{start}-01-01"
        test_mask = (X.index >= f"{start}-01-01") & (X.index <= f"{end}-12-31")
        if train_mask.sum() < 60 or test_mask.sum() < 12:
            continue
        model.fit(X[train_mask], y[train_mask])
        pred = model.predict(X[test_mask])
        f1 = f1_score(y[test_mask], pred)
        f1s.append(f1)
        print(f"  {name} | {start}-{end}: F1 = {f1:.3f}")
    print(f"  >>> {name} average F1: {np.mean(f1s):.3f} (target: beat 0.272)\n")