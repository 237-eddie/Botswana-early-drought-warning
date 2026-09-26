import streamlit as st
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

LEAD = 3
DISTRICT_NAME = "Southern District (Kanye)"

st.set_page_config(page_title="Botswana Drought Early Warning", layout="wide")

# ---------- 1. LOAD DATA + BUILD FEATURES (same pipeline as Models.py) ----------
@st.cache_data
def load_and_prepare():
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
    data = X.join(y.rename("target"))
    return df, m, X, y, data

@st.cache_resource
def train_model(X, y):
    data = X.join(y.rename("target")).dropna()
    Xc, yc = data.drop(columns="target"), data["target"]
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ])
    model.fit(Xc, yc)
    return model, Xc

df, m, X, y, data = load_and_prepare()
model, X_clean = train_model(X, y)

# ---------- 2. HEADER ----------
st.title("🌾 Botswana Drought Early Warning")
st.caption(f"District: {DISTRICT_NAME} · SPI-3 · {LEAD}-month-ahead forecast")

latest_date = m.index[-1]
latest_spi = m["spi"].iloc[-1]
currently_in_drought = latest_spi <= -1

# Most recent row usable for prediction (features exist, doesn't need future target)
latest_features = X.iloc[[-1]]
risk_proba = model.predict_proba(latest_features)[0, 1]
risk_label = "HIGH RISK" if risk_proba >= 0.5 else "Low risk"

col1, col2, col3 = st.columns(3)
col1.metric("Current SPI-3", f"{latest_spi:.2f}", "Drought" if currently_in_drought else "Normal")
col2.metric(f"Drought risk in {LEAD} months", f"{risk_proba:.0%}", risk_label)
col3.metric("Data through", latest_date.strftime("%b %Y"))

if risk_proba >= 0.5:
    st.warning(
        f"⚠️ Model flags elevated drought risk for {(latest_date + pd.DateOffset(months=LEAD)).strftime('%B %Y')}. "
        f"Note: this model has ~21% precision historically — treat as an early signal, not a certainty."
    )
else:
    st.success(f"✅ No elevated drought risk flagged for {(latest_date + pd.DateOffset(months=LEAD)).strftime('%B %Y')}.")

# ---------- 3. SPI HISTORY CHART ----------
st.subheader("SPI-3 History")
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(m.index, m["spi"], lw=1, color="steelblue")
ax.axhline(-1, color="red", ls="--", label="Drought threshold (SPI = -1)")
drought_points = m[m["spi"] <= -1]
ax.scatter(drought_points.index, drought_points["spi"], color="red", s=15, zorder=3)
ax.set_ylabel("SPI-3")
ax.legend()
st.pyplot(fig)

# ---------- 4. RAINFALL CHART ----------
st.subheader("Monthly Rainfall")
fig2, ax2 = plt.subplots(figsize=(12, 3))
ax2.bar(m.index, m["precip"], width=20, color="skyblue")
ax2.set_ylabel("Rainfall (mm)")
st.pyplot(fig2)

# ---------- 5. MODEL INFO ----------
with st.expander("About this model"):
    st.markdown(f"""
    - **Model:** Logistic Regression (class-balanced), no ENSO features
    - **Test-fold performance (2024-2026):** F1 = 0.333, Recall = 75%, Precision = 21%
    - **Baseline comparison:** beats climatology baseline (F1 = 0.272)
    - **Known limitation:** the model favors catching real droughts (high recall) over
      avoiding false alarms (low precision). Error analysis showed false alarms are
      not timing errors near real droughts — they reflect a genuine calibration
      limitation rather than a tunable decision threshold. Treat predictions as an
      early-warning signal to investigate further, not a definitive forecast.
    """)