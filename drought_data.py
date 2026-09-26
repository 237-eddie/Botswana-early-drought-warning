import requests
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
DISTRICTS = {
    "Gaborone":      (-24.63, 25.92),
    "Southern (Kanye)":  (-24.98, 25.34),
    "Kgalagadi (Tshabong)": (-26.02, 22.40),
}
DISTRICT = "Southern (Kanye)"        # <- pick yours
LAT, LON = DISTRICTS[DISTRICT]
START, END = "20010101", "20261231"

# ---------- 1. PULL NASA POWER DATA ----------
url = (
    "https://power.larc.nasa.gov/api/temporal/daily/point"
    f"?parameters=PRECTOTCORR,T2M&community=AG"
    f"&longitude={LON}&latitude={LAT}"
    f"&start={START}&end={END}&format=JSON"
)
resp = requests.get(url, timeout=60).json()
params = resp["properties"]["parameter"]

df = pd.DataFrame({
    "date": pd.to_datetime(list(params["PRECTOTCORR"].keys()), format="%Y%m%d"),
    "rainfall_mm": pd.to_numeric(list(params["PRECTOTCORR"].values())),
    "temp_c": pd.to_numeric(list(params["T2M"].values())),
})
df = df.replace(-999, np.nan).dropna().sort_values("date").reset_index(drop=True)
df.to_csv(f"nasa_power_{DISTRICT.split()[0].lower()}.csv", index=False)
print(f"✅ {len(df)} days of data ({df['date'].min().date()} to {df['date'].max().date()})")

# ---------- 2. COMPUTE SPI-3 ----------
df["precip_90d"] = df["rainfall_mm"].rolling(90).sum()
spi_df = df.dropna(subset=["precip_90d"]).copy()
spi_df["month"] = spi_df["date"].dt.month

# Fit gamma distribution separately per calendar month (standard SPI method)
fits = {}
for m, g in spi_df.groupby("month"):
    vals = g["precip_90d"].values
    p_zero = (vals <= 0).mean()
    nz = vals[vals > 0]
    if len(nz) < 5:
        continue
    shape, _, scale = stats.gamma.fit(nz, floc=0)
    fits[m] = (p_zero, shape, scale)

def to_spi(row):
    p_zero, shape, scale = fits[row["month"]]
    v = row["precip_90d"]
    cdf = p_zero if v <= 0 else p_zero + (1 - p_zero) * stats.gamma.cdf(v, a=shape, scale=scale)
    cdf = min(max(cdf, 1e-6), 1 - 1e-6)     # avoid inf
    return stats.norm.ppf(cdf)

spi_df["spi"] = spi_df.apply(to_spi, axis=1)
spi_df.to_csv("spi3_history.csv", index=False)

# ---------- 3. PLOT DROUGHT HISTORY ----------
drought = spi_df[spi_df["spi"] <= -1]
plt.figure(figsize=(14, 4))
plt.plot(spi_df["date"], spi_df["spi"], lw=1)
plt.axhline(-1, color="red", ls="--", label="Drought threshold (SPI = −1)")
plt.scatter(drought["date"], drought["spi"], color="red", s=8)
plt.ylabel("SPI-3"); plt.legend()
plt.title(f"Drought history — {DISTRICT} (2001–2026)")
plt.tight_layout(); plt.show()
print(f"Drought months detected: {len(drought)} / {len(spi_df)}")