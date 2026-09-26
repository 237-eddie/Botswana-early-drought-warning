# 🌾 Botswana Drought Early Warning

A machine learning system that predicts, 3 months in advance, whether **Southern District (Kanye), Botswana** will enter drought conditions — using the Standardized Precipitation Index (SPI-3).

**🔗 Live dashboard:** https://botswana-early-drought-warning-ab3ypevaxlcdmlktrq9yjx.streamlit.app

---

## Problem Statement

Given historical daily rainfall and temperature for a Botswana district, can we predict — 1 to 3 months in advance — whether the district will enter drought, using SPI-3?

Drought early warning gives farmers, water authorities, and local government a lead time to prepare (adjust planting, manage water reserves, plan relief) rather than reacting after a drought has already set in.

**Scope (deliberately narrow):** one district (Southern District / Kanye), one index (SPI-3), one primary horizon (3 months ahead). This discipline kept the project achievable in the available time rather than sprawling across multiple districts and indices.

---

## Data Sources

| Source | What it provides |
|---|---|
| [NASA POWER](https://power.larc.nasa.gov) | Daily rainfall (PRECTOTCORR) and temperature (T2M), 2001–2026, via free API |
| [NOAA CPC](https://www.cpc.ncep.noaa.gov) | Nino 3.4 ENSO index (tested as a feature; see Results) |

SPI-3 was computed from 90-day rolling precipitation totals, fit to a gamma distribution separately per calendar month (the standard SPI methodology), then transformed to a standard normal Z-score.

---

## Methodology

1. **Feature engineering:** SPI lags (1–2 months), 3-month rolling SPI mean, rolling precipitation sums (3/6/12-month), and a 12-month temperature anomaly.
2. **Baselines:** climatology (is this calendar month historically more drought-prone than average?) and persistence (does this month's drought status carry into next month?) — computed before any ML model, as the actual bar to beat.
3. **Models:** Logistic Regression, Random Forest, and XGBoost, each evaluated with **walk-forward (rolling-origin) cross-validation** — never a random train/test split, since that would leak future information into training for a time series problem.
4. **Evaluation metric:** F1-score, chosen over accuracy because drought months are a minority class; a model that always predicts "no drought" would still look accurate while being useless.

---

## Results

### 3-month-ahead F1 scores

| Method | F1 |
|---|---|
| Always predict no drought | 0.000 |
| Persistence baseline | 0.212 |
| **Climatology baseline** | **0.272** |
| Logistic Regression (no ENSO) | 0.322 |
| Random Forest (no ENSO) | 0.315 |

The checkpoint model — **Logistic Regression, no ENSO features** — beats the climatology baseline by a solid margin. This is the model used in the live dashboard.

### Feature importance (Random Forest)

`precip_roll3` (3-month rolling rainfall) was the strongest predictor by a clear margin, followed by `precip_roll6` and `temp_anom12`. SPI's own lagged values contributed less than raw precipitation trends — an interesting finding worth further investigation.

### Did adding ENSO (Nino 3.4) help?

No — not consistently. Logistic regression saw a small gain (0.322 → 0.338), but Random Forest and XGBoost both got noticeably *worse* with ENSO added (0.315 → 0.204, and 0.254 respectively), including a complete failure (F1 = 0.000) on one test fold. ENSO features also ranked lowest in feature importance. **ENSO was dropped from the final model** — a legitimate negative result, not just an omission.

### Error analysis (2024–2026 test fold)

- 3 true positives, 1 false negative, 11 false positives, 18 true negatives
- **Recall ≈ 75%, precision ≈ 21%** — the model favors catching real droughts over avoiding false alarms, which is a defensible trade-off for an early-warning system (a missed drought is costlier than a false alarm), but the false-alarm rate is a real weakness.
- False alarms were **not** timing errors near real droughts — the median distance from a false alarm to the nearest actual drought month was **19 months**, meaning most false alarms happened in periods that were genuinely wet.
- **Threshold tuning did not fix this.** Raising the decision threshold from 0.5 up to 0.8 did not trade false alarms for precision — true positives collapsed just as fast as false positives did, indicating the model does not cleanly separate confident correct predictions from confident wrong ones. This points to a genuine modeling/calibration limitation rather than a simple decision-boundary fix.

---

## Limitations

- Single district only — findings may not generalize to other parts of Botswana with different rainfall regimes.
- Precision on unseen recent data (2024–2026) is low (~21%); the dashboard is intended as an early **signal to investigate**, not a certain forecast.
- Small sample of actual drought events in the most recent test fold limits how confidently onset-vs-ongoing drought patterns can be characterized.
- ENSO at 0–2 month lags did not help; longer lag windows (4–6 months) were not tested and remain a natural next step.

---

## Dashboard

Built with Streamlit — shows current SPI-3, the model's 3-month drought risk probability, historical SPI and rainfall charts, and a transparent summary of the model's known limitations. Deployed free on Streamlit Community Cloud.

## Tech Stack

Python, pandas, NumPy, scikit-learn, XGBoost, SciPy (gamma distribution fitting for SPI), Matplotlib, Streamlit.

## Author

Eddie — BSc Data Science student, Botho University.