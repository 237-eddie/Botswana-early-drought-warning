import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

spi_df = pd.read_csv("spi3_history.csv", parse_dates=["date"]).sort_values("date")

# month-end SPI-3 = monthly drought value
monthly = spi_df.set_index("date")["spi"].resample("ME").last().dropna()
drought = (monthly <= -1).astype(int)
overall_rate = drought.mean()
print(f"Base rate: drought in {overall_rate:.1%} of months\n")

for lead in [1, 2, 3]:
    y_true = drought.shift(-lead).dropna()

    # Baseline 1: persistence — "next month = this month"
    y_pers = drought.loc[y_true.index]

    # Baseline 2: climatology — "drought if this calendar month is historically
    # more drought-prone than average"
    month_risk = drought.groupby(drought.index.month).mean()
    y_clim = (pd.Series(y_true.index.month, index=y_true.index)
                .map(month_risk) > overall_rate).astype(int)

    print(f"--- {lead}-month lead ---")
    print(f"  persistence F1: {f1_score(y_true, y_pers):.3f}")
    print(f"  climatology F1: {f1_score(y_true, y_clim):.3f}")
    print(f"  (always-predict-no-drought F1: {f1_score(y_true, [0]*len(y_true)):.3f})")
    