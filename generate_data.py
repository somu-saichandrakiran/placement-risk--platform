# generate_data.py — stronger signal, more realistic
import pandas as pd
import numpy as np

rng = np.random.default_rng(42)
n = 1000  # double the size

df = pd.DataFrame({
    "sl_no":          range(1, n+1),
    "ssc_p":          rng.uniform(50, 95, n).round(2),
    "ssc_b":          rng.choice(["Central", "Others"], n),
    "hsc_p":          rng.uniform(45, 95, n).round(2),
    "hsc_b":          rng.choice(["Central", "Others"], n),
    "hsc_s":          rng.choice(["Commerce", "Science", "Arts"], n),
    "degree_p":       rng.uniform(50, 90, n).round(2),
    "degree_t":       rng.choice(["Sci&Tech", "Comm&Mgmt", "Others"], n),
    "workex":         rng.choice(["Yes", "No"], n, p=[0.35, 0.65]),
    "etest_p":        rng.uniform(40, 98, n).round(2),
    "specialisation": rng.choice(["Mkt&HR", "Mkt&Fin"], n),
    "mba_p":          rng.uniform(50, 90, n).round(2),
})

# Stronger, more realistic placement logic
work_exp     = (df["workex"] == "Yes").astype(float)
sci_stream   = (df["hsc_s"] == "Science").astype(float)
tech_degree  = (df["degree_t"] == "Sci&Tech").astype(float)

score = (
    (df["degree_p"] - 60) * 0.08 +
    (df["etest_p"]  - 60) * 0.06 +
    (df["mba_p"]    - 60) * 0.06 +
    (df["ssc_p"]    - 60) * 0.03 +
    (df["hsc_p"]    - 60) * 0.03 +
    work_exp               * 0.40 +
    sci_stream             * 0.20 +
    tech_degree            * 0.15 +
    rng.normal(0, 0.08, n)        # less noise
)

prob   = 1 / (1 + np.exp(-score))
placed = rng.binomial(1, prob, n)

df["status"] = np.where(placed == 1, "Placed", "Not Placed")
df["salary"] = np.where(
    placed == 1,
    (
        200000 +
        (df["degree_p"] * 3000).astype(int) +
        (df["etest_p"]  * 2000).astype(int) +
        (work_exp * 100000).astype(int) +
        rng.integers(0, 50000, n)
    ),
    np.nan
)

df.to_csv("data/raw/placements_raw.csv", index=False)
print(f"Generated {n} records")
print(f"Placement rate: {placed.mean():.1%}")
print("Saved to data/raw/placements_raw.csv")