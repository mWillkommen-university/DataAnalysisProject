import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


os.makedirs("eda_plots", exist_ok=True)
os.makedirs("eda_tables", exist_ok=True)

df = pd.read_csv("dataset-20251215_2.csv") 

df["High_impact"] = df["Accident_severity"].isin(
    ["serious", "fatal"]
)

# Base binary risk features
# -----------------------------
df["risk_night"] = df["Light_conditions"].isin(
    ["darkness", "lights"]
)

df["risk_wet"] = df["Weather_conditions"].isin(
    ["raining", "snow", "fog"]
)

df["risk_speed"] = df["Cause_of_accident"].eq("speeding")

df["risk_young_driver"] = df["Age_band_of_driver"].isin(["young","minor"])

df["risk_unlit_road"] = df["Light_conditions"].eq(
    "darkness"
)

# -----------------------------
# Define composite risk indicators
# -----------------------------
composite_risks = {
    "risk_speed_wet": ["risk_speed", "risk_wet"],
    "risk_night_rain": ["risk_night", "risk_wet"],
    "risk_young_night": ["risk_young_driver", "risk_night"],
    "risk_night_rain_unlit": ["risk_night", "risk_wet", "risk_unlit_road"],
}

# -----------------------------
# Create composite features
# -----------------------------
for name, components in composite_risks.items():
    df[name] = df[components].all(axis=1).astype(int)

# -----------------------------
# Analyze high-impact share
# -----------------------------
results = {}

for risk in composite_risks.keys():
    ct = pd.crosstab(
        df[risk],
        df["High_impact"],
        normalize="index"
    ) * 100

    ct = ct.rename(
        columns={True: "Serious or Fatal", False: "Slight"}
    )

    results[risk] = ct.round(2)

# -----------------------------
# Print results nicely
# -----------------------------
for risk, table in results.items():
    print(f"\n=== High-impact share for {risk} ===")
    print(table)
