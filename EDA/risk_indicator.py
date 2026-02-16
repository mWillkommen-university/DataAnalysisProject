import pandas as pd
import os


os.makedirs("eda_plots", exist_ok=True)

df = pd.read_csv("dataset-20251215_2.csv") 

df["High_impact"] = df["Accident_severity"].isin(
    ["serious", "fatal"]
)

# Base binary risk features
# -----------------------------
df["risk_night"] = df["Light_conditions"].isin(
    ["darkness"]
)

df["risk_wet"] = df["Weather_conditions"].isin(
    ["raining", "snow", "fog"]
)

df["risk_speed"] = df["Cause_of_accident"].eq("speeding")

df["risk_young_driver"] = df["Age_band_of_driver"].isin(["young","minor"])

df["risk_unexperienced"] = df["Driving_experience"].isin(["junior","novice"])

df["risk_cause"] = df["Cause_of_accident"].isin(["overloading"])

df["risk_wet_road"] = df["Road_surface_conditions"].eq("wet")

# -----------------------------
# Define composite risk indicators
# -----------------------------
composite_risks = {
    "risk_speed_wet": ["risk_speed", "risk_wet"],
    "risk_night_rain": ["risk_night", "risk_wet"],
    "risk_young_night": ["risk_young_driver", "risk_night"],
    "risk_darkness_unexperienced":["risk_night","risk_unexperienced"],
    "risk_causes_wet":["risk_cause", "risk_wet_road"]
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
