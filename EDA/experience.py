import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("eda_plots", exist_ok=True)

df = pd.read_csv("dataset-20251215_2.csv")

ct_exp_weather_sev = (
    pd.crosstab(
        [df["Driving_experience"], df["Weather_conditions"]],
        df["Accident_severity"],
        normalize="index"
    ) * 100
).round(2)

ct_exp_weather_sev.head()


ct_exp_weather_sev.to_csv(
    "eda_tables/experience_weather_severity_percent.csv"
)

high_impact = ct_exp_weather_sev[["serious", "fatal"]].sum(axis=1)

high_impact = high_impact.reset_index()
high_impact.columns = [
    "Driving_experience",
    "Weather_conditions",
    "Serious_or_Fatal_%"
]


pivot_heatmap = high_impact.pivot(
    index="Driving_experience",
    columns="Weather_conditions",
    values="Serious_or_Fatal_%"
)

plt.figure(figsize=(10,6))
sns.heatmap(
    pivot_heatmap,
    annot=True,
    fmt=".1f",
    cmap="Reds"
)

plt.title("Share of Serious or Fatal Accidents (%)\nby Driving Experience and Weather")
plt.xlabel("Weather Conditions")
plt.ylabel("Driving Experience")
plt.tight_layout()

plt.savefig("eda_plots/experience_weather_high_impact_heatmap.png")
plt.close()


ct_exp_light_sev = (
    pd.crosstab(
        [df["Driving_experience"], df["Light_conditions"]],
        df["Accident_severity"],
        normalize="index"
    ) * 100
).round(2)

ct_exp_light_sev.to_csv(
    "eda_tables/experience_light_severity_percent.csv"
)


high_impact_light = (
    ct_exp_light_sev[["serious", "fatal"]]
    .sum(axis=1)
    .reset_index()
)

high_impact_light.columns = [
    "Driving_experience",
    "Light_conditions",
    "Serious_or_Fatal_%"
]


pivot_light = high_impact_light.pivot(
    index="Driving_experience",
    columns="Light_conditions",
    values="Serious_or_Fatal_%"
)

plt.figure(figsize=(9,6))
sns.heatmap(
    pivot_light,
    annot=True,
    fmt=".1f",
    cmap="Oranges"
)

plt.title("High-Impact Accidents (%)\nby Driving Experience and Light Conditions")
plt.xlabel("Light Conditions")
plt.ylabel("Driving Experience")
plt.tight_layout()

plt.savefig("eda_plots/experience_light_high_impact_heatmap.png")
plt.close()

