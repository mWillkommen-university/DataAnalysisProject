import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


os.makedirs("eda_plots", exist_ok=True)

df = pd.read_csv("dataset-20251215_2.csv")

# Crosstable: Cause of accident vs Accident severity (row-normalized)
ct_cause_sev = pd.crosstab(
    df["Cause_of_accident"],
    df["Accident_severity"],
    normalize="index"
) * 100

ct_cause_sev = ct_cause_sev.round(2)

ct_cause_sev.head()
print(ct_cause_sev)

ct_cause_sev["High_severity_%"] = (
    ct_cause_sev.get("fatal", 0) +
    ct_cause_sev.get("serious", 0)
)

ranked_causes = ct_cause_sev.sort_values(
    by="High_severity_%",
    ascending=False
)

ranked_causes.head(10)


top10 = ranked_causes.head(10)

plt.figure(figsize=(10, 6))
top10["High_severity_%"].plot(kind="bar")
plt.ylabel("Share of Fatal + Serious Accidents (%)")
plt.xlabel("Cause of Accident")
plt.title("Top 10 Accident Causes by High-Severity Share")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("eda_plots/cause_vs_severity_ranked.png", dpi=300)
plt.show()


# Crosstable: Cause of accident vs Accident severity (row-normalized)
ct_cause_sev = pd.crosstab(
    df["Cause_of_accident"],
    df["Accident_severity"],
    normalize="index"
) * 100

ct_cause_sev = ct_cause_sev.round(2)


fatal_ranked = ct_cause_sev.sort_values(
    by="fatal",
    ascending=False
)

# Top 10 causes with highest fatal risks
top_fatal = fatal_ranked.head(10)

plt.figure(figsize=(10, 6))
top_fatal["fatal"].plot(kind="bar")
plt.ylabel("Share of Fatal Accidents (%)")
plt.xlabel("Cause of Accident")
plt.title("Top 10 Accident Causes by Fatal Accident Share")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("eda_plots/cause_vs_fatal_share.png", dpi=300)
plt.show()

df["High_impact"] = df["Accident_severity"].isin(
    ["serious", "fatal"]
)
critical_causes = ["overloading", "lane violation", "rollover", "speeding","overtaking"]
df["High_impact"] = df["High_impact"].map(
    {True: "Serious or Fatal", False: "Slight"}
)

df_critical = df[df["Cause_of_accident"].isin(critical_causes)]

ct_weather_hi = pd.crosstab(
    [df_critical["Cause_of_accident"], df_critical["Weather_conditions"]],
    df_critical["High_impact"],
    normalize="index"
) * 100

ct_weather_hi = ct_weather_hi.reset_index()

plt.figure(figsize=(12, 6))
sns.barplot(
    data=ct_weather_hi,
    x="Cause_of_accident",
    y="Serious or Fatal",
    hue="Weather_conditions"
)

plt.ylabel("Share of Serious or Fatal Accidents (%)")
plt.xlabel("Cause of Accident")
plt.title("High-Impact Accidents by Cause and Weather Conditions")
plt.legend(title="Weather", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig("eda_plots/high_impact_causes_weather.png", dpi=300)
plt.show()

ct_road_hi = pd.crosstab(
    [df_critical["Cause_of_accident"], df_critical["Road_surface_conditions"]],
    df_critical["High_impact"],
    normalize="index"
) * 100

ct_road_hi = ct_road_hi.reset_index()

plt.figure(figsize=(12, 6))
sns.barplot(
    data=ct_road_hi,
    x="Cause_of_accident",
    y="Serious or Fatal",
    hue="Road_surface_conditions"
)

plt.ylabel("Share of Serious or Fatal Accidents (%)")
plt.xlabel("Cause of Accident")
plt.title("High-Impact Accidents by Cause and Road Surface Conditions")
plt.legend(title="Road Surface", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig("eda_plots/high_impact_causes_road_surface.png", dpi=300)
plt.show()


