import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -----------------------------
# Setup
# -----------------------------
sns.set(style="whitegrid")

df = pd.read_csv("dataset-20251215_2.csv")

os.makedirs("eda_plots", exist_ok=True)

def save_plot(filename):
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches="tight")
    plt.close()

# -----------------------------
# 1️⃣ Accident Severity (Counts)
# -----------------------------
plt.figure(figsize=(6,4))
df["Accident_severity"].value_counts().plot(kind="bar")
plt.title("Distribution of Accident Severity")
plt.xlabel("Accident Severity")
plt.ylabel("Count")
save_plot("01_accident_severity_distribution.png")

# -----------------------------
# 2️⃣ Accidents by Hour (Counts)
# -----------------------------
plt.figure(figsize=(7,4))
df["# Time"].value_counts().sort_index().plot(kind="bar")
plt.title("Accidents by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Count")
save_plot("02_accidents_by_hour.png")

# -----------------------------
# 3️⃣ Accidents by Day of Week (Counts)
# -----------------------------
plt.figure(figsize=(7,4))
df["Day_of_week"].value_counts().plot(kind="bar")
plt.title("Accidents by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Count")
save_plot("03_accidents_by_day.png")

# -----------------------------
# 4️⃣ Age Band × Severity (Normalized)
# -----------------------------
ct_age = pd.crosstab(
    df["Age_band_of_driver"],
    df["Accident_severity"],
    normalize="index"
)

ct_age.plot(kind="bar", stacked=True, figsize=(8,4))
plt.title("Accident Severity Distribution by Driver Age Group")
plt.xlabel("Age Band of Driver")
plt.ylabel("Proportion")
plt.legend(title="Accident Severity")
save_plot("04_age_band_vs_severity.png")

# -----------------------------
# 5️⃣ Weather × Severity (Normalized)
# -----------------------------
ct_weather = pd.crosstab(
    df["Weather_conditions"],
    df["Accident_severity"],
    normalize="index"
)

ct_weather.plot(kind="bar", stacked=True, figsize=(10,4))
plt.title("Accident Severity Distribution by Weather Conditions")
plt.xlabel("Weather Conditions")
plt.ylabel("Proportion")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Accident Severity")
save_plot("05_weather_vs_severity.png")

# -----------------------------
# 6️⃣ Light × Severity (Normalized)
# -----------------------------
ct_light = pd.crosstab(
    df["Light_conditions"],
    df["Accident_severity"],
    normalize="index"
)

ct_light.plot(kind="bar", stacked=True, figsize=(7,4))
plt.title("Accident Severity Distribution by Light Conditions")
plt.xlabel("Light Conditions")
plt.ylabel("Proportion")
plt.legend(title="Accident Severity")
save_plot("06_light_vs_severity.png")

# -----------------------------
# 7️⃣ Collision Type × Severity (Normalized, Top 5)
# -----------------------------
top_collisions = df["Type_of_collision"].value_counts().nlargest(5).index

ct_collision = pd.crosstab(
    df[df["Type_of_collision"].isin(top_collisions)]["Type_of_collision"],
    df["Accident_severity"],
    normalize="index"
)

ct_collision.plot(kind="bar", stacked=True, figsize=(10,4))
plt.title("Accident Severity Distribution by Collision Type")
plt.xlabel("Type of Collision")
plt.ylabel("Proportion")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Accident Severity")
save_plot("07_collision_type_vs_severity.png")

# -----------------------------
# 8️⃣ Heatmap: Age × Severity
# -----------------------------
plt.figure(figsize=(6,4))
sns.heatmap(ct_age, annot=True, fmt=".2f", cmap="Blues")
plt.title("Normalized Accident Severity by Driver Age Group")
plt.xlabel("Accident Severity")
plt.ylabel("Age Band of Driver")
save_plot("08_heatmap_age_vs_severity.png")

print(f"EDA plots saved to folder: {OUTPUT_DIR}")


plt.figure(figsize=(7,4))
df["Age_band_of_driver"].value_counts().plot(kind="bar")
plt.title("Distribution of Driver Age Groups")
plt.xlabel("Age Band of Driver")
plt.ylabel("Count")
save_plot("09_distribution_driver_age.png")

plt.figure(figsize=(7,4))
df["Driving_experience"].value_counts().plot(kind="bar")
plt.title("Distribution of Driving Experience")
plt.xlabel("Driving Experience")
plt.ylabel("Count")
save_plot("10_distribution_driving_experience.png")

top_vehicles = df["Type_of_vehicle"].value_counts().nlargest(6)

plt.figure(figsize=(8,4))
top_vehicles.plot(kind="bar")
plt.title("Most Common Vehicle Types Involved in Accidents")
plt.xlabel("Type of Vehicle")
plt.ylabel("Count")
save_plot("11_distribution_vehicle_type.png")

plt.figure(figsize=(8,4))
df["Weather_conditions"].value_counts().plot(kind="bar")
plt.title("Distribution of Weather Conditions")
plt.xlabel("Weather Conditions")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
save_plot("12_distribution_weather_conditions.png")

plt.figure(figsize=(7,4))
df["Road_surface_conditions"].value_counts().plot(kind="bar")
plt.title("Distribution of Road Surface Conditions")
plt.xlabel("Road Surface Conditions")
plt.ylabel("Count")
save_plot("14_distribution_road_surface.png")

top_causes = df["Cause_of_accident"].value_counts().nlargest(6)

plt.figure(figsize=(9,4))
top_causes.plot(kind="bar")
plt.title("Most Common Causes of Accidents")
plt.xlabel("Cause of Accident")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
save_plot("15_distribution_cause_of_accident.png")

