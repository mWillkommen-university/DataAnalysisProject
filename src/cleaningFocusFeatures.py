import numpy as np


path = "/Users/marcelwillkommen/Coding/DataForAI/DataAnalysisProject/DataAnalysisProject/data/processed/dataset-20251130.csv"
# Load the first row (headers) separately
with open(path, "r") as f:
    headers = f.readline().strip().split(",")

# Skip the first row (headers) using skiprows=1
data = np.genfromtxt(path, delimiter=",", dtype=str, skip_header=1)

# Convert to list of rows
data_list = data.tolist()
print("Number of rows loaded:", len(data_list))

# Create a dictionary: key = column name, value = column data
accident_dict = {header: data[:, i] for i, header in enumerate(headers)}

selected_headers= ['Time', 'Day_of_week', 'Age_band_of_driver', 'Vehicle_driver_relation', 'Driving_experience',
                    'Type_of_vehicle', 'Owner_of_vehicle', 'Area_accident_occured', 'Lanes_or_Medians',
                    'Road_allignment', 'Types_of_Junction', 'Road_surface_type', 'Road_surface_conditions',
                    'Light_conditions', 'Weather_conditions', 'Cause_of_accident']

# Prepare an empty NumPy array to store the counts
missing_counts = np.zeros(len(selected_headers), dtype=int)

# Loop through the selected headers
for i, header in enumerate(selected_headers):
    col_data = accident_dict[header]

    # Missing check: NULL or empty
    missing_mask = (col_data == "NULL") | (col_data == "")

    # Store count in the array
    missing_counts[i] = np.sum(missing_mask)

print(missing_counts)
mean_missing = np.mean(missing_counts)
print("Mean of missing values:", mean_missing)