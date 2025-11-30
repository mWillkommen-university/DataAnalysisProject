import numpy as np


path = "/Users/marcelwillkommen/Coding/DataForAI/DataAnalysisProject/DataAnalysisProject/data/processed/dataset-20251130.csv"
# Load the first row (headers) separately
with open(path, "r") as f:
    headers = f.readline().strip().split(",")
print("Headers:", headers)

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
                    'Light_conditions', 'Weather_conditions' 'Cause_of_accident']

# Now can check the focus features - example area
col_name = "Area_accident_occured" 
col_data = accident_dict[col_name] 

# Create a boolean mask for missing values
missing_mask = (col_data == "NULL") | (col_data == "")

# Count how many rows have missing values
missing_count = np.sum(missing_mask)
print(f"Rows with missing values in column '{col_name}':", missing_count)