import numpy as np

path = "/Users/marcelwillkommen/Coding/DataForAI/DataAnalysisProject/DataAnalysisProject/data/raw/RTA Dataset.csv"
path_new = "/Users/marcelwillkommen/Coding/DataForAI/DataAnalysisProject/DataAnalysisProject/data/processed/dataset-20251130.csv"
expected_cols = 32

#Load the data from the .csv
try:
    dataList = np.loadtxt(path, delimiter=",")
    print("Loaded successfully (all rows have 32 columns).")
except ValueError as e:
    print("ValueError occurred while loading CSV:")
    print(e)
    # Load to inspect rows
    lines = np.genfromtxt(path, delimiter="\n", dtype=str)
    rows = np.char.split(lines, sep=",")

    # Count rows with too many columns
    too_long_count = np.sum([len(r) > expected_cols for r in rows])

    # Filter valide rows
    clean_rows = [r for r in rows if len(r) == expected_cols]

    print("Total rows:", len(rows))
    print("Good rows:", len(clean_rows))
    print("Skipped rows:", len(rows) - len(clean_rows))

    dataList = np.array(clean_rows, dtype=str)

# Create new csv
np.savetxt(path_new, dataList, delimiter=",", fmt="%s")

