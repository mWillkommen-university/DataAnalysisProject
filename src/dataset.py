import numpy as np

path = "/Users/marcelwillkommen/Coding/DataForAI/DataAnalysisProject/DataAnalysisProject/data/raw/RTA Dataset.csv"
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