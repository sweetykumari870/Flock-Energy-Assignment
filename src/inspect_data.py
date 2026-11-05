
import os
import pandas as pd

# Project directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
CLEAN_DIR = os.path.join(BASE_DIR, "data", "cleaned")


def inspect_dataset(folder_path, dataset_name):

    print("\n" + "=" * 70)
    print(f"INSPECTING {dataset_name}")
    print("=" * 70)

    csv_files = []

    # Find CSV files inside all subfolders
    for root, dirs, files in os.walk(folder_path):

        for file in files:

            if file.lower().endswith(".csv"):

                csv_files.append(os.path.join(root, file))

    if not csv_files:

        print("No CSV files found!")
        return

    print(f"\nTotal CSV files found: {len(csv_files)}")

    for file_path in csv_files:

        print("\n" + "-" * 60)
        print("FILE:", os.path.basename(file_path))
        print("PATH:", file_path)

        try:

            df = pd.read_csv(file_path, nrows=5)

            print("\nFirst 5 rows:")
            print(df.head())

            print("\nColumns:")
            print(df.columns.tolist())

            print("\nData types:")
            print(df.dtypes)

            print("\nMissing values in sample:")
            print(df.isnull().sum())

        except Exception as e:

            print("Error reading file:", e)


def main():

    print("\nFLOCK ENERGY - REFIT DATA INSPECTION")

    inspect_dataset(RAW_DIR, "RAW REFIT DATA")

    inspect_dataset(CLEAN_DIR, "CLEANED REFIT DATA")


if __name__ == "__main__":

    main()

