
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# Project directory
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

OUTPUT_DIR = os.path.join(
    BASE_DIR, "data", "processed"
)

PLOT_DIR = os.path.join(
    BASE_DIR, "outputs", "plots"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# STEP 1: Find House 1 CSV
def find_house1():

    files = glob.glob(
        os.path.join(RAW_DIR, "**", "*.csv"),
        recursive=True
    )

    house_files = [
        f for f in files
        if "house1" in os.path.basename(f).lower()
        or "house_1" in os.path.basename(f).lower()
    ]

    if not house_files:
        raise FileNotFoundError(
            "House 1 CSV file not found!"
        )

    return house_files[0]


# STEP 2: Load Dataset
def clean_data():

    print("\n========== DATA CLEANING ==========")

    file_path = find_house1()

    df = pd.read_csv(file_path)

    print("\nOriginal dataset shape:", df.shape)

    # STEP 3: Convert timestamp
    df["Time"] = pd.to_datetime(
        df["Time"],
        errors="coerce"
    )

    # Remove invalid timestamps
    df = df.dropna(subset=["Time"])

    # STEP 4: Sort timestamps
    df = df.sort_values("Time")

    # STEP 5: Remove duplicate timestamps
    duplicates = df["Time"].duplicated().sum()

    print("Duplicate timestamps:", duplicates)

    # Average duplicate readings
    df = df.groupby(
        "Time",
        as_index=False
    ).mean(numeric_only=True)

    # STEP 6: Set timestamp as index
    df = df.set_index("Time")

    # STEP 7: Select electricity columns
    power_columns = [
        col for col in df.columns
        if col == "Aggregate"
        or col.startswith("Appliance")
    ]

    df[power_columns] = df[power_columns].apply(
        pd.to_numeric,
        errors="coerce"
    )

    # STEP 8: Resample to 1-minute intervals
    cleaned_df = df[power_columns].resample("1min").mean()

    print("\nResampling completed!")

    # STEP 9: Missing values report
    print("\nMissing values after resampling:")

    print(cleaned_df.isnull().sum())

    # STEP 10: Save cleaned dataset
    output_file = os.path.join(
        OUTPUT_DIR,
        "house1_cleaned_1min.csv"
    )

    cleaned_df.to_csv(output_file)

    print("\nCleaned dataset saved:", output_file)

    # STEP 11: Generate comparison graph
    plt.figure(figsize=(14, 5))

    df["Aggregate"].iloc[:1000].plot(
        label="Original Data",
        alpha=0.7
    )

    plt.title("House 1 - Original Aggregate Consumption")
    plt.xlabel("Time")
    plt.ylabel("Power (W)")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOT_DIR,
            "house1_original_consumption.png"
        )
    )

    plt.close()

    plt.figure(figsize=(14, 5))

    cleaned_df["Aggregate"].iloc[:1000].plot(
        label="Cleaned Data",
        color="green"
    )

    plt.title("House 1 - Cleaned Aggregate Consumption")
    plt.xlabel("Time")
    plt.ylabel("Power (W)")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOT_DIR,
            "house1_cleaned_consumption.png"
        )
    )

    plt.close()

    print("\nGraphs generated successfully!")

    print("\nOriginal rows:", len(df))
    print("Cleaned rows:", len(cleaned_df))

    print("\nDATA CLEANING COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":

    clean_data()

