import os
import glob
import pandas as pd
import numpy as np

# =====================================
# PROJECT PATHS
# =====================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

CLEAN_DIR = os.path.join(
    BASE_DIR, "data", "cleaned"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR, "data", "processed"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================================
# HOUSEHOLD MAPPING
# =====================================

HOUSE_MAPPING = {
    1: "Appliance5",
    2: "Appliance2",
    3: "Appliance6",
    15: "Appliance3"
}


# =====================================
# LOAD HOUSEHOLD DATA
# =====================================

def load_house_data(house, appliance):

    pattern = os.path.join(
        CLEAN_DIR,
        "**",
        f"CLEAN_House{house}.csv"
    )

    files = glob.glob(
        pattern,
        recursive=True
    )

    if not files:
        raise FileNotFoundError(
            f"House {house} not found"
        )

    df = pd.read_csv(
        files[0],
        usecols=["Time", "Aggregate", appliance]
    )

    df["Time"] = pd.to_datetime(
        df["Time"],
        errors="coerce"
    )

    df = df.dropna(subset=["Time"])

    df = df.sort_values("Time")

    df = df.drop_duplicates(
        subset="Time",
        keep="last"
    )

    df = df.set_index("Time")

    df = df.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Resample to one-minute resolution
    df = df.resample("1min").mean()

    df = df.rename(
        columns={
            appliance: "Washing_Machine"
        }
    )

    return df


# =====================================
# FEATURE ENGINEERING
# =====================================

def create_features(df, house):

    data = df.copy()

    # Time-based features
    data["Hour"] = data.index.hour

    data["DayOfWeek"] = data.index.dayofweek

    # Cyclical time features
    data["Hour_Sin"] = np.sin(
        2 * np.pi * data["Hour"] / 24
    )

    data["Hour_Cos"] = np.cos(
        2 * np.pi * data["Hour"] / 24
    )

    # Lag features
    data["Aggregate_Lag1"] = (
        data["Aggregate"].shift(1)
    )

    data["Aggregate_Lag5"] = (
        data["Aggregate"].shift(5)
    )

    data["Aggregate_Lag15"] = (
        data["Aggregate"].shift(15)
    )

    # Rolling statistics
    data["Rolling_Mean_5"] = (
        data["Aggregate"]
        .rolling(window=5, min_periods=5)
        .mean()
    )

    data["Rolling_Mean_15"] = (
        data["Aggregate"]
        .rolling(window=15, min_periods=15)
        .mean()
    )

    data["Rolling_Std_15"] = (
        data["Aggregate"]
        .rolling(window=15, min_periods=15)
        .std()
    )

    data["Rolling_Max_15"] = (
        data["Aggregate"]
        .rolling(window=15, min_periods=15)
        .max()
    )

    data["Rolling_Min_15"] = (
        data["Aggregate"]
        .rolling(window=15, min_periods=15)
        .min()
    )

    # Add household identifier
    data["House"] = house

    # Remove rows with missing required values
    data = data.dropna()

    return data


# =====================================
# MAIN FUNCTION
# =====================================

def main():

    print("\nFEATURE ENGINEERING STARTED")

    all_data = []

    for house, appliance in HOUSE_MAPPING.items():

        print("\nProcessing House:", house)

        df = load_house_data(
            house,
            appliance
        )

        print("Original shape:", df.shape)

        features = create_features(
            df,
            house
        )

        print("Feature shape:", features.shape)

        all_data.append(features)

    # Combine households
    final_data = pd.concat(
        all_data,
        axis=0
    )

    # Save final dataset
    output_file = os.path.join(
        OUTPUT_DIR,
        "nilm_features.csv"
    )

    final_data.to_csv(output_file)

    print("\nFINAL DATASET CREATED")

    print("Total rows:", len(final_data))

    print("Total columns:", len(final_data.columns))

    print("\nAvailable features:")

    print(final_data.columns.tolist())

    print("\nHousehold distribution:")

    print(final_data["House"].value_counts())

    print("\nSaved successfully:", output_file)

    print("\nFEATURE ENGINEERING COMPLETED!")


if __name__ == "__main__":

    main()