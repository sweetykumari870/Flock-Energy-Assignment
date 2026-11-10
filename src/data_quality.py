
import os
import glob
import pandas as pd
import numpy as np

# Project directory
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

REPORT_DIR = os.path.join(
    BASE_DIR, "outputs", "reports"
)

os.makedirs(REPORT_DIR, exist_ok=True)


def find_house1_parts():
    """Find all CSV files for House 1 (Part1, Part2, etc.)."""

    files = glob.glob(
        os.path.join(RAW_DIR, "**", "*.csv"),
        recursive=True
    )

    print("\nAll CSV files found:")

    for file in files:
        print(os.path.basename(file))

    # Match both RAW_House1_Part1 and RAW_House1_Part2 patterns
    house1_files = sorted([
        file for file in files
        if "house1" in os.path.basename(file).lower()
        or "house_1" in os.path.basename(file).lower()
    ])

    if not house1_files:
        raise FileNotFoundError(
            "House 1 CSV files not found. "
            "Check that RAW_House1_Part1.csv and RAW_House1_Part2.csv "
            "exist under data/raw/."
        )

    return house1_files


def analyse_data():

    print("\nFLOCK ENERGY - DATA QUALITY ANALYSIS")

    # Bug fix: load ALL parts of House 1 and concatenate
    house_files = find_house1_parts()

    print(f"\nLoading {len(house_files)} file(s) for House 1:")
    for f in house_files:
        print("  -", os.path.basename(f))

    parts = [pd.read_csv(f) for f in house_files]
    df = pd.concat(parts, ignore_index=True)

    print("\nDataset loaded successfully!")

    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    # 1. Timestamp validation
    df["Time"] = pd.to_datetime(
        df["Time"], errors="coerce"
    )

    invalid_timestamps = df["Time"].isna().sum()

    # 2. Timestamp ordering
    df = df.sort_values("Time")
    is_sorted = df["Time"].is_monotonic_increasing

    # 3. Duplicate timestamps
    duplicate_timestamps = df["Time"].duplicated().sum()

    # 4. Missing values
    missing_values = df.isnull().sum()

    # 5. Time interval analysis
    valid_times = df["Time"].dropna().sort_values()

    intervals = valid_times.diff().dt.total_seconds()

    interval_counts = intervals.value_counts().head(10)

    # 6. Numerical statistics
    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    statistics = df[numeric_columns].describe().T

    # 7. Negative power readings
    power_columns = [
        col for col in numeric_columns
        if col == "Aggregate" or col.startswith("Appliance")
    ]

    negative_values = {}

    for col in power_columns:

        negative_values[col] = int(
            (df[col] < 0).sum()
        )

    # 8. Extreme values using IQR
    outlier_counts = {}

    for col in power_columns:

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outlier_counts[col] = int(
            ((df[col] < lower) | (df[col] > upper)).sum()
        )

    # 9. Create report
    report = {
        "Total Rows": len(df),
        "Total Columns": len(df.columns),
        "Invalid Timestamps": invalid_timestamps,
        "Timestamps Sorted": is_sorted,
        "Duplicate Timestamps": duplicate_timestamps,
    }

    print("\n========== DATA QUALITY REPORT ==========")

    for key, value in report.items():
        print(f"{key}: {value}")

    print("\nMissing Values:")
    print(missing_values)

    print("\nMost Common Time Intervals (seconds):")
    print(interval_counts)

    print("\nNegative Power Readings:")
    print(negative_values)

    print("\nPotential Outliers (IQR method):")
    print(outlier_counts)

    print("\nNumerical Statistics:")
    print(statistics)

    # Save reports
    missing_values.to_csv(
        os.path.join(
            REPORT_DIR,
            "missing_values.csv"
        )
    )

    statistics.to_csv(
        os.path.join(
            REPORT_DIR,
            "numerical_statistics.csv"
        )
    )

    pd.DataFrame({
        "Metric": list(report.keys()),
        "Value": list(report.values())
    }).to_csv(
        os.path.join(
            REPORT_DIR,
            "data_quality_summary.csv"
        ),
        index=False
    )

    print("\nReports saved successfully!")


if __name__ == "__main__":

    analyse_data()
