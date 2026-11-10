import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

CLEAN_DIR = os.path.join(
    BASE_DIR, "data", "cleaned"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR, "outputs"
)

PLOT_DIR = os.path.join(
    OUTPUT_DIR, "plots"
)

REPORT_DIR = os.path.join(
    OUTPUT_DIR, "reports"
)

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


# ==========================================
# HOUSEHOLD AND APPLIANCE MAPPING
# Houses and the column name for their washing machine
# ==========================================

HOUSE_MAPPING = {
    1: "Appliance5",
    2: "Appliance2",
    3: "Appliance6",
    15: "Appliance3"
}

# Cycle detection rules
POWER_THRESHOLD = 10       # Watts — below this the machine is considered OFF
MIN_DURATION = 5           # Minimum minutes for a valid wash cycle
MAX_GAP = 2                # Maximum OFF minutes to bridge between ON periods


# ==========================================
# LOAD HOUSEHOLD DATA
# ==========================================

def load_house_data(house_number):
    """Load and resample one household's data to 1-minute intervals.

    Returns
    -------
    df : pd.DataFrame  with columns [Aggregate, Washing_Machine]
    appliance_col : str  original appliance column name from the CSV
    """

    appliance_col = HOUSE_MAPPING[house_number]

    pattern = os.path.join(
        CLEAN_DIR,
        "**",
        f"CLEAN_House{house_number}.csv"
    )

    files = glob.glob(pattern, recursive=True)

    if not files:
        raise FileNotFoundError(
            f"Cleaned House {house_number} CSV not found "
            f"under {CLEAN_DIR}."
        )

    df = pd.read_csv(files[0])

    # Validate that the required columns exist before proceeding
    required = ["Time", "Aggregate", appliance_col]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"House {house_number} CSV is missing columns: {missing_cols}. "
            f"Available columns: {df.columns.tolist()}"
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

    df[["Aggregate", appliance_col]] = df[
        ["Aggregate", appliance_col]
    ].apply(pd.to_numeric, errors="coerce")

    # Resample to one-minute average power
    df = df[["Aggregate", appliance_col]].resample("1min").mean()

    # Rename to a consistent column name so downstream code is not
    # fragile to the original appliance column name.
    df = df.rename(columns={appliance_col: "Washing_Machine"})

    return df, appliance_col


# ==========================================
# DETECT WASHING MACHINE CYCLES
# ==========================================

def detect_cycles(df):
    """Detect washing machine ON cycles from the Washing_Machine column."""

    power = df["Washing_Machine"]

    # ON/OFF status per minute
    is_on = power > POWER_THRESHOLD

    # Identify contiguous run groups
    group_id = is_on.ne(is_on.shift()).cumsum()

    runs = []

    for _, group in is_on.groupby(group_id):

        runs.append({
            "start": group.index[0],
            "end": group.index[-1],
            "state": bool(group.iloc[0]),
            "duration": len(group)
        })

    # Fill short OFF gaps between ON periods (bridge gaps <= MAX_GAP minutes)
    for i in range(1, len(runs) - 1):

        if (
            not runs[i]["state"]
            and runs[i - 1]["state"]
            and runs[i + 1]["state"]
            and runs[i]["duration"] <= MAX_GAP
        ):

            runs[i]["state"] = True

    # Extract valid cycles (must be ON and long enough)
    cycles = []

    for run in runs:

        if run["state"] and run["duration"] >= MIN_DURATION:

            cycle_data = power.loc[
                run["start"]:run["end"]
            ]

            duration = len(cycle_data)

            energy_kwh = (
                cycle_data.fillna(0).sum() * 60 / 3_600_000
            )

            cycles.append({

                "Start": run["start"],

                "End": run["end"],

                "Duration_Minutes": duration,

                "Average_Power_W": cycle_data.mean(),

                "Maximum_Power_W": cycle_data.max(),

                "Energy_kWh": energy_kwh

            })

    return pd.DataFrame(cycles)


# ==========================================
# MAIN EDA
# ==========================================

def main():

    print("\nWASHING MACHINE EDA STARTED")

    household_results = []

    all_cycles = []

    hourly_profiles = []

    for house, appliance_original in HOUSE_MAPPING.items():

        print("\n" + "=" * 50)

        print(f"ANALYSING HOUSE {house}")

        # Bug fix: use a distinct variable name for the return value so the
        # loop variable `appliance_original` is never accidentally overwritten.
        df, appliance_col = load_house_data(house)

        print("Dataset loaded:", df.shape)

        # Detect cycles using the unified "Washing_Machine" column
        cycles = detect_cycles(df)

        print("Washing machine cycles:", len(cycles))

        if not cycles.empty:

            cycles["House"] = house

            all_cycles.append(cycles)

        # Hourly consumption profile — use the renamed column "Washing_Machine"
        hourly = df[
            ["Aggregate", "Washing_Machine"]
        ].groupby(
            df.index.hour
        ).mean()

        hourly.index.name = "Hour"

        hourly["House"] = house

        hourly_profiles.append(hourly.reset_index())

        # Household summary
        summary = {

            "House": house,

            "Total_Cycles": len(cycles),

            "Average_Power_W": (
                cycles["Average_Power_W"].mean()
                if not cycles.empty else np.nan
            ),

            "Maximum_Power_W": (
                cycles["Maximum_Power_W"].max()
                if not cycles.empty else np.nan
            ),

            "Average_Duration_Minutes": (
                cycles["Duration_Minutes"].mean()
                if not cycles.empty else np.nan
            ),

            "Average_Energy_kWh": (
                cycles["Energy_kWh"].mean()
                if not cycles.empty else np.nan
            ),

            "Total_Energy_kWh": (
                cycles["Energy_kWh"].sum()
                if not cycles.empty else 0
            )

        }

        household_results.append(summary)

    # ======================================
    # SAVE RESULTS
    # ======================================

    summary_df = pd.DataFrame(household_results)

    summary_df.to_csv(
        os.path.join(
            REPORT_DIR,
            "washing_machine_household_summary.csv"
        ),
        index=False
    )

    print("\nHOUSEHOLD COMPARISON")

    print(summary_df.to_string(index=False))

    if all_cycles:

        cycles_df = pd.concat(
            all_cycles,
            ignore_index=True
        )

        cycles_df.to_csv(
            os.path.join(
                REPORT_DIR,
                "washing_machine_cycles.csv"
            ),
            index=False
        )

    hourly_df = pd.concat(
        hourly_profiles,
        ignore_index=True
    )

    hourly_df.to_csv(
        os.path.join(
            REPORT_DIR,
            "washing_machine_hourly_profile.csv"
        ),
        index=False
    )

    # ======================================
    # GRAPH 1: 24-HOUR USAGE PROFILE
    # Bug fix: use "Washing_Machine" column name (consistent),
    # not .iloc[:, 1] (fragile positional indexing).
    # Removed the dead `if False else None` block entirely.
    # ======================================

    plt.figure(figsize=(12, 6))

    for house in HOUSE_MAPPING:

        house_data = hourly_df[
            hourly_df["House"] == house
        ]

        plt.plot(
            house_data["Hour"],
            house_data["Washing_Machine"],   # named column — not positional
            marker="o",
            label=f"House {house}"
        )

    plt.title("Washing Machine Average Power by Hour")

    plt.xlabel("Hour of Day")

    plt.ylabel("Average Power (W)")

    plt.xticks(range(24))

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOT_DIR,
            "washing_machine_hourly_comparison.png"
        )
    )

    plt.close()

    # ======================================
    # GRAPH 2: HOUSEHOLD CYCLE COUNT COMPARISON
    # ======================================

    plt.figure(figsize=(10, 6))

    plt.bar(
        summary_df["House"].astype(str),
        summary_df["Total_Cycles"]
    )

    plt.title("Washing Machine Cycles by Household")

    plt.xlabel("Household")

    plt.ylabel("Number of Cycles")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOT_DIR,
            "washing_machine_cycle_comparison.png"
        )
    )

    plt.close()

    # ======================================
    # GRAPH 3: TOTAL ENERGY COMPARISON
    # ======================================

    plt.figure(figsize=(10, 6))

    plt.bar(
        summary_df["House"].astype(str),
        summary_df["Total_Energy_kWh"]
    )

    plt.title("Total Washing Machine Energy by Household")

    plt.xlabel("Household")

    plt.ylabel("Energy (kWh)")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOT_DIR,
            "washing_machine_energy_comparison.png"
        )
    )

    plt.close()

    print("\nEDA COMPLETED SUCCESSFULLY!")

    print("\nReports saved inside:", REPORT_DIR)

    print("\nGraphs saved inside:", PLOT_DIR)


if __name__ == "__main__":

    main()
