import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =====================================
# PROJECT PATHS
# =====================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

REPORT_DIR = os.path.join(
    BASE_DIR, "outputs", "reports"
)

PLOT_DIR = os.path.join(
    BASE_DIR, "outputs", "plots"
)

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# =====================================
# MAIN FUNCTION
# Bug fix: all executable code is now inside main() so that importing
# this module from another script does NOT trigger evaluation automatically.
# =====================================

def main():

    # ------------------------------------------
    # STEP 1: LOAD PREDICTIONS
    # ------------------------------------------

    print("\nLOADING MODEL PREDICTIONS")

    file_path = os.path.join(
        REPORT_DIR,
        "model_predictions.csv"
    )

    df = pd.read_csv(file_path)

    actual = df["Actual"].to_numpy()

    predicted = df["Predicted"].to_numpy()

    print("Total test samples:", len(df))

    # ------------------------------------------
    # STEP 2: CALCULATE OVERALL METRICS
    # ------------------------------------------

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    r2 = r2_score(
        actual,
        predicted
    )

    print("\nOVERALL MODEL PERFORMANCE")

    print("MAE:", round(mae, 3))

    print("RMSE:", round(rmse, 3))

    print("R2 Score:", round(r2, 3))

    # ------------------------------------------
    # STEP 3: ACTIVE PERIOD EVALUATION
    # Only rows where the washing machine is actually ON (power > 10 W)
    # ------------------------------------------

    THRESHOLD = 10

    active_mask = actual > THRESHOLD

    active_actual = actual[active_mask]

    active_predicted = predicted[active_mask]

    print("\nACTIVE WASHING MACHINE READINGS")

    print("Active readings:", len(active_actual))

    print(
        "Active percentage:",
        round(active_mask.mean() * 100, 2),
        "%"
    )

    if len(active_actual) > 0:

        active_mae = mean_absolute_error(
            active_actual,
            active_predicted
        )

        active_rmse = np.sqrt(
            mean_squared_error(
                active_actual,
                active_predicted
            )
        )

        print("Active MAE:", round(active_mae, 3))

        print("Active RMSE:", round(active_rmse, 3))

    else:

        active_mae = np.nan
        active_rmse = np.nan

        print("No active readings found!")

    # ------------------------------------------
    # STEP 4: BASELINE COMPARISON
    # ------------------------------------------

    baseline = np.zeros(len(actual))

    baseline_mae = mean_absolute_error(
        actual,
        baseline
    )

    baseline_rmse = np.sqrt(
        mean_squared_error(
            actual,
            baseline
        )
    )

    print("\nBASELINE COMPARISON")

    print("Baseline MAE:", round(baseline_mae, 3))

    print("Model MAE:", round(mae, 3))

    if baseline_mae > 0:

        improvement = (
            (baseline_mae - mae) / baseline_mae
        ) * 100

        print(
            "MAE improvement over baseline:",
            round(improvement, 2),
            "%"
        )

    # ------------------------------------------
    # STEP 5: ERROR ANALYSIS
    # ------------------------------------------

    df["Absolute_Error"] = np.abs(
        actual - predicted
    )

    df["Error"] = actual - predicted

    df["Actual_State"] = np.where(
        actual > THRESHOLD,
        "ON",
        "OFF"
    )

    df["Predicted_State"] = np.where(
        predicted > THRESHOLD,
        "ON",
        "OFF"
    )

    df["State_Mismatch"] = (
        df["Actual_State"] != df["Predicted_State"]
    )

    print("\nERROR ANALYSIS")

    print(
        "Average absolute error:",
        round(df["Absolute_Error"].mean(), 3)
    )

    print(
        "Maximum absolute error:",
        round(df["Absolute_Error"].max(), 3)
    )

    print(
        "State mismatches:",
        df["State_Mismatch"].sum()
    )

    # ------------------------------------------
    # STEP 6: IDENTIFY FAILURE CASES
    # ------------------------------------------

    worst_cases = df.nlargest(
        20,
        "Absolute_Error"
    )

    worst_cases.to_csv(
        os.path.join(
            REPORT_DIR,
            "worst_prediction_cases.csv"
        ),
        index=False
    )

    print("\nTOP 20 WORST PREDICTIONS")

    print(
        worst_cases[
            ["Actual", "Predicted", "Absolute_Error"]
        ].head(10)
    )

    # ------------------------------------------
    # STEP 7: SAVE FINAL METRICS
    # ------------------------------------------

    metrics = pd.DataFrame({

        "Metric": [
            "Overall MAE",
            "Overall RMSE",
            "R2 Score",
            "Active MAE",
            "Active RMSE",
            "Baseline MAE",
            "Baseline RMSE"
        ],

        "Value": [
            mae,
            rmse,
            r2,
            active_mae,
            active_rmse,
            baseline_mae,
            baseline_rmse
        ]

    })

    metrics.to_csv(
        os.path.join(
            REPORT_DIR,
            "detailed_evaluation.csv"
        ),
        index=False
    )

    # ------------------------------------------
    # STEP 8: ERROR DISTRIBUTION GRAPH
    # ------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.hist(
        df["Absolute_Error"],
        bins=50
    )

    plt.title("Distribution of Prediction Errors")

    plt.xlabel("Absolute Error (Watts)")

    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOT_DIR,
            "error_distribution.png"
        )
    )

    plt.close()

    # ------------------------------------------
    # STEP 9: ACTUAL VS PREDICTED SCATTER
    # ------------------------------------------

    plt.figure(figsize=(8, 8))

    plt.scatter(
        actual,
        predicted,
        alpha=0.3
    )

    limit = max(
        np.max(actual),
        np.max(predicted),
        1
    )

    plt.plot(
        [0, limit],
        [0, limit],
        linestyle="--"
    )

    plt.xlabel("Actual Power (W)")

    plt.ylabel("Predicted Power (W)")

    plt.title("Actual vs Predicted Power")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOT_DIR,
            "actual_predicted_scatter.png"
        )
    )

    plt.close()

    # ------------------------------------------
    # FINAL MESSAGE
    # ------------------------------------------

    print("\n================================")

    print("MODEL EVALUATION COMPLETED!")

    print("Reports saved:", REPORT_DIR)

    print("Graphs saved:", PLOT_DIR)

    print("================================")


if __name__ == "__main__":

    main()
