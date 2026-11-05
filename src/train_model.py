import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "nilm_features.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")

REPORT_DIR = os.path.join(
    BASE_DIR, "outputs", "reports"
)

PLOT_DIR = os.path.join(
    BASE_DIR, "outputs", "plots"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# ==========================================
# STEP 1: LOAD DATASET
# ==========================================

print("\nLOADING FEATURE DATASET")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

print(df.head())


# ==========================================
# STEP 2: DEFINE FEATURES AND TARGET
# ==========================================

TARGET = "Washing_Machine"

FEATURES = [
    "Aggregate",
    "Hour",
    "DayOfWeek",
    "Hour_Sin",
    "Hour_Cos",
    "Aggregate_Lag1",
    "Aggregate_Lag5",
    "Aggregate_Lag15",
    "Rolling_Mean_5",
    "Rolling_Mean_15",
    "Rolling_Std_15",
    "Rolling_Max_15",
    "Rolling_Min_15"
]

# Check required columns

required_columns = FEATURES + [
    TARGET,
    "House"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )

df = df.dropna(
    subset=required_columns
)

df["House"] = pd.to_numeric(
    df["House"]
)


# ==========================================
# STEP 3: SPLIT BY HOUSEHOLD
# ==========================================

train_houses = [1, 2, 3]

test_house = 15

train_df = df[
    df["House"].isin(train_houses)
].copy()

test_df = df[
    df["House"] == test_house
].copy()

if train_df.empty or test_df.empty:
    raise ValueError(
        "Training or testing data is empty. "
        "Check household IDs."
    )

print("\nTRAINING DATA:", train_df.shape)

print("TESTING DATA:", test_df.shape)

print("Training households:", train_houses)

print("Testing household:", test_house)


# ==========================================
# STEP 4: PREPARE X AND Y
# ==========================================

X_train = train_df[FEATURES]

y_train = train_df[TARGET]

X_test = test_df[FEATURES]

y_test = test_df[TARGET]


# ==========================================
# STEP 5: LIMIT TRAINING SIZE
# ==========================================

# Sampling makes training faster for large REFIT data.

MAX_TRAIN_ROWS = 150000

if len(X_train) > MAX_TRAIN_ROWS:

    sampled_indices = train_df.sample(
        n=MAX_TRAIN_ROWS,
        random_state=42
    ).index

    X_train = train_df.loc[
        sampled_indices, FEATURES
    ]

    y_train = train_df.loc[
        sampled_indices, TARGET
    ]


# ==========================================
# STEP 6: BASELINE MODEL
# ==========================================

print("\nCALCULATING BASELINE")

# Always predict zero washing-machine power.

baseline_predictions = np.zeros(
    len(y_test)
)

baseline_mae = mean_absolute_error(
    y_test,
    baseline_predictions
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        baseline_predictions
    )
)

print("Baseline MAE:", baseline_mae)

print("Baseline RMSE:", baseline_rmse)


# ==========================================
# STEP 7: TRAIN RANDOM FOREST
# ==========================================

print("\nTRAINING RANDOM FOREST MODEL")

model = RandomForestRegressor(

    n_estimators=150,

    max_depth=20,

    min_samples_split=5,

    min_samples_leaf=2,

    random_state=42,

    n_jobs=-1

)

model.fit(
    X_train,
    y_train
)

print("\nMODEL TRAINING COMPLETED!")


# ==========================================
# STEP 8: MAKE PREDICTIONS
# ==========================================

print("\nGENERATING PREDICTIONS")

predictions = model.predict(X_test)

# Power cannot be negative.

predictions = np.maximum(
    predictions,
    0
)


# ==========================================
# STEP 9: MODEL EVALUATION
# ==========================================

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)

print("\n========== MODEL RESULTS ==========")

print("Baseline MAE:", baseline_mae)

print("Baseline RMSE:", baseline_rmse)

print("----------------------------------")

print("Random Forest MAE:", mae)

print("Random Forest RMSE:", rmse)

print("Random Forest R2:", r2)


# ==========================================
# STEP 10: SAVE EVALUATION REPORT
# ==========================================

results = pd.DataFrame({

    "Model": [
        "Always-Off Baseline",
        "Random Forest"
    ],

    "MAE": [
        baseline_mae,
        mae
    ],

    "RMSE": [
        baseline_rmse,
        rmse
    ],

    "R2": [
        np.nan,
        r2
    ]

})

results.to_csv(

    os.path.join(
        REPORT_DIR,
        "model_evaluation.csv"
    ),

    index=False

)

print("\nEvaluation report saved!")


# ==========================================
# STEP 11: SAVE PREDICTIONS
# ==========================================

prediction_df = pd.DataFrame({

    "Actual": y_test.to_numpy(),

    "Predicted": predictions

})

prediction_df.to_csv(

    os.path.join(
        REPORT_DIR,
        "model_predictions.csv"
    ),

    index=False

)


# ==========================================
# STEP 12: ACTUAL VS PREDICTED GRAPH
# ==========================================

plt.figure(figsize=(15, 6))

plot_size = min(
    1440,
    len(test_df)
)

plt.plot(

    y_test.iloc[:plot_size].to_numpy(),

    label="Actual Washing Machine Power"

)

plt.plot(

    predictions[:plot_size],

    label="Predicted Washing Machine Power"

)

plt.title(
    "Actual vs Predicted Washing Machine Consumption - House 15"
)

plt.xlabel("Minute")

plt.ylabel("Power Consumption (W)")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(

    os.path.join(
        PLOT_DIR,
        "actual_vs_predicted.png"
    )

)

plt.close()


# ==========================================
# STEP 13: FEATURE IMPORTANCE
# ==========================================

importance = pd.DataFrame({

    "Feature": FEATURES,

    "Importance": model.feature_importances_

})

importance = importance.sort_values(

    by="Importance",

    ascending=False

)

importance.to_csv(

    os.path.join(
        REPORT_DIR,
        "feature_importance.csv"
    ),

    index=False

)

plt.figure(figsize=(10, 7))

plt.barh(

    importance["Feature"],

    importance["Importance"]

)

plt.xlabel("Importance")

plt.title("Random Forest Feature Importance")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(

    os.path.join(
        PLOT_DIR,
        "feature_importance.png"
    )

)

plt.close()


# ==========================================
# STEP 14: SAVE TRAINED MODEL
# ==========================================

joblib.dump(

    model,

    os.path.join(
        MODEL_DIR,
        "washing_machine_model.pkl"
    )

)

print("\nMODEL SAVED SUCCESSFULLY!")

print("\nALL TASKS COMPLETED!")