# Flock Energy Assignment

## Household Electricity Consumption Analysis and Washing Machine Energy Prediction

## 1. Project Overview

This project focuses on analyzing household electricity consumption data and predicting washing machine energy consumption using Machine Learning techniques.

The project uses the REFIT Electrical Load Measurements dataset to understand household energy usage patterns, identify appliance-level consumption trends, and develop a machine learning model for washing machine energy prediction.

The main objective is to analyze electricity consumption, understand energy usage behavior, and explore opportunities for improving household energy efficiency.

## 2. Project Objectives

* Analyze household electricity consumption patterns.
* Perform data cleaning and preprocessing.
* Identify missing values and data quality issues.
* Perform Exploratory Data Analysis (EDA).
* Understand washing machine energy consumption patterns.
* Perform feature engineering.
* Develop a machine learning model for energy prediction.
* Evaluate model performance using standard regression metrics.
* Generate visualizations and reports.
* Provide recommendations for improving energy efficiency.

## 3. Dataset Description

**Dataset Name:** REFIT Electrical Load Measurements

The REFIT dataset contains household electricity consumption measurements collected from residential buildings.

It provides electricity consumption information at household and individual appliance levels.

The dataset is used to analyze energy consumption patterns and understand how different appliances contribute to household electricity usage.

For more information, refer to `DATA.md`.

## 4. Technologies Used

| Technology   | Purpose                   |
| ------------ | ------------------------- |
| Python       | Programming language      |
| Pandas       | Data manipulation         |
| NumPy        | Numerical operations      |
| Matplotlib   | Data visualization        |
| Seaborn      | Statistical visualization |
| Scikit-learn | Machine Learning          |
| Joblib       | Model serialization       |
| Git          | Version control           |
| GitHub       | Project hosting           |

## 5. Project Workflow

The project follows these major steps:

1. Data Inspection – Understanding the dataset structure and available features.
2. Data Quality Analysis – Identifying missing values and data quality issues.
3. Data Cleaning – Handling missing values and preparing the dataset.
4. Exploratory Data Analysis – Analyzing household and appliance-level consumption patterns.
5. Feature Engineering – Creating useful features for machine learning.
6. Model Training – Training a Random Forest Regression model.
7. Model Evaluation – Evaluating prediction performance using regression metrics.
8. Visualization – Generating graphs and reports to understand the results.
9. Recommendations – Identifying possible improvements in energy consumption and model performance.

## 6. Machine Learning Model

**Model Used:** Random Forest Regression

Random Forest Regression is an ensemble machine learning algorithm that combines multiple decision trees to generate predictions.

It is used in this project to predict washing machine energy consumption based on the engineered input features.

### Model Evaluation Metrics

The model is evaluated using the following metrics:

* MAE (Mean Absolute Error)
* RMSE (Root Mean Squared Error)
* R² Score

These metrics help measure prediction errors and understand how well the model performs.

## 7. Project Structure

```text
Flock-Energy-Assignment/
│
├── models/
│   └── washing_machine_model.pkl
│
├── outputs/
│   ├── plots/
│   │   ├── actual_predicted_scatter.png
│   │   ├── actual_vs_predicted.png
│   │   ├── error_distribution.png
│   │   ├── feature_importance.png
│   │   ├── house1_cleaned_consumption.png
│   │   ├── house1_original_consumption.png
│   │   ├── washing_machine_cycle_comparison.png
│   │   ├── washing_machine_energy_comparison.png
│   │   └── washing_machine_hourly_comparison.png
│   │
│   └── reports/
│       ├── data_quality_summary.csv
│       ├── detailed_evaluation.csv
│       ├── feature_importance.csv
│       ├── missing_values.csv
│       ├── model_evaluation.csv
│       ├── model_predictions.csv
│       ├── numerical_statistics.csv
│       ├── washing_machine_cycles.csv
│       ├── washing_machine_hourly_profile.csv
│       ├── washing_machine_household_summary.csv
│       └── worst_prediction_cases.csv
│
├── src/
│   ├── __init__.py
│   ├── data_cleaning.py
│   ├── data_quality.py
│   ├── evaluate_model.py
│   ├── feature_engineering.py
│   ├── inspect_data.py
│   ├── train_model.py
│   └── washing_machine_eda.py
│
├── .gitignore
├── DATA.md
├── README.md
├── RESULTS.md
├── Recommendation.md
└── requirements.txt
```

## 8. Installation and Setup

Follow these steps to run the project on your local system.

### Step 1: Clone the Repository

```bash
git clone https://github.com/sweetykumari870/Flock-Energy-Assignment.git
```

### Step 2: Navigate to the Project Directory

```bash
cd Flock-Energy-Assignment
```

### Step 3: Create Virtual Environment

```powershell
python -m venv venv
```

### Step 4: Activate Virtual Environment

For Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

## 9. How to Run

Run the project scripts from the root project directory.

Execute the scripts in the following order:

```bash
python src/inspect_data.py

python src/data_quality.py

python src/data_cleaning.py

python src/washing_machine_eda.py

python src/feature_engineering.py

python src/train_model.py

python src/evaluate_model.py
```

**Note:** Make sure the REFIT dataset is downloaded and placed in the expected directory. Check the dataset paths configured in the Python scripts before executing them.

The trained model and generated outputs are saved in their respective directories according to the project configuration.

## 10. Results and Evaluation

The Random Forest Regression model was evaluated using different regression metrics.

### Overall Model Performance

| Evaluation Metric |   Result |
| ----------------- | -------: |
| MAE               |  15.80 W |
| RMSE              | 140.19 W |
| R² Score          |    0.323 |

### Active Consumption Performance

| Evaluation Metric |   Result |
| ----------------- | -------: |
| Active MAE        | 437.07 W |
| Active RMSE       | 782.61 W |

The results indicate that the model has scope for improvement, particularly in predicting active washing machine energy consumption.

Detailed evaluation results
