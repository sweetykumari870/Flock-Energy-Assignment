# Household Energy Consumption Analysis and Washing Machine Energy Prediction

## 1. Project Overview

This project focuses on analysing household electricity consumption and estimating washing machine energy usage using machine learning.

The project uses the REFIT Electrical Load Measurements dataset, which contains electricity consumption readings from different household appliances.

The main objective is to understand household energy usage patterns, analyse washing machine consumption, and develop a machine learning model to estimate washing machine power consumption.

## 2. Project Objectives

* Analyse household electricity consumption.
* Perform data cleaning and preprocessing.
* Study washing machine usage patterns across different households.
* Identify washing machine operating cycles.
* Perform exploratory data analysis.
* Create meaningful features for machine learning.
* Train a Random Forest regression model.
* Evaluate model performance on an unseen household.
* Generate useful visualisations and reports.

## 3. Dataset

The project uses the REFIT Electrical Load Measurements dataset.

Selected households:

* House 1
* House 2
* House 3
* House 15

The project focuses on washing machine consumption and aggregate household electricity consumption.

## 4. Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* VS Code

## 5. Project Structure

```text
Flock-Energy-Assignment/
│
├── data/
│   ├── raw/
│   ├── cleaned/
│   └── processed/
│
├── models/
│   └── washing_machine_model.pkl
│
├── notebooks/
│
├── outputs/
│   ├── plots/
│   └── reports/
│
├── src/
│   ├── data_cleaning.py
│   ├── data_quality.py
│   ├── washing_machine_eda.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   └── evaluate_model.py
│
├── DATA.md
├── RESULTS.md
├── Recommendation.md
└── README.md
```

## 6. Methodology

### Step 1: Data Quality Analysis

Checked missing values, numerical statistics, duplicate records, and general data quality.

### Step 2: Data Cleaning

Prepared the household electricity readings, handled timestamps, and resampled House 1 data to one-minute intervals.

### Step 3: Exploratory Data Analysis

Analysed washing machine consumption, operating cycles, household differences, and hourly usage patterns.

### Step 4: Feature Engineering

Created time-based features, lag features, and rolling statistical features using aggregate electricity consumption.

### Step 5: Model Training

Developed a Random Forest regression model.

Training households:

* House 1
* House 2
* House 3

Testing household:

* House 15

### Step 6: Model Evaluation

Evaluated the model using MAE, RMSE, and R² score. Its performance was also compared with a zero-prediction baseline.

## 7. Model Results

| Metric             |   Result |
| ------------------ | -------: |
| Overall MAE        |  15.80 W |
| Overall RMSE       | 140.19 W |
| R² Score           |    0.323 |
| Active-period MAE  | 437.07 W |
| Active-period RMSE | 782.61 W |
| Baseline MAE       |  17.15 W |
| Baseline RMSE      | 171.20 W |

The model performed better than the zero-prediction baseline in overall MAE and RMSE.

However, active-period prediction errors remain high, indicating scope for further improvement.

## 8. Generated Outputs

### Reports

* Data quality summary
* Missing values report
* Numerical statistics
* Washing machine household summary
* Washing machine cycle analysis
* Model evaluation
* Feature importance
* Detailed evaluation
* Worst prediction cases

### Visualisations

* Original and cleaned household consumption
* Washing machine hourly comparison
* Washing machine cycle comparison
* Washing machine energy comparison
* Actual vs predicted consumption
* Error distribution
* Feature importance
* Actual vs predicted scatter plot

All generated reports are saved in `outputs/reports/`.

All generated plots are saved in `outputs/plots/`.

## 9. Installation and Setup

Clone or download the project repository.

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

## 10. How to Run

Run the project scripts from the root project directory.

Example:

```bash
python src/data_quality.py
python src/data_cleaning.py
python src/washing_machine_eda.py
python src/feature_engineering.py
python src/train_model.py
python src/evaluate_model.py
```

Make sure the dataset is placed in the expected folders and the paths in the scripts match your local project structure before running them.

## 11. Recommendations

* Collect more household electricity consumption data.
* Improve active washing machine consumption prediction.
* Experiment with additional machine learning algorithms.
* Perform further feature selection and hyperparameter tuning.
* Validate the model on more unseen households.
* Explore practical household energy-saving strategies.

For more details, refer to `Recommendation.md`.

## 12. Documentation

* `DATA.md` – Dataset information and preprocessing.
* `RESULTS.md` – Analysis findings and model evaluation.
* `Recommendation.md` – Recommendations and project limitations.

## 13. Conclusion

This project demonstrates a complete workflow for household electricity consumption analysis and washing machine energy prediction.

It combines data preprocessing, exploratory analysis, feature engineering, machine learning, and model evaluation to understand appliance-level energy consumption.

The results provide a foundation for further research and improvement in household energy monitoring.

---

**Project:** Flock Energy Assignment
**Model:** Random Forest Regression
**Dataset:** REFIT Electrical Load Measurements
