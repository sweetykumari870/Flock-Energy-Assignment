# Project Results and Model Evaluation

## 1. Overview

This project analyses household electricity consumption and estimates washing machine energy usage using the REFIT dataset.

The analysis includes household-wise consumption patterns, washing machine operating cycles, feature engineering, and machine learning model evaluation.

## 2. Household-wise Washing Machine Analysis

The following results were obtained from the selected households:

| Household | Total Cycles | Average Power (W) | Maximum Power (W) | Average Duration (Minutes) | Average Energy (kWh) | Total Energy (kWh) |
| --------- | -----------: | ----------------: | ----------------: | -------------------------: | -------------------: | -----------------: |
| House 1   |          908 |            540.68 |           2461.75 |                      17.24 |               0.1590 |             144.34 |
| House 2   |          796 |            271.67 |           2250.56 |                      41.62 |               0.2526 |             201.07 |
| House 3   |          834 |            471.93 |           2592.00 |                      53.73 |               0.4476 |             373.31 |
| House 15  |          849 |            382.09 |           2359.70 |                      25.47 |               0.2357 |             200.11 |

### Observations

* House 1 has the highest average power consumption.
* House 3 has the highest average washing machine cycle duration.
* House 3 also has the highest average and total energy consumption.
* House 2 has the lowest average power consumption.
* Washing machine usage patterns vary across households.

## 3. Machine Learning Model Evaluation

A Random Forest model was trained using Houses 1, 2, and 3 and tested on House 15.

The following evaluation results were obtained:

| Evaluation Metric  | Model Result |
| ------------------ | -----------: |
| Overall MAE        |      15.80 W |
| Overall RMSE       |     140.19 W |
| R² Score           |        0.323 |
| Active-period MAE  |     437.07 W |
| Active-period RMSE |     782.61 W |
| Baseline MAE       |      17.15 W |
| Baseline RMSE      |     171.20 W |

## 4. Model Performance Analysis

* The model achieved an overall MAE of approximately 15.80 W.
* The overall RMSE was approximately 140.19 W.
* The R² score was approximately 0.323, indicating modest explanatory performance.
* The model performed better than the zero-prediction baseline on both overall MAE and RMSE.
* However, active-period errors are considerably higher, showing that predicting actual washing machine consumption during operation remains challenging.

## 5. Visual Results

The project generated the following visualisations:

* Actual vs Predicted Consumption
* Actual vs Predicted Scatter Plot
* Error Distribution
* Feature Importance
* Household-wise Washing Machine Energy Comparison
* Washing Machine Cycle Comparison
* Hourly Consumption Comparison

All plots are available in:

`outputs/plots/`

## 6. Key Findings

1. Washing machine consumption patterns differ between households.
2. Operating duration and energy consumption vary across households.
3. The Random Forest model performs better than the zero-prediction baseline overall.
4. Active washing machine consumption is more difficult to predict accurately.
5. More representative household data and further model tuning may improve performance.

## 7. Limitations

* The model was tested on only one unseen household.
* Overall error metrics can be influenced by periods when the washing machine is switched off.
* Active-period prediction errors remain high.
* The results depend on the selected households and available dataset.
* The model requires further validation before real-world deployment.

## 8. Conclusion

This project demonstrates a complete workflow for household electricity consumption analysis and washing machine energy estimation.

It includes data cleaning, exploratory data analysis, feature engineering, Random Forest model training, and evaluation.

The model shows improvement over the zero-prediction baseline in overall error metrics. However, active-period prediction needs further improvement.
