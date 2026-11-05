# Data Documentation – REFIT Dataset

## 1. Dataset Overview

This project uses the REFIT Electrical Load Measurements dataset provided by the University of Strathclyde.

The dataset contains electricity consumption measurements from 20 UK households.

It includes:

* Whole-house aggregate electricity consumption.
* Individual appliance electricity readings.
* Timestamp information.

## 2. Dataset Selection

Two versions of the dataset were used:

1. Raw REFIT dataset.
2. Official cleaned REFIT dataset.

The raw dataset was used for House 1 data-quality analysis and cleaning.

The official cleaned dataset was used for washing-machine exploratory analysis and appliance-disaggregation modelling.

## 3. Data Cleaning

The following preprocessing operations were implemented:

* Timestamp conversion into datetime format.
* Removal of invalid timestamps.
* Chronological sorting.
* Duplicate timestamp handling using average readings.
* Conversion of appliance readings into numerical format.
* Resampling into one-minute intervals.

## 4. Missing Values

Missing readings were inspected during data-quality analysis.

Missing intervals were preserved during the initial resampling process rather than automatically filling all missing readings.

Long outages require separate investigation before interpolation or imputation.

## 5. Outlier Analysis

Potential outliers were identified using the Interquartile Range (IQR) method.

Negative electricity readings were also checked.

Potential outliers should be investigated before removal because unusually high readings may represent genuine appliance activity.

## 6. Sampling Frequency

The analysis was performed at one-minute resolution.

## 7. Data Limitations

* Missing readings can affect analysis.
* Long outages can interrupt appliance activity.
* Household consumption patterns differ.
* Appliance signatures can overlap.
* REFIT represents UK households rather than Indian households.

## 8. Reproducibility

The Python scripts used for data inspection, cleaning, feature engineering, exploratory analysis and modelling are maintained in the `src` directory.

The original REFIT datasets must be downloaded separately before running the complete pipeline.
