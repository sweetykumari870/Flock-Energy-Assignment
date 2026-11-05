# Practical Recommendations – Flock Energy

## 1. Project Overview

This project analyses household electricity consumption using the REFIT Electrical Load Measurements dataset.

The objective is to identify washing machine usage patterns and estimate washing machine electricity consumption using aggregate household electricity readings.

A Random Forest Regression model was developed to estimate appliance-level power consumption.

## 2. Practical Energy-Saving Opportunity

### Identify High-Energy Washing Machine Usage

Flock Energy can use appliance disaggregation to help customers understand their washing machine usage patterns.

The system can identify:

* How frequently washing machines are used.
* Approximate electricity consumption per cycle.
* Peak power consumption during washing cycles.
* Household usage patterns across different times of the day.

Based on these insights, customers could receive personalised energy-saving suggestions.

### Suggested Customer Recommendations

1. Run washing machines with full loads whenever practical.
2. Prefer energy-efficient washing programmes.
3. Avoid unnecessary repeated washing cycles.
4. Consider shifting flexible washing loads to suitable off-peak periods where time-of-use tariffs are available.
5. Monitor unusual increases in electricity consumption.

## 3. How Flock Energy Can Measure Improvement

A pilot programme can be conducted with participating households.

The following metrics can be monitored:

| Metric                   | Measurement                 |
| ------------------------ | --------------------------- |
| Energy per washing cycle | kWh per cycle               |
| Daily washing energy     | Total daily kWh             |
| Weekly consumption       | Weekly kWh                  |
| Number of cycles         | Cycles per week             |
| Energy reduction         | Percentage reduction in kWh |

Energy savings can be calculated as:

Energy Reduction (%) =

(Baseline Energy - Post-Intervention Energy)
/ Baseline Energy × 100

Consumption should be compared over similar periods while considering changes in household occupancy and usage behaviour.

## 4. Limitations

### Overlapping Appliance Signatures

Different appliances can operate simultaneously and produce similar power consumption patterns.

This makes it difficult to identify the exact contribution of an individual appliance from aggregate readings.

### Missing Data

Missing readings and long data outages can affect cycle detection and model predictions.

### Household Differences

Different households have different appliances, operating habits and electricity consumption patterns.

A model trained on a limited number of households may not generalise to every household.

### UK-to-India Generalisation

REFIT represents UK households.

Indian households may have different appliance types, usage patterns, electricity supply conditions and climatic influences.

Therefore, the model should be validated using Indian household smart-meter data before commercial deployment.

## 5. Challenges with 15-Minute or 30-Minute Smart-Meter Data

Our analysis uses one-minute electricity readings.

If only 15-minute or 30-minute readings were available, appliance disaggregation would become more difficult.

Washing machines have changing power consumption during washing, heating, rinsing and spinning.

Lower-frequency measurements may miss these short-duration changes and combine multiple appliance activities into a single reading.

As a result:

* Short operating events may be missed.
* Individual washing cycles may become difficult to identify.
* Peak power consumption may be underestimated.
* Appliance-level predictions may become less accurate.

A model designed specifically for lower-frequency data would therefore require separate training and validation.

## 6. Final Recommendation

Flock Energy can use appliance disaggregation as a customer-insight tool to help households understand their electricity consumption.

The proposed solution can support personalised energy-saving recommendations and identify unusual consumption patterns.

However, before real-world deployment, the model should be tested on more diverse households, different smart-meter sampling intervals and local electricity consumption datasets.
