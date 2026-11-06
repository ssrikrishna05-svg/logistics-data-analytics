# Week 2 – Logistics Data Cleaning and Preprocessing Using Python

## Project Overview

This project demonstrates the process of collecting, cleaning, and preprocessing logistics data using Python. The objective is to improve data quality and prepare datasets for further analysis and decision-making.

## Project Objectives

- Identify missing values and duplicate records.
- Handle incorrect and inconsistent data.
- Detect outliers in logistics data.
- Apply data normalization techniques.
- Prepare a cleaned dataset for further analysis.

## Dataset Information

This project uses a simulated logistics dataset containing 62 initial records to demonstrate common data quality issues.

The Brazilian E-Commerce Public Dataset by Olist was used as a real-world research reference.

Reference: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- CSV and JSON

## Data Preprocessing Steps

1. Data collection and initial inspection
2. Duplicate record identification and removal
3. Handling missing values
4. Standardizing categorical values
5. Validating numerical and date fields
6. Outlier detection using the IQR method
7. Data normalization
8. Final data quality validation

## Project Results

| Data Quality Measure | Result |
|---|---:|
| Initial records | 62 |
| Duplicate records removed | 2 |
| Records quarantined for investigation | 6 |
| Final cleaned records | 54 |
| Missing state values resolved | 3 |
| Missing freight values imputed | 4 |
| Missing distance values imputed | 2 |

All numerical results were generated using simulated data and do not represent actual Olist logistics performance.

## Project Files

- `generate_sample_data.py` – Generates the simulated dataset.
- `week2_preprocessing.py` – Performs data cleaning and preprocessing.
- `sample_logistics_raw.csv` – Original sample dataset.
- `cleaned_logistics_orders.csv` – Cleaned dataset.
- `quarantined_records.csv` – Records requiring investigation.
- `quality_summary.json` – Summary of preprocessing results.
- `requirements.txt` – Python library dependencies.
- `missing_values.png` – Data quality visualization.
- `record_flow.png` – Record processing visualization.

## Expected Benefits

The preprocessing workflow improves dataset consistency, supports more reliable logistics analysis, and helps identify records requiring additional investigation.

## Conclusion

This project demonstrates how Python can be used to prepare logistics data for meaningful analysis. It also highlights the importance of data validation, transparent cleaning decisions, and reliable data quality checks.

**Internship Project – Week 2 | Data Analytics Using Python**
