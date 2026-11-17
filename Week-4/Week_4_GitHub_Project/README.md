# Week 4 — Predictive Modeling & Optimization in Logistics Systems

**Internship assignment:** Predict delivery duration with Python, compare and validate models, then propose and solve a constrained transport-allocation optimization problem.

> **Data integrity note:** All 3,600 records are **hypothetical and fully simulated** with a fixed random seed. The Olist dataset is a *public research reference only*, not the model-training dataset. The optimization is a **counterfactual educational scenario**, not observed commercial savings or confirmed causal effects.

## What this demonstrates

1. Define target: **delivery_hours** (elapsed hours from planned dispatch to delivery in a synthetic scenario).
2. Generate `data/hypothetical_shipments.csv` with `generate_data.py`, and audit missing/duplicate values.
3. Chronological train/validation/test split **by booking date**, avoiding date overlap and post-booking target leakage.
4. Threefold expanding-window cross-validation of 7 hyperparameter configurations covering Ridge, Random Forest, and HistGradientBoosting; select model by independent validation MAE. Compare to median baseline.
5. Hold out final test dates until after model selection; report MAE, RMSE, R², error segments, permutation importance, and approximate interval coverage.
6. Solve a mixed-integer transport-mode assignment for **60 sample demand profiles**, with model-predicted hours, declared hypothetical tariffs, 18 rail slots, 10 air slots, and a 15% budget allowance above all-road tariffs.
7. Report budget sensitivity and positive-error stress tests, plus limitations and action recommendations.

## Results (seed-fixed, executed locally)

| Test metric | Selected HistGradientBoosting | Median baseline |
|---|---:|---:|
| MAE | **1.668 hours** | 7.158 hours |
| RMSE | **2.107 hours** | 8.963 hours |
| R² | **0.9428** | -0.0343 |

For the **60-shipment simulated planning scenario**, modeled late hours change from **206.78** (all-road benchmark) to **3.62** (optimized assignment), with hypothetical total tariffs changing from **51,325.85** to **58,834.15 planning units**. **This is a model-generated comparison, not a realized outcome.** Actual effect depends on true carrier availability, real costs, causal mode effects, and model calibration.

## Quick start (Windows / VS Code)

Open the extracted `Week-4` folder in VS Code or Windows Terminal. Install Python 3.10+.

```bash
python -m pip install -r requirements.txt
python generate_data.py
python week4_predict_optimize.py
python verify_project.py
```

All result CSVs/JSONs will be recreated in `outputs/` and all graphs in `figures/`.

## GitHub folder structure

```text
Week-4/
  README.md
  DATA_DICTIONARY.md
  requirements.txt
  generate_data.py
  week4_predict_optimize.py
  verify_project.py
  Week_4_Predictive_Modeling_Optimization_Report.docx
  data/
    hypothetical_shipments.csv
  outputs/
    results.json
    validation_model_comparison.csv
    cross_validation_grid.csv
    holdout_predictions.csv
    error_by_segment.csv
    permutation_importance.csv
    optimized_mode_assignments.csv
    budget_sensitivity.csv
    optimization_risk_sensitivity.csv
  figures/
    01_delivery_time_distribution.png
    ...
    09_feature_importance.png
```

## Methodological safeguards

- Dataset fabrication is disclosed; no real observational data are claimed.
- Model uses booking-time covariates only. Outcome, shipment ID, booked date and actual SLA breach do not enter the predictor.
- Three distinct chronological splits prevent training on later observations.
- Scalers/encoders are inside sklearn Pipelines and fitted within folds.
- Model comparison uses CV + validation; test results are reported **once** after final fitting.
- Per-mode scenario predictions are not causal estimates. They are conditional extrapolations that require real controlled pilots.
- Optimization is integer-constrained; it does **not** compute actual routes, driver rosters, or observed financial gains.
- The 90% absolute-error width is an empirical calibration heuristic, not a rigorous universal prediction interval.

## Sources / references

- Olist Brazilian E-Commerce Public Dataset (research context only): https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
- scikit-learn TimeSeriesSplit: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
- scikit-learn regression metrics: https://scikit-learn.org/stable/api/sklearn.metrics.html
- SciPy mixed-integer optimization: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html

## Submission

Upload `Week_4_Predictive_Modeling_Optimization_Report.docx` as your report and link to this `Week-4` folder in the internship portal. Read and understand the results before submitting; do not present this as an analysis of actual commercial logistics data.
