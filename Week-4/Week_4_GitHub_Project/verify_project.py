"""Simple deterministic structural and output checks for Week 4."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent
raw=pd.read_csv(P/'data/hypothetical_shipments.csv')
res=json.loads((P/'outputs/results.json').read_text())
validation=pd.read_csv(P/'outputs/validation_model_comparison.csv')
holdout=pd.read_csv(P/'outputs/holdout_predictions.csv')
assign=pd.read_csv(P/'outputs/optimized_mode_assignments.csv')
risk=pd.read_csv(P/'outputs/optimization_risk_sensitivity.csv')
assert len(raw)==3600 and raw.shipment_id.is_unique and not raw.isna().any().any()
assert len(holdout)==res['test_rows'] and len(assign)==60
assert set(validation.model)=={'Ridge','RandomForest','HistGradientBoosting','Median baseline'}
assert set(assign.optimized_mode)<={'Road','Rail','Air'}
assert (assign.optimized_mode=='Rail').sum()<=18 and (assign.optimized_mode=='Air').sum()<=10
assert np.isclose(assign.optimized_tariff_units.sum(),res['optimization']['optimized']['total_tariff_units'],atol=.2)
assert len(risk)==6 and len(list((P/'figures').glob('*.png')))==9
assert res['test_metrics_selected']['MAE_hours'] < res['test_metrics_baseline']['MAE_hours']
print('PASS: reproducibility audit (data, splits, model outputs, optimizer, charts)')
