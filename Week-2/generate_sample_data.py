"""Generate invented order-level logistics records for a Week 2 teaching exercise.
These are NOT actual Olist customer orders. Run before week2_preprocessing.py.
"""
from pathlib import Path
import csv
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / 'sample_logistics_raw.csv'
rng = np.random.default_rng(20261106)
rows = []
states = ['SP', 'RJ', 'MG', 'BA', 'PR', 'RS', 'SC']
for i in range(60):
    purchase = pd.Timestamp('2025-03-01') + pd.Timedelta(days=i)
    estimate = purchase + pd.Timedelta(days=int(rng.integers(7, 17)))
    status = 'delivered' if i < 52 else ('shipped' if i < 57 else 'canceled')
    actual = purchase + pd.Timedelta(days=int(rng.integers(5, 20))) if status == 'delivered' else pd.NaT
    rows.append({
        'order_id':f'SIM-{i+1:04d}',
        'customer_state':states[i % len(states)],
        'order_status':status,
        'order_purchase_timestamp':purchase.strftime('%Y-%m-%d'),
        'order_estimated_delivery_date':estimate.strftime('%Y-%m-%d'),
        'order_delivered_customer_date': actual.strftime('%Y-%m-%d') if pd.notna(actual) else '',
        'freight_value':f'{float(rng.uniform(30, 170)):.2f}',
        'distance_km':f'{float(rng.uniform(35, 600)):.1f}',
    })

# Inject realistic but intentionally artificial data quality problems.
rows[2]['order_status'] = ' DELIVERED '
rows[8]['order_status'] = 'Delivered'
rows[53]['order_status'] = ' Shipped '
rows[12]['customer_state'] = 'sp'
for i in [4, 16, 38]: rows[i]['customer_state'] = ''
for i in [7, 13, 29, 37]: rows[i]['freight_value'] = ''
for i in [14, 41]: rows[i]['distance_km'] = ''
rows[9]['freight_value'] = '-25.00'          # impossible negative amount
rows[15]['freight_value'] = '3000.00'        # investigate, do not silently delete
rows[20]['distance_km'] = '2200.0'          # investigate, do not silently delete
rows[24]['order_delivered_customer_date'] = '' # delivered, but delivery time absent
rows[31]['order_delivered_customer_date'] = '2025-15-99' # parse failure
rows[46]['order_delivered_customer_date'] = '2025-04-01' # earlier than purchase
rows[3]['order_estimated_delivery_date'] = '2025-99-15'  # parse failure
rows[5]['order_id'] = ''                     # required ID missing

# Duplicate entire order-level rows: 62 inputs, 60 originally generated orders.
rows.append(rows[2].copy())
rows.append(rows[8].copy())

with OUT.open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f'Created {OUT.name}: {len(rows)} rows; synthetic examples only.')
