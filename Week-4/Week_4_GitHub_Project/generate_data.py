"""Create the explicitly SIMULATED shipment records for Week 4.
This is hypothetical educational data, NOT Olist or real company operations.
Python >=3.10; deterministic seed for reproducibility.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
DATA.mkdir(parents=True, exist_ok=True)
SEED = 20261117
N = 3600


def main():
    rng = np.random.default_rng(SEED)
    start = pd.Timestamp('2025-01-01')
    dates = start + pd.to_timedelta(rng.integers(0, 320, size=N), unit='D')
    distance = np.round(rng.uniform(80, 1450, N), 1)
    weight = np.round(rng.lognormal(np.log(38), 0.72, N).clip(1, 250), 1)
    region = rng.choice(['North', 'South', 'East', 'West', 'Central'], N, p=[.22, .22, .18, .23, .15])
    weather = rng.choice(['Clear', 'Rain', 'Storm'], N, p=[.67, .25, .08])
    traffic = rng.choice(['Low', 'Medium', 'High'], N, p=[.37, .43, .20])
    modes = rng.choice(['Road', 'Rail', 'Air'], N, p=[.50, .31, .19])
    priority = rng.choice(['Standard', 'Express'], N, p=[.70, .30])
    base = {'Road': 8., 'Rail': 13., 'Air': 12.}
    speed = {'Road': 48., 'Rail': 95., 'Air': 380.}
    traffic_mult = {'Road': 1., 'Rail': .28, 'Air': .08}
    wpenalty = {'Clear': 0., 'Rain': 2.4, 'Storm': 6.5}
    tpenalty = {'Low': 0., 'Medium': 3.0, 'High': 8.5}
    rpenalty = {'North': 1.1, 'South': 0.3, 'East': 1.6, 'West': 0.0, 'Central': 0.8}
    fixed = np.array([base[x] for x in modes])
    travel = distance / np.array([speed[x] for x in modes])
    delay_traffic = np.array([tpenalty[x] for x in traffic]) * np.array([traffic_mult[x] for x in modes])
    delay_weather = np.array([wpenalty[x] for x in weather]) * np.where(modes=='Air',.60,1.0)
    reg = np.array([rpenalty[x] for x in region])
    # Operational assumptions are invented; no external observation or causal validation.
    delivery = fixed + travel + delay_traffic + delay_weather + reg + .32 * np.log1p(weight) - np.where(priority=='Express', 1.0, 0.0)
    delivery += rng.normal(0., 2.0, N)
    delivery = np.round(np.maximum(3.5, delivery), 2)
    promised = np.where(priority=='Express', 25., 37.)
    df = pd.DataFrame({'shipment_id': [f'SIM-{i:05d}' for i in range(1,N+1)],
                       'booking_date':dates.strftime('%Y-%m-%d'), 'region':region,
                       'distance_km':distance, 'weight_kg':weight,
                       'weather_forecast': weather, 'traffic_forecast':traffic,
                       'transport_mode':modes, 'priority':priority,
                       'promised_hours':promised, 'delivery_hours':delivery})
    df = df.sort_values(['booking_date','shipment_id']).reset_index(drop=True)
    assert len(df) == N and df.shipment_id.nunique()==N
    assert df.isna().sum().sum()==0 and (df.delivery_hours>0).all()
    df.to_csv(DATA / 'hypothetical_shipments.csv', index=False)
    print('Saved', N, 'hypothetical records;', df.booking_date.min(), 'to', df.booking_date.max())

if __name__ == '__main__': main()
