"""Reproducible Week 2 logistics data-cleaning pipeline.
Inputs are DEMONSTRATION DATA, not real Olist records. Python 3.10+.
Run: python generate_sample_data.py && python week2_preprocessing.py
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

HERE = Path(__file__).resolve().parent
FIG = HERE / 'figures'
FIG.mkdir(exist_ok=True)
RAW = HERE / 'sample_logistics_raw.csv'
raw = pd.read_csv(RAW, dtype='string', keep_default_na=True)
required = ['order_id','customer_state','order_status','order_purchase_timestamp',
            'order_estimated_delivery_date','order_delivered_customer_date',
            'freight_value','distance_km']
assert all(c in raw.columns for c in required), 'Dataset is missing expected columns'
raw = raw[required].copy()
for c in raw.columns:
    raw[c] = raw[c].str.strip().replace('', pd.NA)

# Sample is 1 row per order. Do NOT drop duplicate order_id in Olist order-items
# (that table intentionally has multiple items per order).
duplicate_rows = int(raw.duplicated(subset=['order_id'], keep='first').sum())
df = raw.drop_duplicates(subset=['order_id'], keep='first').copy()
df['order_status'] = df['order_status'].str.lower()
df['customer_state'] = df['customer_state'].str.upper()

raw_missing = {c:int(df[c].isna().sum()) for c in ['customer_state','freight_value','distance_km']}

for c in ['order_purchase_timestamp','order_estimated_delivery_date','order_delivered_customer_date']:
    df[c] = pd.to_datetime(df[c], format='%Y-%m-%d', errors='coerce')
for c in ['freight_value','distance_km']:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# Explainable quarantine prevents impossible or unresolved records from biasing KPIs.
reasons=[]
for _,r in df.iterrows():
    why=[]
    if pd.isna(r['order_id']): why.append('missing_order_id')
    if r['order_status'] not in ('delivered','shipped','canceled','processing'):
        why.append('unrecognized_order_status')
    if pd.isna(r['order_purchase_timestamp']): why.append('invalid_purchase_date')
    if pd.isna(r['order_estimated_delivery_date']): why.append('invalid_estimated_date')
    if pd.notna(r['freight_value']) and r['freight_value'] < 0: why.append('negative_freight')
    if pd.notna(r['distance_km']) and r['distance_km'] < 0: why.append('negative_distance')
    if r['order_status'] == 'delivered' and pd.isna(r['order_delivered_customer_date']):
        why.append('missing_or_invalid_delivered_date')
    if (pd.notna(r['order_purchase_timestamp']) and pd.notna(r['order_estimated_delivery_date'])
            and r['order_estimated_delivery_date'] < r['order_purchase_timestamp']):
        why.append('estimate_before_purchase')
    if (pd.notna(r['order_purchase_timestamp']) and pd.notna(r['order_delivered_customer_date'])
            and r['order_delivered_customer_date'] < r['order_purchase_timestamp']):
        why.append('delivery_before_purchase')
    reasons.append('; '.join(why))
df['quality_issue'] = reasons
quarantine = df[df['quality_issue'] != ''].copy()
clean = df[df['quality_issue'] == ''].copy()

# Impute non-critical continuous fields using robust median of retained records.
fill_values={}
for c in ['freight_value','distance_km']:
    median = float(clean[c].median())
    fill_values[c] = round(median,3)
    clean[c] = clean[c].fillna(median)
clean['customer_state'] = clean['customer_state'].fillna('UNKNOWN')

# Derived measures. Missing delivery dates on shipped/canceled orders are valid.
clean['delivery_days'] = (clean['order_delivered_customer_date'] - clean['order_purchase_timestamp']).dt.days
clean['delay_days'] = (clean['order_delivered_customer_date'] - clean['order_estimated_delivery_date']).dt.days
clean['on_time'] = np.where(clean['order_status'] == 'delivered', clean['delay_days'] <= 0, pd.NA)

# IQR outlier flags retain extreme records for investigation; no blind deletion.
outlier_thresholds={}
for c in ['freight_value','distance_km']:
    q1,q3 = clean[c].quantile([.25,.75])
    iqr = q3-q1
    lower,upper = float(q1-1.5*iqr),float(q3+1.5*iqr)
    clean[c+'_outlier'] = (clean[c]<lower)|(clean[c]>upper)
    outlier_thresholds[c]={'lower':round(lower,2),'upper':round(upper,2),
                          'flagged':int(clean[c+'_outlier'].sum())}

# Clip features only for training-style scaled features; keep raw values intact.
# For a real ML workflow, fit clip thresholds and scaler on training data ONLY.
model_features = ['freight_value','distance_km']
clipped = clean[model_features].copy()
clip_bounds={}
for c in model_features:
    lo,hi = clean[c].quantile([.05,.95])
    clipped[c] = clipped[c].clip(lo,hi)
    clip_bounds[c]={'p05':round(float(lo),2),'p95':round(float(hi),2)}
scaled = MinMaxScaler().fit_transform(clipped)
clean[['freight_scaled','distance_scaled']] = scaled

# Auditable outputs and invariant checks.
assert clean['order_id'].notna().all()
assert clean['order_id'].is_unique
assert clean['freight_value'].ge(0).all()
assert clean['distance_km'].ge(0).all()
assert (clean.loc[clean['order_status']=='delivered','delivery_days']>=0).all()
assert clean[['freight_scaled','distance_scaled']].ge(-1e-10).all().all()
assert clean[['freight_scaled','distance_scaled']].le(1+1e-10).all().all()

clean.to_csv(HERE/'cleaned_logistics_orders.csv',index=False,date_format='%Y-%m-%d')
quarantine.to_csv(HERE/'quarantined_records.csv',index=False,date_format='%Y-%m-%d')
delivered = clean[clean['order_status']=='delivered'].copy()
all_delivered = len(delivered)
on_time_count = int(delivered['on_time'].sum())
summary = {
 'dataset_type':'synthetic teaching example inspired by Olist schema; NOT real Olist records',
 'raw_rows':len(raw),'duplicate_order_rows_removed':duplicate_rows,
 'unique_order_rows':len(df),'quarantined_rows':len(quarantine),'clean_rows':len(clean),
 'clean_order_ids_unique':bool(clean['order_id'].is_unique),
 'missing_before_after_duplicate_removal':raw_missing,
 'imputation_medians':fill_values,
 'delivered_orders_in_clean':all_delivered,'on_time_deliveries':on_time_count,
 'on_time_rate_pct':round(100*on_time_count/all_delivered,2),
 'mean_delivery_days':round(float(delivered['delivery_days'].mean()),2),
 'mean_freight_brl':round(float(clean['freight_value'].mean()),2),
 'missing_after':{c:int(clean[c].isna().sum()) for c in ['customer_state','freight_value','distance_km']},
 'outlier_thresholds':outlier_thresholds,'model_clipping_bounds':clip_bounds,
 'quarantine_reasons':quarantine['quality_issue'].value_counts().to_dict(),
 'normalization_min':float(clean[['freight_scaled','distance_scaled']].min().min()),
 'normalization_max':float(clean[['freight_scaled','distance_scaled']].max().max()),
 }
(HERE/'quality_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')

# Publication-ready visual evidence (two separate charts).
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(8,3.65))
stages=['Raw records','After dedupe','Analysis-ready']
values=[len(raw),len(df),len(clean)]
bars=ax.bar(stages,values,color=['#C9D7E9','#729CCB','#1D5E9E'],width=.55)
ax.bar_label(bars,padding=3,fontweight='bold')
ax.set_ylim(0,max(values)+9);ax.set_ylabel('Number of records')
ax.set_title('Record-level data quality pipeline',fontweight='bold',pad=12)
fig.tight_layout()
fig.savefig(FIG/'record_flow.png',dpi=180,bbox_inches='tight')
plt.close(fig)

fig,ax=plt.subplots(figsize=(8,3.65))
cols=['customer_state','freight_value','distance_km']
x=np.arange(len(cols));width=.33
bars1=ax.bar(x-width/2,[raw_missing[c] for c in cols],width,label='Before',color='#D68B58')
bars2=ax.bar(x+width/2,[summary['missing_after'][c] for c in cols],width,label='After',color='#1D5E9E')
ax.set_xticks(x, ['Customer state','Freight value','Distance (km)'])
ax.bar_label(bars1,padding=3);ax.bar_label(bars2,padding=3)
ax.set_ylim(0,max(raw_missing.values())+2)
ax.set_ylabel('Missing fields (count)');ax.set_title('Missing-value reduction in usable rows',fontweight='bold',pad=12)
ax.legend(frameon=False)
fig.tight_layout();fig.savefig(FIG/'missing_values.png',dpi=180,bbox_inches='tight');plt.close(fig)

print(json.dumps(summary,indent=2))
