"""Week 3 advanced logistics analytics, reproducible on Week 2 synthetic data.

No real Olist rows are included. The small synthetic example is for learning and
method demonstration, NOT for real-world predictions or operational decisions.
Run: python week3_analysis.py (after installing requirements.txt).
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler, FunctionTransformer

ROOT = Path(__file__).resolve().parent
DATA = ROOT/'data'
FIG = ROOT/'figures'
OUT = ROOT/'outputs'
FIG.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)
source = DATA/'cleaned_logistics_orders.csv'
df = pd.read_csv(source)
for c in ['order_purchase_timestamp','order_estimated_delivery_date','order_delivered_customer_date']:
    df[c] = pd.to_datetime(df[c], errors='coerce')
for c in ['freight_value','distance_km','delivery_days','delay_days']:
    df[c] = pd.to_numeric(df[c], errors='coerce')
assert len(df)==54 and df['order_id'].is_unique
assert (df['freight_value']>=0).all() and (df['distance_km']>=0).all()
assert df['order_status'].isin(['delivered','shipped','canceled']).all()
delivered = df[df['order_status']=='delivered'].copy()
assert delivered[['delivery_days','delay_days']].notna().all().all()
delivered['on_time_clean'] = delivered['delay_days'] <= 0
# Several rows are undelivered, so punctuality denominator is delivered-only.
n_delivered=len(delivered); n_on=int(delivered['on_time_clean'].sum()); n_late=n_delivered-n_on
rng=np.random.default_rng(2026)
boot=rng.choice(delivered['on_time_clean'].astype(int).to_numpy(), size=(10000,n_delivered), replace=True).mean(axis=1)
ci_lower,ci_upper = np.quantile(boot,[0.025,0.975])
flag = df['freight_value_outlier'].astype(str).str.lower().eq('true')
mean_all = float(df['freight_value'].mean())
mean_unflagged = float(df.loc[~flag,'freight_value'].mean())
median_freight = float(df['freight_value'].median())
# Spearman rank correlation is robust to outlier magnitude, but not to data artifacts.
rank_rho,rank_p = spearmanr(df['distance_km'],df['freight_value'])
regional = delivered.groupby('customer_state',dropna=False).agg(
    delivered_orders=('order_id','size'),on_time_orders=('on_time_clean','sum'),
    average_delivery_days=('delivery_days','mean'),average_delay_days=('delay_days','mean'))
regional['on_time_rate_pct']=100*regional['on_time_orders']/regional['delivered_orders']
regional=regional.sort_values(['delivered_orders','customer_state'],ascending=[False,True]).reset_index()
regional=regional.round(3)
regional.to_csv(OUT/'regional_performance.csv',index=False)

# Prediction uses only fields nominally observable by purchase/checkout. Do not
# include actual delivery date, derived delay, or the Week 2 pre-fitted scale.
features=['distance_km','freight_value','customer_state']
X=delivered[features].copy(); y=delivered['delivery_days'].to_numpy(dtype=float)
X_train,X_test,y_train,y_test,ids_train,ids_test=train_test_split(X,y,delivered['order_id'].to_numpy(),test_size=0.30,random_state=42)
num=['distance_km','freight_value']; cat=['customer_state']
prep=ColumnTransformer([
    ('num',Pipeline([('impute',SimpleImputer(strategy='median')),('log',FunctionTransformer(np.log1p)),('scale',RobustScaler())]),num),
    ('cat',OneHotEncoder(handle_unknown='ignore'),cat)
])
model=Pipeline([('prep',prep),('regression',Ridge(alpha=100))])
base=DummyRegressor(strategy='median')
base.fit(X_train,y_train); model.fit(X_train,y_train)
y_base=base.predict(X_test); y_pred=model.predict(X_test)
assert np.all(y_pred >= 0), "Predicted duration must be nonnegative"
def metrics(actual,pred):
    return dict(mae_days=round(float(mean_absolute_error(actual,pred)),3),
                rmse_days=round(float(np.sqrt(mean_squared_error(actual,pred))),3),
                r2=round(float(r2_score(actual,pred)),3))
base_m=metrics(y_test,y_base); model_m=metrics(y_test,y_pred)
# CV is performed only on the training portion; preprocessing lives inside folds.
cv=KFold(n_splits=5,shuffle=True,random_state=42)
cv_mae = -cross_val_score(model,X_train,y_train,cv=cv,scoring='neg_mean_absolute_error')
preds=pd.DataFrame({'order_id':ids_test,'actual_delivery_days':y_test,
                    'baseline_prediction_days':np.round(y_base,3),
                    'ridge_prediction_days':np.round(y_pred,3)})
preds.to_csv(OUT/'regression_test_predictions.csv',index=False)

# Clustering is descriptive, not predictive. Cap outliers before fitting and
# explicitly disclose the cap. Scale the two fields separately.
cf=df[['distance_km','freight_value']].copy()
cap={c:float(cf[c].quantile(.95)) for c in cf}
cf=cf.clip(upper=pd.Series(cap), axis=1)
scaled=StandardScaler().fit_transform(np.log1p(cf))
cluster_trials=[]
for k in [2,3,4]:
    km=KMeans(n_clusters=k,random_state=42,n_init=20).fit(scaled)
    sizes=np.bincount(km.labels_,minlength=k)
    score=float(silhouette_score(scaled,km.labels_))
    cluster_trials.append({'k':k,'silhouette':round(score,4),'sizes':[int(z) for z in sizes]})
eligible=[z for z in cluster_trials if min(z['sizes'])>=5]
chosen=max(eligible if eligible else cluster_trials,key=lambda z:z['silhouette'])
k_best=chosen['k']
kmeans=KMeans(n_clusters=k_best,random_state=42,n_init=20).fit(scaled)
df['cluster'] = kmeans.labels_.astype(int)
df['on_time_for_profile']=np.where(df['order_status'].eq('delivered'), (df['delay_days']<=0).astype(float),np.nan)
profile=df.groupby('cluster').agg(orders=('order_id','size'),
    mean_distance_km=('distance_km','mean'), median_distance_km=('distance_km','median'),
    mean_freight_brl=('freight_value','mean'),median_freight_brl=('freight_value','median'),
    delivered_orders=('on_time_for_profile','count'),
    on_time_rate=('on_time_for_profile','mean'))
profile['on_time_rate_pct']=100*profile['on_time_rate']
profile.drop(columns='on_time_rate',inplace=True)
profile=profile.reset_index().round(3)
profile.to_csv(OUT/'cluster_profile.csv',index=False)
df[['order_id','customer_state','order_status','distance_km','freight_value','cluster']].to_csv(OUT/'cluster_assignments.csv',index=False)

metrics_dict={
 'provenance':'Week 2 SYNTHETIC example; not Olist actual orders',
 'all_clean_orders':int(len(df)), 'delivered_orders':int(n_delivered),
 'shipped_orders':int(df['order_status'].eq('shipped').sum()),
 'canceled_orders':int(df['order_status'].eq('canceled').sum()),
 'on_time_deliveries':int(n_on),'late_deliveries':int(n_late),
 'on_time_rate_pct':round(100*n_on/n_delivered,2),
 'on_time_bootstrap_95pct_ci_pct':[round(100*float(ci_lower),1),round(100*float(ci_upper),1)],
 'average_delivery_days':round(float(delivered['delivery_days'].mean()),2),
 'median_delivery_days':round(float(delivered['delivery_days'].median()),2),
 'average_delay_days_signed':round(float(delivered['delay_days'].mean()),2),
 'mean_freight_brl_all_orders':round(mean_all,2),
 'mean_freight_brl_excluding_iqr_flagged':round(mean_unflagged,2),
 'median_freight_brl':round(median_freight,2),
 'freight_outlier_records':int(flag.sum()),
 'distance_freight_spearman_rho':round(float(rank_rho),3),
 'distance_freight_spearman_p':round(float(rank_p),4),
 'regression':{'train_rows':len(X_train),'test_rows':len(X_test),'baseline':base_m,
  'ridge':model_m,'ridge_cv_train_mae_mean':round(float(cv_mae.mean()),3),
  'ridge_cv_train_mae_std':round(float(cv_mae.std()),3)},
 'clustering':{'chosen_k':k_best,'chosen_silhouette':chosen['silhouette'],
               'outlier_cap_95pct':{c:round(v,2) for c,v in cap.items()},
               'candidate_results':cluster_trials},
 'warning':'Small 54-row simulated sample; models are demonstrations only. No causal or real-world generalization.'
}
(OUT/'analysis_summary.json').write_text(json.dumps(metrics_dict,indent=2),encoding='utf-8')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
NAVY='#243449'; BLUE='#3479A8'; TEAL='#24867C'; ORANGE='#D58A4E'; GREY='#A6B3BF'; RED='#C05B55'
def chart_save(filename):
    plt.tight_layout();plt.savefig(FIG/filename,dpi=180,bbox_inches='tight');plt.close()
# 1. Status and punctuality
fig,ax=plt.subplots(figsize=(7.6,3.5));labs=['On time','Late','Shipped','Canceled'];vals=[n_on,n_late,int(df.order_status.eq('shipped').sum()),int(df.order_status.eq('canceled').sum())]
bar=ax.bar(labs,vals,color=[TEAL,ORANGE,GREY,RED],width=.56)
ax.bar_label(bar,padding=3);ax.set_ylim(0,max(vals)+6);ax.set_ylabel('Order count');ax.set_title('Order outcomes (synthetic sample)',fontweight='bold',color=NAVY);chart_save('01_order_outcomes.png')
# 2 distribution
fig,ax=plt.subplots(figsize=(7.6,3.5));ax.hist(delivered['delivery_days'],bins=np.arange(3.5,21.0,2),color=BLUE,edgecolor='white')
ax.axvline(delivered['delivery_days'].mean(),color=RED,lw=2,linestyle='--',label=f"Mean {delivered['delivery_days'].mean():.1f} days")
ax.legend(frameon=False);ax.set_xlabel('Days from purchase to delivery');ax.set_ylabel('Delivered orders');ax.set_title('Delivery-time distribution',fontweight='bold',color=NAVY);chart_save('02_delivery_days_histogram.png')
# 3 region; only 5+ delivered for display
show=regional[regional['delivered_orders']>=5].sort_values('on_time_rate_pct')
fig,ax=plt.subplots(figsize=(7.6,3.6));bars=ax.barh(show['customer_state'],show['on_time_rate_pct'],color=BLUE,height=.55)
for b,n in zip(bars,show['delivered_orders']):ax.text(b.get_width()+1,b.get_y()+b.get_height()/2,f"{b.get_width():.0f}% (n={n})",va='center',fontsize=9)
ax.set_xlim(0,115);ax.set_xlabel('On-time rate among delivered orders (%)');ax.set_title('Regional punctuality: small samples',fontweight='bold',color=NAVY);chart_save('03_state_on_time.png')
#4 Outlier sensitivity
fig,ax=plt.subplots(figsize=(7.6,3.5));lbl=['Mean: all 54','Mean: excluding flagged','Median: all 54'];vv=[mean_all,mean_unflagged,median_freight]
bars=ax.bar(lbl,vv,color=[ORANGE,BLUE,TEAL],width=.58)
for b,v in zip(bars,vv):ax.text(b.get_x()+b.get_width()/2,v+4,f'R$ {v:,.1f}',ha='center',fontsize=10)
ax.set_ylabel('Freight value (BRL)');ax.set_ylim(0,max(vv)*1.28);ax.set_title('One extreme value changes the average',fontweight='bold',color=NAVY);chart_save('04_freight_sensitivity.png')
#5 regression comparison
fig,ax=plt.subplots(figsize=(7.6,3.5));vv=[base_m['mae_days'],model_m['mae_days']];bar=ax.bar(['Training-median baseline','Ridge regression'],vv,color=[GREY,BLUE],width=.55)
ax.bar_label(bar,fmt='%.2f d',padding=4);ax.set_ylabel('Hold-out mean absolute error (days)');ax.set_ylim(0,max(vv)*1.25+0.2);ax.set_title('Model test-set comparison (14 orders)',fontweight='bold',color=NAVY);chart_save('05_model_mae.png')
#6 actual-vs-pred scatter
fig,ax=plt.subplots(figsize=(7.6,4.0));ax.scatter(y_test,y_pred,s=55,alpha=.85,color=BLUE,edgecolors='white',linewidth=.6)
xy=[min(y_test.min(),y_pred.min())-1,max(y_test.max(),y_pred.max())+1];ax.plot(xy,xy,color=RED,linestyle='--',lw=1.4,label='Perfect prediction')
ax.set_xlim(xy);ax.set_ylim(xy);ax.set_xlabel('Actual delivery days');ax.set_ylabel('Predicted delivery days');ax.set_title('Ridge regression: actual versus predicted',fontweight='bold',color=NAVY);ax.legend(frameon=False);chart_save('06_regression_scatter.png')
#7 clustering scatter (raw numbers with axis truncated to show main body; include outliers in table)
fig,ax=plt.subplots(figsize=(7.6,4.0));cols=[BLUE,TEAL,ORANGE,RED]
for k in sorted(df.cluster.unique()):
 part=df[df.cluster==k];ax.scatter(part.distance_km.clip(upper=650),part.freight_value.clip(upper=220),s=55,alpha=.8,color=cols[int(k)%4],label=f'Cluster {k} (n={len(part)})',edgecolors='white',linewidth=.5)
ax.set_xlabel('Distance (km), displayed cap 650');ax.set_ylabel('Freight (BRL), displayed cap 220');ax.set_title('K-Means groupings (display values capped)',fontweight='bold',color=NAVY);ax.legend(frameon=False);chart_save('07_delivery_clusters.png')
# 8 numerical relationship - winsorized scatter
fig,ax=plt.subplots(figsize=(7.6,4.0));ax.scatter(df.distance_km.clip(upper=650),df.freight_value.clip(upper=220),alpha=.65,color=BLUE,s=45)
ax.set_xlabel('Distance (km), display cap 650');ax.set_ylabel('Freight (BRL), display cap 220');ax.set_title(f'Distance versus freight (Spearman rho={rank_rho:.2f})',fontweight='bold',color=NAVY);chart_save('08_distance_freight.png')
print(json.dumps(metrics_dict,indent=2))
print('\nREGIONS\n',regional.to_string(index=False))
print('\nCLUSTERS\n',profile.to_string(index=False))
