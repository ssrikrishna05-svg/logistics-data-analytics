"""Week 4: end-to-end predictive ETA estimation and hypothetical mode optimization.
Inputs: data/hypothetical_shipments.csv; all outputs derived by code.
All results are SIMULATED, not observed real-world improvements.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from scipy.optimize import milp, Bounds, LinearConstraint

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'; OUT=ROOT/'outputs'; FIG=ROOT/'figures'
for p in (OUT,FIG): p.mkdir(parents=True,exist_ok=True)
SEED=42
FEATURES=['distance_km','weight_kg','region','weather_forecast','traffic_forecast','transport_mode','priority']
NUM=['distance_km','weight_kg']; CAT=list(set(FEATURES)-set(NUM))
CAT=['region','weather_forecast','traffic_forecast','transport_mode','priority']
MODES=['Road','Rail','Air']


def score(y,pred):
    return {'MAE_hours':float(mean_absolute_error(y,pred)),
            'RMSE_hours':float(np.sqrt(mean_squared_error(y,pred))),
            'R2':float(r2_score(y,pred))}


def make_pipe(name, params):
    categorical=OneHotEncoder(handle_unknown='ignore',sparse_output=False)
    if name=='Ridge':
        pre=ColumnTransformer([('scale',StandardScaler(),NUM),('cat',categorical,CAT)])
        model=Ridge(alpha=params['alpha'])
    elif name=='RandomForest':
        pre=ColumnTransformer([('num','passthrough',NUM),('cat',categorical,CAT)])
        model=RandomForestRegressor(n_estimators=110, max_depth=params['max_depth'],min_samples_leaf=3,
                                    random_state=SEED,n_jobs=1)
    elif name=='HistGradientBoosting':
        pre=ColumnTransformer([('num','passthrough',NUM),('cat',categorical,CAT)])
        model=HistGradientBoostingRegressor(max_iter=135,max_leaf_nodes=params['max_leaf_nodes'],
                                           learning_rate=.08,l2_regularization=1,random_state=SEED)
    return Pipeline([('preprocess',pre),('regressor',model)])


def date_cv(df, n_splits=3):
    days=np.array(sorted(df.booking_date.unique()))
    for train_day, val_day in TimeSeriesSplit(n_splits=n_splits).split(days):
        tr_dates=set(days[train_day]); va_dates=set(days[val_day])
        tr=np.flatnonzero(df.booking_date.isin(tr_dates).to_numpy())
        va=np.flatnonzero(df.booking_date.isin(va_dates).to_numpy())
        assert df.iloc[tr].booking_date.max() < df.iloc[va].booking_date.min()
        yield tr,va


def cost(mode,dist,weight):
    base={'Road':100.,'Rail':180.,'Air':650.}
    per_km={'Road':.85,'Rail':.58,'Air':1.48}
    per_kg={'Road':.25,'Rail':.17,'Air':.35}
    return base[mode]+per_km[mode]*dist+per_kg[mode]*weight


def optimize(pred_time, tariffs, promise, budget, rail_cap=18, air_cap=10):
    n=pred_time.shape[0]; k=3; assert k==pred_time.shape[1]
    late=np.maximum(pred_time-promise[:,None],0)
    # minimize predicted lateness hours; tiny cost tie-breaker, subject to budget/capacities
    c=(late+0.000002*tariffs).reshape(-1)
    eq=np.zeros((n,n*k))
    for i in range(n): eq[i,i*k:(i+1)*k]=1
    row_rail=np.tile(np.array([0,1,0]),n)[None,:]
    row_air=np.tile(np.array([0,0,1]),n)[None,:]
    row_cost=tariffs.reshape(1,-1)
    res=milp(c,integrality=np.ones(n*k),bounds=Bounds(np.zeros(n*k),np.ones(n*k)),
             constraints=[LinearConstraint(eq,1,1),
                         LinearConstraint(row_rail,-np.inf,rail_cap),
                         LinearConstraint(row_air,-np.inf,air_cap),
                         LinearConstraint(row_cost,-np.inf,budget)],
             options={'time_limit':30,'mip_rel_gap':0.001})
    if res.x is None: raise RuntimeError(f'MILP failed: {res.message}')
    chosen=res.x.reshape(n,k).argmax(axis=1)
    assert np.all(np.bincount(chosen,minlength=k)[1:] <= [rail_cap,air_cap])
    assert tariffs[np.arange(n),chosen].sum() <= budget+1e-6
    return chosen


def opt_metrics(chosen,times,fares,promise):
    ix=np.arange(len(chosen))
    chosen_time=times[ix,chosen]; chosen_cost=fares[ix,chosen]
    return {'predicted_mean_hours':float(np.mean(chosen_time)),
            'predicted_total_late_hours':float(np.maximum(chosen_time-promise,0).sum()),
            'predicted_SLA_breaches':int(np.sum(chosen_time>promise)),
            'total_tariff_units':float(np.sum(chosen_cost)),
            'Road':int((chosen==0).sum()),'Rail':int((chosen==1).sum()),'Air':int((chosen==2).sum())}


def chart_setup():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'figure.dpi':125,'savefig.dpi':190})


def savefig(fname):
    plt.tight_layout(); plt.savefig(FIG/fname,bbox_inches='tight'); plt.close()


def main():
    chart_setup()
    file=DATA/'hypothetical_shipments.csv'
    if not file.exists():
        raise FileNotFoundError('Run python generate_data.py first')
    df=pd.read_csv(file,parse_dates=['booking_date']).sort_values(['booking_date','shipment_id']).reset_index(drop=True)
    assert len(df)==3600 and df.shipment_id.is_unique
    assert not df[FEATURES+['delivery_hours','promised_hours']].isnull().any().any()
    assert df.delivery_hours.between(3.5,150).all()
    day_values=np.array(sorted(df.booking_date.unique()))
    cut1=day_values[int(len(day_values)*.70)]
    cut2=day_values[int(len(day_values)*.85)]
    tr=df[df.booking_date<cut1].copy(); val=df[(df.booking_date>=cut1)&(df.booking_date<cut2)].copy(); test=df[df.booking_date>=cut2].copy()
    assert tr.booking_date.max()<val.booking_date.min()<test.booking_date.min()
    candidates={'Ridge':[{'alpha':.2},{'alpha':5},{'alpha':40}],
                'RandomForest':[{'max_depth':8},{'max_depth':14}],
                'HistGradientBoosting':[{'max_leaf_nodes':12},{'max_leaf_nodes':24}]}
    cv_rows=[]; validation_rows=[]; chosen_pipes={}
    Xtr=tr[FEATURES]; ytr=tr.delivery_hours
    Xva=val[FEATURES]; yva=val.delivery_hours
    for family, params_list in candidates.items():
        for params in params_list:
            fold_maes=[]
            for itr,iva in date_cv(tr):
                pipe=make_pipe(family,params)
                pipe.fit(Xtr.iloc[itr],ytr.iloc[itr]); pred=pipe.predict(Xtr.iloc[iva])
                fold_maes.append(mean_absolute_error(ytr.iloc[iva],pred))
            cv_rows.append({'model':family,'hyperparameters':json.dumps(params,sort_keys=True),
                            'CV_MAE_hours':round(float(np.mean(fold_maes)),5),
                            'CV_MAE_sd':round(float(np.std(fold_maes)),5),
                            'fold_MAEs':json.dumps([round(float(x),4) for x in fold_maes])})
        rows=[r for r in cv_rows if r['model']==family]
        top=min(rows,key=lambda x:x['CV_MAE_hours'])
        bestparams=json.loads(top['hyperparameters'])
        p=make_pipe(family,bestparams)
        p.fit(Xtr,ytr)
        pv=p.predict(Xva)
        stats=score(yva,pv)
        validation_rows.append({'model':family,'CV_MAE_hours':top['CV_MAE_hours'],
                                'validation_MAE_hours':round(stats['MAE_hours'],5),
                                'validation_RMSE_hours':round(stats['RMSE_hours'],5),
                                'validation_R2':round(stats['R2'],5),
                                'hyperparameters':top['hyperparameters']})
        chosen_pipes[family]=(p,bestparams,pv)
    baseline=DummyRegressor(strategy='median');baseline.fit(Xtr,ytr)
    base_val=score(yva,baseline.predict(Xva))
    validation_rows.append({'model':'Median baseline','CV_MAE_hours':None,
                            'validation_MAE_hours':round(base_val['MAE_hours'],5),
                            'validation_RMSE_hours':round(base_val['RMSE_hours'],5),
                            'validation_R2':round(base_val['R2'],5),'hyperparameters':'{}'})
    valid_df=pd.DataFrame(validation_rows).sort_values('validation_MAE_hours').reset_index(drop=True)
    valid_df.to_csv(OUT/'validation_model_comparison.csv',index=False)
    pd.DataFrame(cv_rows).to_csv(OUT/'cross_validation_grid.csv',index=False)
    eligible=valid_df[valid_df.model!='Median baseline']
    champion_name=eligible.iloc[0]['model']
    val_model, params, val_preds=chosen_pipes[champion_name]
    q90=float(np.quantile(np.abs(yva.to_numpy()-val_preds),.90))
    dftrain=pd.concat([tr,val],ignore_index=True)
    champion=make_pipe(champion_name,params)
    champion.fit(dftrain[FEATURES],dftrain.delivery_hours)
    ytest=test.delivery_hours.to_numpy(); ypred=champion.predict(test[FEATURES]);base_test=DummyRegressor(strategy='median').fit(dftrain[FEATURES],dftrain.delivery_hours).predict(test[FEATURES])
    results={'selected_model':champion_name,'selected_hyperparameters':params,
             'sample_count':int(len(df)),'train_rows':len(tr),'validation_rows':len(val),'test_rows':len(test),
             'train_dates':[str(tr.booking_date.min().date()),str(tr.booking_date.max().date())],
             'validation_dates':[str(val.booking_date.min().date()),str(val.booking_date.max().date())],
             'test_dates':[str(test.booking_date.min().date()),str(test.booking_date.max().date())],
             'target_mean_hours':float(df.delivery_hours.mean()),
             'target_median_hours':float(df.delivery_hours.median()),
             'promised_target_breach_rate':float((df.delivery_hours>df.promised_hours).mean()),
             'validation_metrics_selected':score(yva,val_preds),
             'test_metrics_selected':score(ytest,ypred),
             'test_metrics_baseline':score(ytest,base_test),
             'calibrated_90pct_absolute_error_width':q90,
             'test_error_interval_coverage':float((np.abs(ytest-ypred)<=q90).mean())}
    preds=test[['shipment_id','booking_date','region','weather_forecast','traffic_forecast','transport_mode','priority','promised_hours']].copy()
    preds['actual_hours']=ytest;preds['predicted_hours']=np.round(ypred,3);preds['absolute_error_hours']=np.abs(ytest-ypred).round(3)
    preds.to_csv(OUT/'holdout_predictions.csv',index=False)
    segments=[]
    for column in ['region','transport_mode','weather_forecast','priority']:
        for k,grp in preds.groupby(column):
            segments.append({'dimension':column,'segment':k,'n':len(grp),'MAE_hours':round(float(grp.absolute_error_hours.mean()),3),
                             'actual_mean_hours':round(float(grp.actual_hours.mean()),3)})
    seg_df=pd.DataFrame(segments).sort_values(['dimension','segment'])
    seg_df.to_csv(OUT/'error_by_segment.csv',index=False)
    importance=permutation_importance(champion,test[FEATURES],ytest,scoring='neg_mean_absolute_error',n_repeats=4,random_state=SEED,n_jobs=1)
    import_df=pd.DataFrame({'feature':FEATURES,'MAE_increase_when_shuffled':importance.importances_mean}).sort_values('MAE_increase_when_shuffled',ascending=False)
    import_df.to_csv(OUT/'permutation_importance.csv',index=False)
    # Production-planning exercise: 60 historical holdout profiles treated as hypothetical future demand.
    # No observed counterfactual carrier/route outcomes exist.
    rng=np.random.default_rng(27); opt_n=60
    sample=test.iloc[np.sort(rng.choice(len(test),size=opt_n,replace=False))].copy().reset_index(drop=True)
    expected=np.zeros((opt_n,3)); fare=np.zeros((opt_n,3))
    for j,mode in enumerate(MODES):
        q=sample[FEATURES].copy();q['transport_mode']=mode
        expected[:,j]=champion.predict(q)
        fare[:,j]=cost(mode,sample.distance_km.to_numpy(),sample.weight_kg.to_numpy())
    promises=sample.promised_hours.to_numpy()
    base=np.zeros(opt_n,dtype=int); base_m=opt_metrics(base,expected,fare,promises)
    budget_multiplier=1.15
    budget=float(base_m['total_tariff_units']*budget_multiplier)
    choice=optimize(expected,fare,promises,budget)
    optimized=opt_metrics(choice,expected,fare,promises)
    results['optimization']={'n_shipments':opt_n,'budget_units':budget,'budget_relative_to_all_road':budget_multiplier,
                             'max_rail_slots':18,'max_air_slots':10,
                             'all_road_baseline':base_m,'optimized':optimized,
                             'predicted_lateness_reduction_pct':float(100*(base_m['predicted_total_late_hours']-optimized['predicted_total_late_hours'])/max(base_m['predicted_total_late_hours'],1e-12)),
                             'tariff_change_pct':float(100*(optimized['total_tariff_units']-base_m['total_tariff_units'])/base_m['total_tariff_units'])}
    # Risk stress test: an identical positive model-error shock is added to both plans.
    stress=[]
    for shock_label, delta in [('point estimate',0.0),('add validation 90pct absolute-error width',q90),('10pct slower actual service',None)]:
        for label, selected in [('all road',base),('optimized',choice)]:
            travel=expected[np.arange(opt_n),selected]
            altered=(travel+delta) if delta is not None else (travel*1.10)
            stress.append({'shock':shock_label,'plan':label,'late_hours':float(np.maximum(altered-promises,0).sum()),
                           'SLA_breaches':int(np.sum(altered>promises))})
    pd.DataFrame(stress).to_csv(OUT/'optimization_risk_sensitivity.csv',index=False)
    results['optimization']['risk_sensitivity']=stress
    outopt=sample[['shipment_id','priority','region','distance_km','weather_forecast','traffic_forecast','promised_hours']].copy()
    outopt['baseline_mode']='Road';outopt['optimized_mode']=[MODES[x] for x in choice]
    outopt['baseline_predicted_hours']=expected[:,0].round(3)
    outopt['optimized_predicted_hours']=expected[np.arange(opt_n),choice].round(3)
    outopt['baseline_tariff_units']=fare[:,0].round(2)
    outopt['optimized_tariff_units']=fare[np.arange(opt_n),choice].round(2)
    outopt.to_csv(OUT/'optimized_mode_assignments.csv',index=False)
    scenarios=[]
    for mult in [1.0,1.05,1.10,1.15,1.20]:
        c=optimize(expected,fare,promises,base_m['total_tariff_units']*mult)
        m=opt_metrics(c,expected,fare,promises);m['budget_multiplier']=mult
        scenarios.append(m)
    pd.DataFrame(scenarios).to_csv(OUT/'budget_sensitivity.csv',index=False)
    results['sensitivity']=scenarios
    # Figures
    fig,ax=plt.subplots(figsize=(7.0,3.7));ax.hist(df.delivery_hours,bins=30,color='#2b6086',edgecolor='white')
    ax.set(xlabel='Realized delivery time in hypothetical dataset (hours)',ylabel='Shipments',title='1. Shipment delivery-time distribution');savefig('01_delivery_time_distribution.png')
    means=df.groupby('transport_mode').delivery_hours.mean().reindex(MODES)
    fig,ax=plt.subplots(figsize=(7.,3.8));ax.bar(means.index,means.values,color=['#376a94','#9d762c','#699979']);ax.set(ylabel='Mean observed hours (simulated)',title='2. Mean delivery hours by transport mode')
    for i,v in enumerate(means):ax.text(i,v+.3,f'{v:.1f}',ha='center',fontsize=9)
    savefig('02_mode_descriptive.png')
    fig,ax=plt.subplots(figsize=(7.5,3.8));sub=valid_df.sort_values('validation_MAE_hours')
    ax.barh(sub.model,sub.validation_MAE_hours,color='#4e8098');ax.invert_yaxis();ax.set(xlabel='Validation MAE (hours, smaller is better)',title='3. Candidate model comparison');savefig('03_validation_models.png')
    fig,ax=plt.subplots(figsize=(5.8,4.5));ax.scatter(ytest,ypred,alpha=.35,s=13,color='#397d91');m=max(ytest.max(),ypred.max());ax.plot([0,m],[0,m],'--',color='#ca695d');ax.set(xlabel='Actual delivery hours',ylabel='Predicted delivery hours',title='4. Holdout predictions vs actual');savefig('04_holdout_actual_predicted.png')
    fig,ax=plt.subplots(figsize=(7.2,3.7));arr=[preds.loc[preds.transport_mode==m,'absolute_error_hours'] for m in MODES]
    ax.boxplot(arr,tick_labels=MODES,showfliers=False);ax.set(ylabel='Absolute prediction error (hours)',title='5. Holdout error by transport mode');savefig('05_error_by_mode.png')
    fig,ax=plt.subplots(figsize=(6.8,3.5));xs=['All-road plan','Optimized plan'];vs=[base_m['predicted_total_late_hours'],optimized['predicted_total_late_hours']]
    ax.bar(xs,vs,color=['#bc7164','#428b87']);ax.set(ylabel='Total predicted late hours',title='6. Optimization scenario: predicted lateness')
    for i,v in enumerate(vs):
        ax.text(i,v+.5,f'{v:.1f}',ha='center')
    savefig('06_optimization_late_hours.png')
    fig,ax=plt.subplots(figsize=(6.8,3.5));vs=[base_m['total_tariff_units'],optimized['total_tariff_units']]
    ax.bar(xs,vs,color=['#bc7164','#428b87']);ax.set(ylabel='Total illustrative tariff units',title='7. Optimization scenario: total tariff')
    for i,v in enumerate(vs):
        ax.text(i,v+300,f'{v:,.0f}',ha='center')
    savefig('07_optimization_tariff.png')
    fig,ax=plt.subplots(figsize=(7.0,3.8));sdf=pd.DataFrame(scenarios)
    ax.plot(sdf.total_tariff_units,sdf.predicted_total_late_hours,'o-',color='#40788c')
    for _,r in sdf.iterrows():ax.annotate(f"{r['budget_multiplier']:.0%}",(r.total_tariff_units,r.predicted_total_late_hours),xytext=(3,6),textcoords='offset points',fontsize=8)
    ax.set(xlabel='Actual tariff spend (illustrative units)',ylabel='Predicted late hours',title='8. Budget–lateness sensitivity (hypothetical)');savefig('08_budget_sensitivity.png')
    fig,ax=plt.subplots(figsize=(7.2,4.3));temp=import_df.sort_values('MAE_increase_when_shuffled')
    ax.barh(temp.feature,temp.MAE_increase_when_shuffled,color='#468b81');ax.set(xlabel='Increase in MAE when feature permuted (hours)',title='9. Holdout permutation importance');savefig('09_feature_importance.png')
    with open(OUT/'results.json','w',encoding='utf8') as f:json.dump(results,f,indent=2,sort_keys=True)
    print(json.dumps({'selected_model':results['selected_model'],'test':results['test_metrics_selected'],
                      'baseline':results['test_metrics_baseline'],'optimizer':results['optimization']},indent=2))

if __name__=='__main__':main()
