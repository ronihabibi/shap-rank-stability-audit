import numpy as np,pandas as pd,json,sys,os
R=sys.argv[1] if len(sys.argv)>1 else os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','results')
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
E=pd.read_csv(os.path.join(R,'multi_estimates_vs_truth.csv')); rng=np.random.default_rng(1); out={}
def stats(s,thr_t=0.8):
    rp=(s.t_modal>=thr_t).astype(int); fl=(s.srsi>=.85).astype(int)
    tp=((fl==1)&(rp==1)).sum(); 
    return dict(N=len(s),repro=rp.mean(),flag=fl.mean(),prec=tp/max(fl.sum(),1),rec=tp/max(rp.sum(),1),acc=(fl==rp).mean(),
                bias=(s.srsi-s.t_srsi).mean(),mae=(s.srsi-s.t_srsi).abs().mean(),rho_sd=spearmanr(s.sd,s.t_sd)[0])
S=E[E.aggr=='sum']
# per-draw clustered bootstrap CI for key quantities
def cl_boot(df,fn,B=400):
    g={k:v for k,v in df.groupby(['data','n','draw'])}; keys=list(g); v=[]
    for _ in range(B):
        ii=rng.integers(0,len(keys),len(keys)); v.append(fn(pd.concat([g[keys[i]] for i in ii])))
    return np.percentile(v,[2.5,97.5]).round(3).tolist()
rows=[]
for var in ['boot','seed']:
    s=S[S['var']==var]; r=stats(s); r['var']=var
    r['bias_ci']=cl_boot(s,lambda z:(z.srsi-z.t_srsi).mean()); r['flag_ci']=cl_boot(s,lambda z:(z.srsi>=.85).mean()); r['mae_ci']=cl_boot(s,lambda z:(z.srsi-z.t_srsi).abs().mean())
    rows.append(r)
out['main']=rows
# by n
out['by_n']=[dict(n=int(n),var=v,**stats(S[(S.n==n)&(S['var']==v)])) for n in (100,300) for v in ('boot','seed')]
# by dataset
out['by_data']=[dict(data=dn,n=int(n),var=v,**stats(g)) for (dn,n,v),g in S.groupby(['data','n','var'])]
# truth thresholds
out['thr']=[dict(thr=t,var=v,**{k:x for k,x in stats(S[S['var']==v],t).items() if k in('repro','flag','prec','rec','acc')}) for t in (0.7,0.8,0.9) for v in ('boot','seed')]
# measures AUC / rho pooled boot, truth modal>=0.8; AUC CI
def auc_of(z,col,sg): 
    rp=(z.t_modal>=.8).astype(int); return roc_auc_score(rp,sg*z[col]) if rp.nunique()==2 else np.nan
meas=[]
for agg in ['sum','mean']:
  for var in ['boot','seed']:
    z=E[(E.aggr==agg)&(E['var']==var)]
    for col,sg,nm in (('srsi',1,'SRSI'),('sd',-1,'Rank SD'),('modal',1,'Modal-rank freq.'),('width',-1,'Rank interval width'),('top3',1,'Top-3 freq.')):
        meas.append(dict(agg=agg,var=var,measure=nm,auc=auc_of(z,col,sg),auc_ci=cl_boot(z,lambda q:auc_of(q,col,sg),200),rho=spearmanr(sg*z[col],z.t_modal)[0]))
out['meas']=meas
# agreement truth sum vs mean
T=E.drop_duplicates(['data','n','feat','aggr'])[['data','n','feat','aggr','t_modal','t_mean']].pivot_table(index=['data','n','feat'],columns='aggr',values='t_modal')
out['trunc_sum_mean_rho']=spearmanr(T['sum'],T['mean'])[0]; out['truth_flag_diff']=float(((T['sum']>=.8)!=(T['mean']>=.8)).mean())
out['D']=E.groupby(['data','n']).apply(lambda z:int(z.t_modal.notna().sum()>0)).to_dict() if False else None
json.dump(out,open(os.path.join(R,'tables_multi.json'),'w'),indent=1,default=float)
pd.set_option('display.width',250)
print(pd.DataFrame(out['main']).round(3).T.to_string()); print(pd.DataFrame(out['by_n']).round(3).to_string()); print(pd.DataFrame(out['thr']).round(3).to_string())
print(pd.DataFrame(out['meas']).round(3).to_string()); print('truth sum vs mean rho',round(out['trunc_sum_mean_rho'],3),'flag diff',round(out['truth_flag_diff'],3))
print(pd.DataFrame(out['by_data']).round(3).to_string())
