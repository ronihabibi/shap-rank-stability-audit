import numpy as np,pandas as pd,sys
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
d=pd.read_csv(sys.argv[1]); rng=np.random.default_rng(0)
d['R']=d.ranks.map(lambda s:np.array(s.split(';'),float))
def meas(R,k):
    sdmax=np.sqrt((k*k-1)/12); sd=R.std(ddof=1) if len(R)>1 else np.nan
    vals,cnt=np.unique(R,return_counts=True)
    return dict(sd=sd,srsi=max(0,1-sd/sdmax) if len(R)>1 else np.nan,srsi_raw=1-sd/sdmax if len(R)>1 else np.nan,modal=cnt.max()/len(R),top1=(R==1).mean(),top3=(R<=3).mean(),width=np.percentile(R,97.5)-np.percentile(R,2.5))
rows=[]
for (dn,n,agg),g in d.groupby(['data','n','agg']):
    F=g[g['var']=='full']; k=int(g.k.iloc[0])
    tr={}
    for f,h in F.groupby('feat'):
        R=np.array([r[0] for r in h.R]); vals,cnt=np.unique(R,return_counts=True); sd=R.std(ddof=1); tr[f]=dict(t_modal=cnt.max()/len(R),t_sd=sd,t_srsi=max(0,1-sd/np.sqrt((k*k-1)/12)),t_mean=R.mean(),D=len(R))
    for _,r in g[g['var']!='full'].iterrows():
        m=meas(r.R,k); m.update(tr[r.feat]); m.update(data=dn,n=n,aggr=agg,var=r['var'],draw=r.draw,feat=r.feat,k=k); rows.append(m)
E=pd.DataFrame(rows); E['repro']=(E.t_modal>=0.8).astype(int); E.to_csv(sys.argv[2] if len(sys.argv)>2 else sys.argv[1].replace('.csv','_est.csv'),index=False)
def boot_ci(fn,df,B=500):
    keys=df[['data','n','draw']].drop_duplicates().values.tolist(); grp={tuple(k):v for k,v in df.groupby(['data','n','draw'])}
    out=[]
    for _ in range(B):
        idx=rng.integers(0,len(keys),len(keys)); s=pd.concat([grp[tuple(keys[i])] for i in idx]); 
        try: out.append(fn(s))
        except Exception: pass
    return np.percentile(out,[2.5,97.5])
def auc(s,col): return roc_auc_score(s.repro,s[col]) if s.repro.nunique()==2 else np.nan
def rho(s,col): return spearmanr(s[col],s.t_modal)[0]
print('=== Ringkasan (pool semua dataset, n)  baris=',len(E))
res=[]
for agg in ['sum','mean']:
  for var in ['boot','seed']:
    s=E[(E.aggr==agg)&(E['var']==var)]
    row=dict(aggr=agg,var=var,reproducible_share=s.repro.mean(),flagS_srsi=(s.srsi>=.85).mean(),flagS_modal=(s.modal>=.85).mean())
    for col,sg in (('srsi',1),('sd',-1),('modal',1),('top3',1),('width',-1)):
        s2=s.copy(); s2['x']=sg*s[col]; row[f'rho_{col}']=rho(s2,'x'); row[f'auc_{col}']=auc(s2,'x')
    row['bias_srsi']=(s.srsi-s.t_srsi).mean(); row['bias_srsi_se']=(s.srsi-s.t_srsi).std()/np.sqrt(len(s))
    row['acc_srsi85']=((s.srsi>=.85).astype(int)==s.repro).mean(); row['acc_modal85']=((s.modal>=.85).astype(int)==s.repro).mean()
    res.append(row)
P=pd.DataFrame(res); pd.set_option('display.width',250); pd.set_option('display.max_columns',40); print(P.round(3).T.to_string())
print('\n=== Beda SRSI estimasi - SRSI sebenarnya, per dataset/n (agg=sum)')
x=E[E.aggr=='sum'].assign(b=lambda z:z.srsi-z.t_srsi).pivot_table(index=['data','n'],columns='var',values='b',aggfunc='mean'); print(x.round(3).to_string())
print('\n=== AUC per dataset (agg=sum): SRSI|SD|modal|top3 ; boot vs seed')
for (dn,n),s in E[E.aggr=='sum'].groupby(['data','n']):
    line=[]
    for var in ['boot','seed']:
        t=s[s['var']==var]
        line.append(var+': '+' '.join(f"{c}={auc(t.assign(x=sg*t[c]),'x'):.2f}" for c,sg in (('srsi',1),('sd',-1),('modal',1),('top3',1))))
    print(dn,n,'repro share %.2f'%s.repro.mean(),'|',' | '.join(line))
print('\n=== Peringkat truth: sum vs mean, rata-rata t_modal per dataset')
print(E.groupby(['data','n','aggr']).t_modal.mean().unstack().round(3).to_string())
# CI antar-draw untuk selisih boot vs seed pada bias
b=E[(E.aggr=='sum')]
def fn(s): return (s[s['var']=='seed'].eval('srsi-t_srsi').mean()-s[s['var']=='boot'].eval('srsi-t_srsi').mean())
print('\nselisih bias (seed-boot) =',round(fn(b),3),'CI',boot_ci(fn,b).round(3))
for col,sg in (('srsi',1),('sd',-1),('modal',1),('top3',1)):
    def f2(s,col=col,sg=sg):
        a=s[s['var']=='boot']; return auc(a.assign(x=sg*a[col]),'x')-auc(a.assign(x=a['modal']),'x')
    print('AUC boot',col,'-modal:',round(f2(b),3),boot_ci(f2,b,300).round(3))
