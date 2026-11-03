import pickle,numpy as np,pandas as pd,warnings
warnings.filterwarnings('ignore')
r=pickle.load(open('results/sensitivity/res.pkl','rb'))
main=pd.read_csv('results/multi_rank_matrices_all_models.csv'); pm=pd.read_csv('results/pmlb/pmlb_rank_matrices.csv')
def tt(d,grp):
    d=d[(d['var']=='full')&(d['agg']=='sum')].copy(); d['R']=d.ranks.map(lambda s:float(s.split(';')[0]))
    g=d.groupby(['data','n','feat']).R.agg(t3=lambda x:(x<=3).mean(),D='size').reset_index(); g['grp']=grp; return g
T=pd.concat([tt(main,'main'),tt(pm,'pmlb')])
out={}
for var in ('boot','seed'):
    d=r[var].merge(T,on=['data','n','feat','grp'],how='left',suffixes=('','_t'))
    assert d.t3.notna().all(), d.t3.isna().sum()
    d['repro3']=(d.t3>=0.8).astype(int)
    out[var]=d
def pr(y,f):
    tp=(f&y).sum(); fp=(f&~y).sum(); fn=(~f&y).sum(); tn=(~f&~y).sum()
    rec=tp/max(1,tp+fn); fpr=fp/max(1,fp+tn); prec=tp/max(1,tp+fp)
    return dict(n=len(y),prev=y.mean(),flags=int(f.sum()),prec=prec,rec=rec,fpr=fpr,LRp=(rec/fpr if fpr>0 else np.nan))
print('=== Definition 2: top-3 membership reproducible if true top-3 frequency >= 0.8; flag = estimated top-3 freq >= 0.8')
rows=[]
for var,d in out.items():
    for nm,mask in (('all 17',d.n>0),('n=100',d.n==100),('n=300',d.n==300)):
        s=d[mask]; y=s.repro3==1; f=s.top3>=0.8
        rows.append(dict(design=var,subset=nm,**pr(y.values,f.values)))
print(pd.DataFrame(rows).round(3).to_string())
print('\n=== Definition 1 (exact modal>=0.8), same table for comparison')
rows=[]
for var,d in out.items():
    for nm,mask in (('all 17',d.n>0),('n=100',d.n==100),('n=300',d.n==300)):
        s=d[mask]; y=s.repro==1; f=s.modal>=0.8
        rows.append(dict(design=var,subset=nm,**pr(y.values,f.values)))
print(pd.DataFrame(rows).round(3).to_string())
# top-3 on dataset-level: n=100 share of repro3 by ds
b=out['boot']; print('\nrepro3 prevalence by n:',b.groupby('n').repro3.mean().round(3).to_dict(),'| repro(modal)',b.groupby('n').repro.mean().round(3).to_dict())
print('n=100 repro3 by dataset (top):',b[b.n==100].groupby('ds').repro3.mean().sort_values(ascending=False).round(3).head(6).to_dict())
# Conditional bias by true SRSI group
print('\n=== Conditional bias (estimate - true SRSI) by true SRSI group, all 17 datasets')
for var,d in out.items():
    d=d.copy(); d['g']=pd.cut(d.t_srsi,[-0.001,0.4,0.85,1.0001],labels=['<0.40','0.40-0.85','>=0.85'])
    t=d.groupby('g').apply(lambda z:pd.Series(dict(n=len(z),bias=(z.srsi-z.t_srsi).mean(),est=z.srsi.mean(),true=z.t_srsi.mean(),sd=(z.srsi-z.t_srsi).std())))
    print(var); print(t.round(3).to_string())
pickle.dump(out,open('results/sensitivity/sens.pkl','wb'))
