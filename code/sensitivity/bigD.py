import numpy as np,pandas as pd,warnings
warnings.filterwarnings('ignore')
A=pd.read_csv('results/multi_rank_matrices_all_models.csv'); B=pd.read_csv('results/sensitivity/multi_bigD_rank_matrices.csv')
A=A[(A.n==100)&A.data.isin(['adult','diamonds','telco'])&(A['agg']=='sum')]
d=pd.concat([A,B[B['agg']=='sum']]); d=d[d.n==100]
d['R']=d.ranks.map(lambda s:np.array(s.split(';'),float))
def modal(R): _,c=np.unique(R,return_counts=True); return c.max()/len(R)
def metr(y,f):
    tp=(f&y).sum();fp=(f&~y).sum();fn=(~f&y).sum();tn=(~f&~y).sum()
    rec=tp/max(1,tp+fn);fpr=fp/max(1,fp+tn);prec=tp/max(1,tp+fp)
    return tp,fp,fn,tn
rows=[]
for ds,g in d.groupby('data'):
    full=g[g['var']=='full']; draws=sorted(full.draw.unique()); D=len(draws); feats=sorted(full.feat.unique())
    fr=pd.DataFrame({f:{dr:full[(full.feat==f)&(full.draw==dr)].R.iloc[0][0] for dr in draws} for f in feats})  # draw x feat true ranks
    est={}
    for var in ('boot','seed'):
        e=g[g['var']==var]; est[var]={(r.draw,r.feat):modal(r.R) for r in e.itertuples()}
    def evaluate(block):
        tr={f:modal(fr.loc[block,f].values) for f in feats}
        out={}
        for var in ('boot','seed'):
            tp=fp=fn=tn=0
            for dr in block:
                for f in feats:
                    y=tr[f]>=0.8; fl=est[var][(dr,f)]>=0.8
                    tp+=fl and y; fp+=fl and not y; fn+=(not fl) and y; tn+=(not fl) and (not y)
            out[var]=(tp,fp,fn,tn)
        return out
    cfgs={'full D=%d'%D:[draws]}
    if D>=60: cfgs['D=30 blocks']=[draws[i:i+30] for i in range(0,D-D%30,30)]
    cfgs['D=10 blocks']=[draws[i:i+10] for i in range(0,D-D%10,10)]
    for nm,blocks in cfgs.items():
        tot={'boot':np.zeros(4),'seed':np.zeros(4)}; prevs=[]
        for b in blocks:
            o=evaluate(b)
            for v in o: tot[v]+=np.array(o[v])
        for v in ('boot','seed'):
            tp,fp,fn,tn=tot[v]; n=tp+fp+fn+tn
            rows.append(dict(data=ds,ref=nm,design=v,pairs=int(n),prev=(tp+fn)/n,flags=int(tp+fp),prec=tp/max(1,tp+fp),rec=tp/max(1,tp+fn),fpr=fp/max(1,fp+tn),LRp=(tp/max(1,tp+fn))/(fp/max(1,fp+tn)) if fp>0 else np.nan))
R=pd.DataFrame(rows); pd.set_option('display.width',200)
print(R.round(3).to_string())
# pooled over the three datasets
print('\nPOOLED over adult+diamonds+telco, n=100')
for ref in ['full','D=30 blocks','D=10 blocks']:
    s=R[R.ref.str.startswith(ref)]
    for v in ('boot','seed'):
        t=s[s.design==v]; n=t.pairs.sum(); prev=(t.prev*t.pairs).sum()/n; fl=t.flags.sum(); tp=(t.prec*t.flags).sum(); pos=(t.prev*t.pairs).sum(); rec=tp/pos
        fp=fl-tp; neg=n-pos; fpr=fp/neg
        print(f'{ref:12s} {v:5s} pairs {int(n):5d} prev {prev:.3f} flags {int(fl):4d} prec {tp/fl if fl else float("nan"):.3f} rec {rec:.3f} fpr {fpr:.3f} LR+ {rec/fpr if fpr>0 else float("nan"):.1f}')
R.to_csv('results/sensitivity/bigD_res.csv',index=False)
