# Studi multi-dataset: kehandalan peringkat SHAP domain; audit bootstrap vs audit seed-saja; ground truth dari sampel independen (disjoint).
import numpy as np,pandas as pd,sys,time,warnings,os
warnings.filterwarnings('ignore')
import shap
from sklearn.ensemble import RandomForestClassifier
from multiprocessing import Pool
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.abspath(os.path.join(HERE,'..','..')); D=os.path.join(ROOT,'data','pmlb')+os.sep; B=int(os.environ.get('BREP','40')); MODEL=os.environ.get('MODEL','rf')
def cap(s,k=10):
    top=s.value_counts().index[:k]; return s.where(s.isin(top),'other')
def load(name):
    d=pd.read_pickle(D+f'{name}.pkl');y=d.pop('target').values
    vc=pd.Series(y).value_counts().index[0];y=(y==vc).astype(int)
    X=d.values.astype(float);keep=X.std(0)>0;X=X[:,keep];names=[f'f{i}' for i in range(X.shape[1])]
    if len(y)>20000:
        rs=np.random.default_rng(0);i=np.sort(rs.choice(len(y),20000,replace=False));X,y=X[i],y[i]
    return X,y,names,[np.array([j]) for j in range(X.shape[1])]
def ranks(M,y,idx,seed,rows):
    if MODEL=='lr':
        from sklearn.linear_model import LogisticRegression
        Z=M[rows]; mu=Z.mean(0); sd=Z.std(0); sd[sd==0]=1; Zs=(Z-mu)/sd
        lr=LogisticRegression(C=1.0,max_iter=1000,random_state=seed).fit(Zs,y[rows])
        imp=np.abs(lr.coef_[0])*np.abs(Zs-Zs.mean(0)).mean(0); out=[]
    else:
        rf=RandomForestClassifier(n_estimators=100,max_depth=5,random_state=seed,n_jobs=1).fit(M[rows],y[rows])
        sv=shap.TreeExplainer(rf).shap_values(M[rows]); sv=sv[1] if isinstance(sv,list) else (sv[...,1] if sv.ndim==3 else sv)
        imp=np.abs(sv).mean(0); out=[]
    for agg in (np.sum,np.mean):
        sc=np.array([agg(imp[c]) for c in idx]); o=np.argsort(-sc,kind='stable'); r=np.empty(len(idx),int); r[o]=np.arange(1,len(idx)+1); out.append(r)
    return out
def run(a):
    name,n,d,perm=a; M,y,names,idx=load(name); k=len(names)
    rows=perm[d*n:(d+1)*n]; rng=np.random.default_rng(777+d); res=[]
    full=ranks(M,y,idx,42,rows)
    boot=[ranks(M,y,idx,42+b,rows[rng.integers(0,n,n)]) for b in range(B)]
    seed=[ranks(M,y,idx,1000+b,rows) for b in range(B)]
    out=[]
    for ai,agg in enumerate(['sum','mean']):
        for var,L in (('full',[full]),('boot',boot),('seed',seed)):
            R=np.array([l[ai] for l in L])
            for j,f in enumerate(names): out.append((name,n,d,agg,var,f,k,';'.join(map(str,R[:,j]))))
    return out
if __name__=='__main__':
    sets=sys.argv[1].split(','); Dn=int(sys.argv[2]); out=sys.argv[3]; ns=[int(x) for x in sys.argv[4].split(',')]
    jobs=[]
    for s in sets:
        M,y,names,idx=load(s); N=len(y); perm=np.random.default_rng(31).permutation(N)
        for n in ns:
            dd=min(Dn,N//n)
            if dd>=10: jobs+=[(s,n,d,perm) for d in range(dd)]
    print(len(jobs),'jobs',flush=True); t=time.time(); allr=[]
    with Pool(2) as p:
        for k,r in enumerate(p.imap_unordered(run,jobs)):
            allr+=r
            if (k+1)%6==0 or k+1==len(jobs):
                pd.DataFrame(allr,columns=['data','n','draw','agg','var','feat','k','ranks']).to_csv(out,index=False); print(k+1,len(jobs),round(time.time()-t),flush=True)
