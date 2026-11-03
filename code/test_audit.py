import sys,os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, os, time
os.environ['BREP']='40'
import multi as mm   # reuse the study's loader only to build the same domains
from rank_audit import audit_domains, flag_probability
M,y,names,idx=mm.load('german'); rng=np.random.default_rng(5); rows=rng.permutation(len(y))[:300]
t=time.time(); out,R=audit_domains(M[rows],y[rows],idx,names,B=40,seed=1)
print(round(time.time()-t,1),'s'); print(out.head(8).round(3).to_string()); print('flagged Stable (modal>=0.8):',int(out.flag_stable.sum()),'of',len(out))
print('P(reproducible | flag) at prior 0.036, LR+ 25.1 =',round(flag_probability(25.1,0.036),3))
