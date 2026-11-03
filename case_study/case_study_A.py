"""Case study A: domain rank stability (bootstrap audit), documented definitions.
Input: register workbook (sheets MOU, PKS). Outcome: ended = status 'Tidak Aktif'; rows kept: Aktif / Tidak Aktif.
Domains: duration (days between start and end date), year of signing, instrument (MOU/PKS), partner category (10 most frequent + other),
field of cooperation (one binary column per named field), number of scopes (items listed in the scope column), implementation status (3 levels),
implementing unit (home unit vs other), international scope (domestic/foreign). Pipeline as in Section 2.2, B=200 resamples of all N rows."""
import pandas as pd,numpy as np,shap,warnings,re,json,sys,os
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold,cross_val_score
from joblib import Parallel,delayed
PATH=sys.argv[1]; x=pd.ExcelFile(PATH)
def load(s,inst):
    df=x.parse(s,header=None); h=df.index[df.iloc[:,0].astype(str).str.upper().str.startswith('NO')][0]
    d=df.iloc[h+1:].copy(); d.columns=[re.sub(r'\s+',' ',str(c)).strip() for c in df.iloc[h]]; d=d[d['TAHUN'].notna()]
    unit=[c for c in d.columns if 'UNIT' in c.upper() or 'INSTITUSI' in c.upper()][0]; cl=lambda s:s.astype(str).str.strip()
    return pd.DataFrame({'inst':inst,'status':cl(d['STATUS']),'field':cl(d['BIDANG KERJASAMA']),'partner':cl(d['KRITERIA MITRA']),'intl':cl(d['JENIS KERJASAMA']),
      'impl':cl(d['STATUS IMPLEMANTASI']).str.replace('/ ','/').str.replace(' /','/'),'unit':cl(d[unit]),'scope_txt':cl(d['RUANG LINGKUP / PROGRAM KERJASAMA']),
      'year':d['TAHUN'].astype(int),'dur':(pd.to_datetime(d['TANGGAL BERAKHIR'],errors='coerce')-pd.to_datetime(d['TANGGAL MULAI'],errors='coerce')).dt.days})
sheets=x.sheet_names; D=pd.concat([load(sheets[0],'MOU'),load(sheets[1],'PKS')],ignore_index=True); N0=len(D)
D=D[D.status.isin(['Aktif','Tidak Aktif'])].reset_index(drop=True); D['y']=(D.status=='Tidak Aktif').astype(int)
D['dur']=D.dur.fillna(D.dur.median()); D['nscope']=D.scope_txt.map(lambda s:len([t for t in re.split(r'[\n;]+',s) if t.strip()]))
HOME_UNIT=os.environ.get('HOME_UNIT_KEYWORD','').upper()  # keyword of the home unit's code in the unit column (not published)
D['unit_b']=np.where(D.unit.str.upper().str.contains(HOME_UNIT,regex=False),'home','other')
comp=sorted({t.strip() for s in D.field for t in s.split(',') if t.strip()})
FM=pd.DataFrame({'f_'+c:D.field.map(lambda s,c=c:float(c in [t.strip() for t in s.split(',')])) for c in comp})
def cat(col,top=10):
    s=D[col].copy(); s=s.where(s.isin(s.value_counts().index[:top]),'_other'); return pd.get_dummies(s,prefix=col).astype(float)
DOM={'duration':D[['dur']].astype(float),'year':D[['year']].astype(float),'instrument':cat('inst'),'partner':cat('partner'),'field':FM,
     'scopes':D[['nscope']].astype(float),'impl':cat('impl'),'unit':cat('unit_b'),'intl':cat('intl')}
def build(names):
    M=[DOM[g] for g in names]; return pd.concat(M,axis=1).values,np.concatenate([[g]*m.shape[1] for g,m in zip(names,M)])
def model(kind,seed):
    return RandomForestClassifier(100,max_depth=5,random_state=seed,n_jobs=1) if kind=='rf' else GradientBoostingClassifier(n_estimators=100,max_depth=3,learning_rate=0.1,subsample=0.8,random_state=seed)
def one(X,y,grp,names,kind,agg,b):
    rng=np.random.RandomState(1000+b); j=rng.randint(0,len(y),len(y))
    if len(set(y[j]))<2: return None
    m=model(kind,b).fit(X[j],y[j]); sv=shap.TreeExplainer(m).shap_values(X[j]); sv=sv[1] if isinstance(sv,list) else (sv[:,:,1] if sv.ndim==3 else sv)
    s=np.abs(sv).mean(0); f=np.sum if agg=='sum' else np.mean; ds=np.array([f(s[grp==g]) for g in names]); return (-ds).argsort().argsort()+1
def audit(names,kind='rf',agg='sum',B=200):
    X,grp=build(names); y=D.y.values
    R=np.array([r for r in Parallel(n_jobs=4)(delayed(one)(X,y,grp,names,kind,agg,b) for b in range(B)) if r is not None]); k=len(names)
    sdmax=np.sqrt((k*k-1)/12); S=R.sum(0); W=12*((S-S.mean())**2).sum()/(len(R)**2*(k**3-k))
    srsi=np.maximum(0,1-R.std(0)/sdmax); band=['Stable' if s>=.85 else 'Caution' if s>=.6 else 'Unreliable' for s in srsi]
    return dict(names=names,srsi=srsi.round(3).tolist(),band=band,mean=R.mean(0).round(2).tolist(),sd=R.std(0).round(2).tolist(),
                last=[round(float((R[:,i]==k).mean()),3) for i in range(k)],W=round(float(W),3),n=len(R))
def auc(names):
    X,_=build(names); y=D.y.values; r=[]
    for rep in range(10):
        a=cross_val_score(RandomForestClassifier(100,max_depth=5,random_state=rep,n_jobs=1),X,y,cv=RepeatedStratifiedKFold(n_splits=5,n_repeats=1,random_state=rep),scoring='roc_auc'); r.append(a.mean())
    return round(float(np.mean(r)),3),round(float(np.std(r)),3)
if __name__=='__main__':
    four=['duration','instrument','partner','year']; seven=['field','partner','instrument','scopes','impl','unit','intl']; six=[g for g in seven if g!='impl']
    out=dict(records=N0,N=len(D),ended=int(D.y.sum()),pct=round(100*D.y.mean(),1))
    out['auc_all9']=auc(four+seven); out['auc_four']=auc(four); out['auc_seven']=auc(seven); out['auc_six']=auc(six)
    out['four_rf']=audit(four); out['four_gb']=audit(four,'gb'); out['seven_sum']=audit(seven); out['seven_mean']=audit(seven,agg='mean'); out['six_sum']=audit(six)
    json.dump(out,open('results/case_study_A_output.json','w'),indent=1); print('DONE')
