import os
import pandas as pd
for n in ['phoneme','australian','breast_cancer_wisconsin','breast_w','credit_a','diabetes','profb','tokyo1','wdbc']:
    u=f'https://media.githubusercontent.com/media/EpistasisLab/pmlb/master/datasets/{n}/{n}.tsv.gz'
    try: d=pd.read_csv(u,sep='\t',compression='gzip');d.to_pickle(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','data','pmlb',f'{n}.pkl'));print('ok',n,d.shape,flush=True)
    except Exception as e: print('FAIL',n,str(e)[:50],flush=True)
