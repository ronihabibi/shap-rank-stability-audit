import numpy as np,pandas as pd,matplotlib,sys,os; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R=sys.argv[1] if len(sys.argv)>1 else os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','results')
E=pd.read_csv(os.path.join(R,'multi_estimates_vs_truth.csv')); S=E[E.aggr=='sum']
G=S.groupby(['data','n','feat','var']).agg(est=('srsi','mean'),tru=('t_srsi','first')).reset_index()
plt.rcParams.update({'font.family':'serif','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(1,2,figsize=(7.2,3.5),sharex=True,sharey=True)
col={'boot':'#1f5fa8','seed':'#c8661a'}; ttl={'boot':'(a) Bootstrap audit','seed':'(b) Seed-only audit'}
for a,v in zip(ax,['boot','seed']):
    g=G[G['var']==v]
    a.scatter(g.tru,g.est,s=14,color=col[v],alpha=.55,edgecolor='white',linewidth=.4)
    a.plot([0,1],[0,1],color='#555',lw=1,ls='--'); a.axhline(.85,color='#aaa',lw=.7); a.axvline(.8,color='#aaa',lw=.7,alpha=0)
    b=(g.est-g.tru).mean(); a.text(.03,.93,f"mean bias {b:+.2f}",transform=a.transAxes,fontsize=8.5)
    a.set_title(ttl[v],fontsize=9.5,loc='left'); a.set_xlim(0,1.02); a.set_ylim(0,1.02); a.set_xlabel('True SRSI (independent samples)')
ax[0].set_ylabel('Estimated SRSI (single sample)')
ax[1].text(.40,.06,'dashed line: estimate = truth',transform=ax[1].transAxes,fontsize=7.5,color='#555')
plt.tight_layout(); plt.savefig(os.path.join(R,'fig_cal.png'),dpi=400); print(len(G))
