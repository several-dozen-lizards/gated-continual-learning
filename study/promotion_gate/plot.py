import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
data=json.loads((ROOT/'SUMMARY.json').read_text())
policies=['baseline','always_drift','guarded_drift','always_random','guarded_fallback']
names=['Prior weights','Always targeted','Guarded targeted','Always random','Guarded fallback']
fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
for ax,key,title in zip(axes,['accuracy','collateral_retention'],['Overall accuracy after selection','Previously correct answers retained']):
    ax.barh(names,[100*data['means']['evaluation'][p][key] for p in policies],color=['#64748b','#fb923c','#0891b2','#a78bfa','#059669'],alpha=.8)
    for i,policy in enumerate(policies):
        points=[100*r[key] for r in data['records'] if r['policy']==policy and r['family']=='evaluation']
        ax.scatter(points,[i-.12,i,i+.12],s=25,color='#0f172a',zorder=3)
    ax.set_xlim(-1,104);ax.set_xticks([0,25,50,75,100]);ax.invert_yaxis()
    ax.set_title(title,fontsize=12);ax.set_xlabel('Percent • bars: mean; dots: three seeds')
    ax.spines[['top','right']].set_visible(False)
fig.suptitle('Qwen3.5-2B • Check candidates before adopting their weights',fontsize=15)
fig.savefig(ROOT/'OUTCOMES.png',dpi=180);fig.savefig(ROOT/'OUTCOMES.svg')
