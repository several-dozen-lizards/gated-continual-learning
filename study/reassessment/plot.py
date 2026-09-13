import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
data=json.loads((ROOT/'SUMMARY.json').read_text())
arms=['baseline','reassess','random','reassess_replay','random_replay']
names=['Prior checkpoint','Reassess','Matched random','Reassess + replay','Random + replay']
fig,axes=plt.subplots(1,3,figsize=(14,5),layout='constrained')
colors=['#64748b','#0891b2','#a78bfa','#059669','#a78bfa']
for ax,key,title in zip(axes,['accuracy','recovered','false_adoption'],['Overall accuracy ↑','Previously missed facts recovered ↑','Corroborated false claim adopted ↓']):
    ax.barh(names,[100*data['means'][a][key] for a in arms],color=colors,alpha=.8)
    for i,arm in enumerate(arms):
        points=[100*r[key] for r in data['records'] if r['arm']==arm]
        ax.scatter(points,[i-.12,i,i+.12],s=24,color='#0f172a',zorder=3)
    ax.set_xlim(0,105);ax.invert_yaxis();ax.set_title(title,fontsize=11,pad=15)
    ax.set_xlabel('Percent (bars: mean; dots: three seeds)')
    ax.spines[['top','right']].set_visible(False)
fig.suptitle('Qwen3.5-2B • Evidence reassessment and accepted-fact replay',fontsize=16)
fig.savefig(ROOT/'OUTCOMES.png',dpi=180)
fig.savefig(ROOT/'OUTCOMES.svg')
