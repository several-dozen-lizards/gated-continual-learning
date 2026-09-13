import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
data=json.loads((ROOT/'SUMMARY.json').read_text())
arms=['frozen','none','small','full']
names=['Frozen checkpoint','No replay','Four-fact buffer','Full ledger replay']
fig,axes=plt.subplots(1,3,figsize=(14,5),layout='constrained')
for ax,key,title in zip(axes,['target_accuracy','other_original','accuracy'],['Repaired fact retained','Other original facts retained','All twenty facts correct']):
    ax.barh(names,[100*data['means']['evaluation'][a][key] for a in arms],color=['#64748b','#fb923c','#0891b2','#059669'],alpha=.8)
    for i,arm in enumerate(arms):
        points=[100*r[key] for r in data['records'] if r['arm']==arm and r['wave']==2 and r['family']=='evaluation']
        ax.scatter(points,[i-.12,i,i+.12],s=25,color='#0f172a',zorder=3)
    ax.set_xlim(-1,104);ax.set_xticks([0,25,50,75,100]);ax.invert_yaxis()
    ax.set_title(title,fontsize=12);ax.set_xlabel('Percent • bars: mean; dots: three seeds')
    ax.spines[['top','right']].set_visible(False)
fig.suptitle('Qwen3.5-2B • Stability after two later learning waves',fontsize=16)
fig.savefig(ROOT/'OUTCOMES.png',dpi=180);fig.savefig(ROOT/'OUTCOMES.svg')
