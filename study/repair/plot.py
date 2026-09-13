import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
data=json.loads((ROOT/'SUMMARY.json').read_text())
arms=['baseline','correction_only','corrected_replay','unchanged_replay']
names=['Mistaken checkpoint','Correction only','Corrected ledger replay','Unchanged ledger replay']
fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
for row,family in enumerate(['evaluation','transfer']):
    for col,(key,title) in enumerate([('target_accuracy','False admission corrected'),('other_accuracy','Other 11 facts retained')]):
        ax=axes[row,col]
        ax.barh(names,[100*data['means'][family][a][key] for a in arms],color=['#64748b','#0891b2','#059669','#a78bfa'],alpha=.8)
        for i,arm in enumerate(arms):
            points=[100*r[key] for r in data['records'] if r['arm']==arm and r['family']==family]
            ax.scatter(points,[i-.12,i,i+.12],s=24,color='#0f172a',zorder=3)
        ax.set_xlim(-1,104);ax.set_xticks([0,25,50,75,100]);ax.invert_yaxis()
        ax.set_title(title+'\n'+('Original held-out prompts' if row==0 else 'Four additional phrasings'),fontsize=12)
        ax.set_xlabel('Percent • bars: mean; dots: three seeds')
        ax.spines[['top','right']].set_visible(False)
fig.suptitle('Qwen3.5-2B • Repairing a learned false claim',fontsize=17)
fig.savefig(ROOT/'OUTCOMES.png',dpi=180)
fig.savefig(ROOT/'OUTCOMES.svg')
