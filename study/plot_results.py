"""Export a static scientific quality/cost plot from a completed pilot summary."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('directory',type=Path)
parser.add_argument('--title',required=True)
args=parser.parse_args()
data=json.loads((args.directory/'runs/summary_v2.json').read_text())
fig,ax=plt.subplots(figsize=(8,5),dpi=180)
palette={'frozen':'#727b86','all':'#c15749','gated':'#167d69','random':'#8b67a6'}
labels={'frozen':'Frozen after initial learning','all':'All stream','gated':'Gated','random':'Matched random'}
for arm,entry in data['arms'].items():
    x=entry['operational_seconds_mean']
    y=100*entry['accuracy_mean']
    low=100*min(entry['accuracy_by_seed'])
    high=100*max(entry['accuracy_by_seed'])
    ax.errorbar(x,y,yerr=[[max(0,y-low)],[max(0,high-y)]],fmt='o',
        color=palette[arm],markersize=8,capsize=4,lw=1.5)
    offset=(8,-18) if arm=='frozen' else (8,10)
    ax.annotate(labels[arm],(x,y),xytext=offset,textcoords='offset points',fontsize=10,color=palette[arm])
ax.set(xlabel='Mean operational wall time per seed (seconds)',ylabel='Mean factual accuracy (%)',
       ylim=(0,105),title=args.title)
maximum=max(e['operational_seconds_mean'] for e in data['arms'].values())
ax.set_xlim(-0.05*maximum,maximum*1.35)
ax.grid(axis='y',alpha=.2)
ax.spines[['top','right']].set_visible(False)
fig.text(.12,.025,'Bars show paired-seed ranges, not confidence intervals. Three synthetic seeds.\nTime includes training and selection/storage; excludes common setup and evaluation.',fontsize=8,color='#555555')
fig.tight_layout(rect=(0,.1,1,1))
fig.savefig(args.directory/'QUALITY_COST.png',bbox_inches='tight')
fig.savefig(args.directory/'QUALITY_COST.svg',bbox_inches='tight')
print(args.directory/'QUALITY_COST.png')
