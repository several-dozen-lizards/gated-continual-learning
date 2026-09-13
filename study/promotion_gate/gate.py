"""Promotion decisions consume operational probe receipts only."""
import hashlib,json,math,os
from pathlib import Path

def decide(baseline,candidate):
    def index(rows):
        result={}
        for row in rows:
            key=(row['name'],row['template'])
            if key in result:raise ValueError('Duplicate probe')
            p=row['probabilities']
            if len(p)!=4 or not all(math.isfinite(v) and 0<=v<=1 for v in p) or abs(sum(p)-1)>1e-5:
                raise ValueError('Invalid probe probabilities')
            colors=('red','blue','green','gold')
            if row['expected'] not in colors or row['prediction']!=colors[max(range(4),key=p.__getitem__)]:
                raise ValueError('Invalid probe answer')
            result[key]=row
        return result
    before,after=index(baseline),index(candidate)
    if not before or before.keys()!=after.keys():raise ValueError('Probe coverage mismatch')
    losses=[];gains=[]
    for key,old in before.items():
        new=after[key]
        if old['expected']!=new['expected']:raise ValueError('Ledger mismatch')
        was=old['prediction']==old['expected'];now=new['prediction']==new['expected']
        if was and not now:losses.append(dict(name=key[0],template=key[1]))
        if not was and now:gains.append(dict(name=key[0],template=key[1]))
    accepted=not losses and bool(gains)
    return dict(promote=accepted,reason='collateral_loss' if losses else ('improvement_without_loss' if gains else 'no_demonstrated_gain'),
        gains=gains,losses=losses)

def choose(decisions):
    return next((name for name in ('drift','random') if decisions[name]['promote']),'baseline')

def activate(path,selection,checkpoints):
    """Commit an experiment-local reference after rechecking immutable weights."""
    checkpoint=checkpoints[selection]
    for file,digest in checkpoint['hashes'].items():
        actual=hashlib.sha256((Path(checkpoint['path'])/file).read_bytes()).hexdigest()
        if actual!=digest:raise ValueError('Checkpoint hash changed')
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    temporary=path.with_suffix('.pending')
    with temporary.open('x',encoding='utf-8') as f:json.dump(dict(selected=selection,checkpoint=checkpoint),f,indent=2)
    os.replace(temporary,path)
