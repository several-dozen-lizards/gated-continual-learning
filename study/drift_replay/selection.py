"""Rank confidence loss on operational probes; never inspect final evaluation."""
import math,random
from collections import Counter,defaultdict

def rank_losses(before,after,capacity=4):
    assert set(before)==set(after)
    scores=[]
    for name in before:
        b,a=before[name],after[name]
        assert len(b)==len(a)>0
        assert all(math.isfinite(p) and 0<=p<=1 for p in b+a)
        old=sum(-math.log(max(p,1e-30)) for p in b)/len(b)
        new=sum(-math.log(max(p,1e-30)) for p in a)/len(a)
        scores.append(dict(name=name,before_nll=old,after_nll=new,delta_nll=new-old,score=max(0,new-old)))
    scores.sort(key=lambda row:(-row['score'],row['name']))
    return dict(ranking=scores,selected_names=[r['name'] for r in scores[:capacity]])

def matched_random(claims,selected,cost,seed):
    pools=defaultdict(list)
    for c in sorted(claims,key=lambda c:c['name']):pools[cost(c)].append(c)
    counts=Counter(cost(c) for c in selected);rng=random.Random(seed+19000)
    result=[]
    for key,count in sorted(counts.items()):result+=rng.sample(pools[key],count)
    return sorted(result,key=lambda c:c['name'])
