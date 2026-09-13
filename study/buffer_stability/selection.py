"""Outcome-blind bounded replay selection from accepted records."""
import random

def select(ledger,size,seed,wave):
    records=sorted(ledger,key=lambda c:c['name'])
    assert len({c['name'] for c in records})==len(records)
    return sorted(random.Random(seed*1000+wave).sample(records,min(size,len(records))),key=lambda c:c['name'])

def schedule(initial,waves,arm,seed):
    ledger={c['name']:dict(c) for c in initial};result=[]
    for wave,updates in enumerate(waves,1):
        if arm=='none':buffer=[]
        elif arm=='small':buffer=select(list(ledger.values()),4,seed,wave)
        elif arm=='full':buffer=sorted(ledger.values(),key=lambda c:c['name'])
        else:raise ValueError(arm)
        assert not ({c['name'] for c in updates}&set(ledger))
        result.append(dict(wave=wave,updates=updates,buffer=buffer,training=updates+buffer))
        for c in updates:ledger[c['name']]=c
    return result
