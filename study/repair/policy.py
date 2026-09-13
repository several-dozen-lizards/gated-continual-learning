"""Rebuild evidence after explicit source withdrawals; no evaluator access."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'reassessment'))
from evidence import EvidenceGate,COLORS

def repair(payload,sham=False):
    fixture=payload['evidence']
    name=payload['target_name']
    candidates=[dict(id='candidate-'+c,name=name,color=c,relevant=True) for c in COLORS]
    registry=dict(fixture['registry'],**payload['new_registry'])
    histories=dict(fixture['histories'],**payload['new_histories'])
    votes=list(fixture['events']);withdrawn=set()
    def rebuild():
        gate=EvidenceGate(candidates,registry,histories)
        for vote in votes:
            if registry.get(vote['source'],vote['source']) not in withdrawn:gate.ingest(vote)
        decisions={c['color']:gate.decision(c) for c in candidates}
        selected=[c for c in candidates if decisions[c['color']]['route']=='formative']
        assert len(selected)<=1
        return decisions,selected
    initial,_=rebuild();receipts=[]
    for event in ([] if sham else payload['repair_events']):
        before,_=rebuild()
        if event['kind']=='withdraw_origin':withdrawn.add(event['origin'])
        elif event['kind']=='vote':votes.append({k:v for k,v in event.items() if k!='kind'})
        else:raise ValueError(event['kind'])
        after,_=rebuild()
        receipts.append(dict(event=event,before=before,after=after))
    decisions,selected=rebuild()
    ledger={c['name']:dict(c) for c in payload['accepted_ledger']}
    # A demoted claim leaves the training ledger, but stays in evidence receipts.
    ledger.pop(name,None)
    for c in selected:ledger[name]=c
    return dict(initial=initial,decisions=decisions,selected=selected,receipts=receipts,
        ledger=[ledger[n] for n in sorted(ledger)])
