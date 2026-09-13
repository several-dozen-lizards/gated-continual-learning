"""Controlled second-phase evidence; labels remain in the evaluator sidecar."""
import random
from evidence import COLORS


def build_fixture(world):
    claims=[e for e in world['stream'] if e['relevant'] and not e['verified']]
    useful=[c for c in claims if world['evaluator_labels'][c['id']]['useful']]
    target_names={c['name'] for c in useful}
    other_false=[c for c in claims if c['name'] not in target_names]
    events,registry,histories=[],{},{}
    conditions={}

    def source(alias,origin):
        registry[alias]=origin
        if origin not in histories:
            # Audited past tasks, separate from the current world's answer key.
            histories[origin]=[dict(predicted=COLORS[i%4],observed=COLORS[i%4] if i<13 else COLORS[(i+1)%4]) for i in range(16)]
        return alias

    def vote(claim,color,alias,origin):
        events.append(dict(id=f'v{len(events):03}',name=claim['name'],color=color,source=source(alias,origin)))

    for c in useful:
        conditions[c['id']]='legitimate_corroboration'
        for witness in ('a','b'):
            vote(c,c['color'],f"{c['id']}-{witness}",f"independent-{c['id']}-{witness}")
    for idx,c in enumerate(other_false):
        if idx<4:
            conditions[c['id']]='repeated_lineage'
            for copy in range(12):
                vote(c,c['color'],f"{c['id']}-alias-{copy}",f"one-origin-{c['id']}")
        elif idx<8:
            conditions[c['id']]='conflicting_independent_reports'
            vote(c,c['color'],f"{c['id']}-support",f"origin-{c['id']}-a")
            other=COLORS[(COLORS.index(c['color'])+1)%4]
            vote(c,other,f"{c['id']}-oppose",f"origin-{c['id']}-b")
        elif idx==8:
            conditions[c['id']]='coordinated_false_agreement'
            for witness in ('a','b'):
                vote(c,c['color'],f"{c['id']}-wrong-{witness}",f"apparently-independent-{c['id']}-{witness}")
        else:
            conditions[c['id']]='no_new_evidence'
    for c in claims:
        conditions.setdefault(c['id'],'alternative_to_corroborated_claim')
    random.Random(world['seed']+2000).shuffle(events)
    # This sidecar is deliberately never supplied to EvidenceGate.
    evaluator={c['id']:dict(condition=conditions[c['id']],useful=world['evaluator_labels'][c['id']]['useful']) for c in claims}
    return dict(claims=claims,events=events,registry=registry,histories=histories,evaluator=evaluator)


def accepted_ledger(world):
    ledger={n:dict(id='old-'+n,name=n,color=c,relevant=True) for n,c in world['initial'].items()}
    for event in world['stream']:
        if event['relevant'] and event['verified']:
            ledger[event['name']]=event
    return ledger


def update_ledger(ledger, selected):
    result=dict(ledger)
    for claim in selected:
        result[claim['name']]=claim
    return [result[name] for name in sorted(result)]
