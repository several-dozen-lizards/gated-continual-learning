"""Evidence-driven routing. No evaluator answers enter the gate."""
import math
from collections import defaultdict

COLORS=('red','blue','green','gold')


class EvidenceGate:
    def __init__(self, claims, registry, histories, false_update_cost=9, defer_cost=1, collapse_lineage=True):
        self.claims=claims
        self.registry=registry
        self.histories=histories
        self.threshold=false_update_cost/(false_update_cost+defer_cost)
        self.collapse_lineage=collapse_lineage
        self.observations=defaultdict(lambda:defaultdict(set))

    def reliability(self, origin):
        # Beta(1,3) starts at the chance rate for four possible colors.
        history=self.histories.get(origin,[])
        correct=sum(row['predicted']==row['observed'] for row in history)
        alpha=1+correct
        beta=3+len(history)-correct
        return dict(alpha=alpha,beta=beta,mean=alpha/(alpha+beta),audits=len(history))

    def posterior(self, name):
        scores={color:-math.log(len(COLORS)) for color in COLORS}
        for key,votes in self.observations[name].items():
            # Internally contradictory reports from one lineage contribute no vote.
            if len(votes)!=1:
                continue
            source=key if not self.collapse_lineage else None
            origin=self.registry.get(source,source) if source is not None else key
            r=self.reliability(origin)['mean']
            vote=next(iter(votes))
            for color in COLORS:
                scores[color]+=math.log(r if color==vote else (1-r)/(len(COLORS)-1))
        maximum=max(scores.values())
        weights={c:math.exp(v-maximum) for c,v in scores.items()}
        total=sum(weights.values())
        return {c:v/total for c,v in weights.items()}

    def decision(self, claim):
        if not claim['relevant']:
            return dict(route='forgettable',probability=None)
        probability=self.posterior(claim['name'])[claim['color']]
        return dict(route='formative' if probability>=self.threshold else 'informational',probability=probability)

    def ingest(self, event):
        affected=[c for c in self.claims if c['name']==event['name']]
        before={c['id']:self.decision(c) for c in affected}
        origin=self.registry.get(event['source'],'unregistered:'+event['source'])
        if event['color'] is not None:
            if event['color'] not in COLORS:
                raise ValueError('Unknown evidence value')
            key=origin if self.collapse_lineage else event['source']
            self.observations[event['name']][key].add(event['color'])
        after={c['id']:self.decision(c) for c in affected}
        return dict(event_id=event['id'],name=event['name'],origin=origin,
                    reliability=self.reliability(origin),before=before,after=after,
                    posterior=self.posterior(event['name']))


def route_stream(claims, fixture, sham=False, collapse_lineage=True):
    gate=EvidenceGate(claims,fixture['registry'],fixture['histories'],collapse_lineage=collapse_lineage)
    receipts=[]
    for event in fixture['events']:
        receipts.append(gate.ingest(dict(event,color=None) if sham else event))
    decisions={c['id']:gate.decision(c) for c in claims}
    return dict(decisions=decisions,receipts=receipts,
                selected=[c for c in claims if decisions[c['id']]['route']=='formative'])
