import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
from policy import repair
ROOT=Path(__file__).resolve().parent
PRIOR=ROOT.parent/'reassessment'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old=json.loads((PRIOR/'plan.json').read_text())
for name,digest in old['source_hashes'].items():assert sha(PRIOR/name)==digest,name
assert not (ROOT/'plan.json').exists()
(ROOT/'fixtures').mkdir(exist_ok=True)
plan=dict(frozen_at=datetime.now(timezone.utc).isoformat(),seeds=old['seeds'],epochs=20,lr=.0002,
    model_revision=old['model_revision'],source_hashes={},fixtures={},prior_receipts={})
for seed in plan['seeds']:
    run=PRIOR/'runs'/f'seed-{seed}'
    assert json.loads((run/'report.json').read_text())['status']=='complete'
    payload=json.loads((run/'inputs.json').read_text())
    fixture=payload['evidence'];world=payload['world']
    wrong=next(c for c in fixture['claims'] if fixture['evaluator'][c['id']]['condition']=='coordinated_false_agreement')
    bad_votes=[v for v in fixture['events'] if v['name']==wrong['name']]
    assert len(bad_votes)==2
    events=[dict(kind='withdraw_origin',origin=fixture['registry'][v['source']],reason='controlled audit invalidated supporting report') for v in bad_votes]
    registry={};histories={}
    for witness in ('a','b'):
        origin='replacement-'+witness;registry[origin]=origin
        histories[origin]=next(iter(fixture['histories'].values()))
        events.append(dict(kind='vote',id='repair-'+witness,name=wrong['name'],color=world['final'][wrong['name']],source=origin))
    payload.update(target_name=wrong['name'],old_wrong_color=wrong['color'],repair_events=events,new_registry=registry,new_histories=histories,
        accepted_ledger=json.loads((run/'training_claims.json').read_text())['reassess_replay'])
    result=repair(payload);sham=repair(payload,sham=True)
    assert result['selected'][0]['color']==world['final'][wrong['name']]
    assert sham['selected'][0]['color']==wrong['color']
    assert all(x['route']=='informational' for x in result['receipts'][1]['after'].values())
    assert len(result['ledger'])==len(sham['ledger'])==12
    target=ROOT/'fixtures'/f'{seed}.json'
    with target.open('x',encoding='utf-8') as f:json.dump(payload,f,indent=2)
    plan['fixtures'][str(seed)]=sha(target)
    files=['report.json','inputs.json','training_claims.json','reassess_replay_adapter/adapter_config.json','reassess_replay_adapter/adapter_model.safetensors']
    plan['prior_receipts'][str(seed)]={name:sha(run/name) for name in files}
    print(seed,wrong['name'],wrong['color'],'->',result['selected'][0]['color'])
for name in ['runner.py','policy.py','PROTOCOL.md','prepare.py','../reassessment/evidence.py']:
    plan['source_hashes'][name]=sha(ROOT/name)
with (ROOT/'plan.json').open('x',encoding='utf-8') as f:json.dump(plan,f,indent=2)
