import json,hashlib,random
from pathlib import Path
from datetime import datetime,timezone
from selection import schedule
ROOT=Path(__file__).resolve().parent
PRIOR=ROOT.parent/'repair'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old=json.loads((PRIOR/'plan.json').read_text())
for name,digest in old['source_hashes'].items():assert sha(PRIOR/name)==digest,name
assert not (ROOT/'plan.json').exists()
(ROOT/'fixtures').mkdir(exist_ok=True)
plan=dict(frozen_at=datetime.now(timezone.utc).isoformat(),seeds=old['seeds'],epochs=10,lr=.0002,
    model_revision=old['model_revision'],source_hashes={},fixtures={},prior_receipts={})
names=[['Cedar','Maple','Willow','Birch'],['Aspen','Hazel','Alder','Linden']]
for seed in plan['seeds']:
    run=PRIOR/'runs'/f'seed-{seed}'
    assert json.loads((run/'report.json').read_text())['status']=='complete'
    prior=json.loads((run/'inputs.json').read_text())
    ledger=json.loads((run/'training_claims.json').read_text())['corrected_replay']
    waves=[]
    for wave,sites in enumerate(names,1):
        colors=['red','blue','green','gold'];random.Random(seed+wave*7000).shuffle(colors)
        waves.append([dict(id=f'wave-{wave}-{name}',name=name,color=color,relevant=True) for name,color in zip(sites,colors)])
    assert len({c['name'] for c in ledger+sum(waves,[])})==20
    payload=dict(world=prior['world'],target_name=prior['target_name'],old_wrong_color=prior['old_wrong_color'],
        initial_ledger=ledger,waves=waves,schedules={arm:schedule(ledger,waves,arm,seed) for arm in ['none','small','full']})
    target=ROOT/'fixtures'/f'{seed}.json'
    with target.open('x',encoding='utf-8') as f:json.dump(payload,f,indent=2)
    plan['fixtures'][str(seed)]=sha(target)
    files=['report.json','inputs.json','training_claims.json','corrected_replay_adapter/adapter_config.json','corrected_replay_adapter/adapter_model.safetensors']
    plan['prior_receipts'][str(seed)]={name:sha(run/name) for name in files}
    print(seed,'small buffers',[[c['name'] for c in w['buffer']] for w in payload['schedules']['small']])
for name in ['runner.py','selection.py','PROTOCOL.md','prepare.py']:
    plan['source_hashes'][name]=sha(ROOT/name)
with (ROOT/'plan.json').open('x',encoding='utf-8') as f:json.dump(plan,f,indent=2)
