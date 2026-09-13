import json,hashlib,copy
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent
PRIOR=ROOT.parent/'buffer_stability'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old=json.loads((PRIOR/'plan.json').read_text())
for name,digest in old['source_hashes'].items():assert sha(PRIOR/name)==digest,name
assert not (ROOT/'plan.json').exists()
(ROOT/'fixtures').mkdir(exist_ok=True)
plan=dict(frozen_at=datetime.now(timezone.utc).isoformat(),seeds=old['seeds'],epochs=10,lr=.0002,
    model_revision=old['model_revision'],source_hashes={},fixtures={},prior_receipts={})
for seed in plan['seeds']:
    run=PRIOR/'runs'/f'seed-{seed}'
    assert json.loads((run/'report.json').read_text())['status']=='complete'
    p=json.loads((run/'inputs.json').read_text())
    before=copy.deepcopy(p['world']);final=copy.deepcopy(p['world'])
    for wave,updates in enumerate(p['waves'],1):
        for c in updates:
            final['final'][c['name']]=c['color'];final['groups'][c['name']]='wave'+str(wave)
            if wave==1:
                before['final'][c['name']]=c['color'];before['groups'][c['name']]='wave1'
    payload=dict(world=final,before_world=before,target_name=p['target_name'],old_wrong_color=p['old_wrong_color'],
        original_names=[c['name'] for c in p['initial_ledger']],candidates=p['initial_ledger']+p['waves'][0],latest=p['waves'][1])
    assert len(payload['candidates'])==16 and len(final['final'])==20
    target=ROOT/'fixtures'/f'{seed}.json'
    with target.open('x',encoding='utf-8') as f:json.dump(payload,f,indent=2)
    plan['fixtures'][str(seed)]=sha(target)
    files=['report.json','inputs.json','training_claims.json']
    for wave in (1,2):files += [f'small_wave{wave}_adapter/adapter_config.json',f'small_wave{wave}_adapter/adapter_model.safetensors']
    plan['prior_receipts'][str(seed)]={name:sha(run/name) for name in files}
for name in ['runner.py','selection.py','PROTOCOL.md','prepare.py']:
    plan['source_hashes'][name]=sha(ROOT/name)
with (ROOT/'plan.json').open('x',encoding='utf-8') as f:json.dump(plan,f,indent=2)
print('Frozen seeds',plan['seeds'])
