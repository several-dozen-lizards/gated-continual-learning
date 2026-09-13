"""Verify frozen inputs and archive hashes of completed pilot receipts."""
import hashlib
import json
import math
from pathlib import Path

root=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protocol=json.loads((root/'protocol_v2.json').read_text())
for name,expected in protocol['source_hashes'].items():
    assert sha(root/name)==expected, name
paths=[root/'protocol_v2.json']+[root/name for name in protocol['source_hashes']]
for seed in protocol['seeds']:
    run=root/'runs'/f'test-v2-seed-{seed}'
    report=json.loads((run/'report.json').read_text())
    assert report['status']=='complete' and report['reload_passed']
    config=json.loads((run/'config.json').read_text())
    assert config['model']['revision']==protocol['model_revision']
    assert config['world_sha256']==protocol['world_hashes'][str(seed)]
    assert config['epochs']==protocol['epochs'] and config['lr']==protocol['lr']
    for arm in ('frozen','all','gated','random'):
        result=report['arms'][arm]
        assert all(math.isfinite(loss) for loss in result['training'].get('losses',[]))
        for row in result['evaluation']['rows']:
            assert all(math.isfinite(p) and 0<=p<=1 for p in row['probabilities'])
            assert abs(sum(row['probabilities'])-1)<1e-5
        folder=run/(arm+'_adapter')
        assert (folder/'adapter_model.safetensors').stat().st_size>0
        assert (folder/'adapter_config.json').is_file()
    frozen=sha(run/'frozen_adapter/adapter_model.safetensors')
    assert all(sha(run/(arm+'_adapter/adapter_model.safetensors'))!=frozen for arm in ('all','gated','random'))
    paths.extend(p for p in run.rglob('*') if p.is_file() and p.suffix in ('.json','.safetensors'))
manifest={str(p.relative_to(root)):sha(p) for p in paths}
with (root/'runs/receipt_manifest_v2.json').open('x',encoding='utf-8') as output:
    json.dump(manifest,output,indent=2)
print(f'Frozen inputs, saved adapters, and {len(manifest)} receipt hashes verified.')
