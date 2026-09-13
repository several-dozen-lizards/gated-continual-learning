"""Freeze a passing development configuration before test evaluation."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from world import digest, make_world

root=Path(__file__).resolve().parent
development=root/'runs/dev-001'
report=json.loads((development/'report.json').read_text())
config=json.loads((development/'config.json').read_text())
assert report['development_passed'] and report['status']=='complete'
seeds=[17,29,43]
protocol=dict(version=2,frozen_at=datetime.now(timezone.utc).isoformat(),
    development_passed=True,development_report_sha256=digest(report),
    epochs=config['epochs'],lr=config['lr'],seeds=seeds,
    model_revision=config['model']['revision'],quality_loss_margin=0.05,
    source_hashes={name:hashlib.sha256((root/name).read_bytes()).hexdigest()
                   for name in ['world.py','benchmark.py','PROTOCOL_V2.md']},
    world_hashes={str(seed):digest(make_world('test',seed)) for seed in seeds})
with (root/'protocol_v2.json').open('x',encoding='utf-8') as output:
    json.dump(protocol,output,indent=2)
print('Protocol frozen:',digest(protocol))
