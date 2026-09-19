"""Recompute descriptive scale results from archived per-run receipts; no GPU."""
import json,statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parent/'framework_evidence'
def collect():
    rows=[]
    for phase in ['scale_comparison','scale_comparison_v2']:
        for size in ['0.5b','2b','7b']:
            for arm in ['ungated','gated']:
                reports=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((ROOT/phase/'runs'/f'{arm}_{size}').glob('seed_*/report.json'))]
                assert sorted(r['seed'] for r in reports)==[17,29,43]
                scores=[r['evaluation']['accuracy']*100 for r in reports]
                row=dict(phase=phase,model=reports[0]['model']['name'],arm=arm,
                    seeds=[r['seed'] for r in reports],accuracy_percent=scores,
                    mean_percent=statistics.mean(scores),range_pp=max(scores)-min(scores),
                    sample_sd_pp=statistics.stdev(scores),events=[r['events_trained'] for r in reports],
                    stream_input_tokens=[r['stream_training']['input_tokens'] for r in reports])
                if phase.endswith('_v2'):
                    assert all(r['v2_params']=={'lr':5e-5,'epochs':5} for r in reports)
                    row['canary']=[dict(seed=r['seed'],baseline=r['canary']['pre_training']['score'],
                        minimum=min(x['canary']['score'] for t in ['initial_training','stream_training'] for x in r[t]['epoch_checkpoints']),
                        final=r['canary']['final']['score']) for r in reports]
                rows.append(row)
    return rows

if __name__=='__main__':
    print(json.dumps(collect(),indent=2))
