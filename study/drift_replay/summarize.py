"""Audit matched replay budgets, selection provenance and recovery."""
import json,hashlib,statistics,math
from pathlib import Path
from selection import rank_losses
ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan=json.loads((ROOT/'plan.json').read_text())
for file,digest in plan['source_hashes'].items():assert sha(ROOT/file)==digest,file
records=[];selections=[];manifest={};deltas=[]
for seed in plan['seeds']:
    run=ROOT/'runs'/f'seed-{seed}'
    report=json.loads((run/'report.json').read_text())
    assert report['status']=='complete'
    for key in ['before_reload_delta','prior_reload_delta','prior_transfer_reload_delta','final_reload_delta','transfer_reload_delta']:
        assert report[key]<=1e-5;deltas.append(report[key])
    p=json.loads((run/'inputs.json').read_text())
    assert sha(ROOT/'fixtures'/f'{seed}.json')==plan['fixtures'][str(seed)]
    assert p==json.loads((ROOT/'fixtures'/f'{seed}.json').read_text())
    assert json.loads((run/'plan.json').read_text())==plan
    for file,digest in plan['prior_receipts'][str(seed)].items():
        assert sha(ROOT.parent/'buffer_stability'/'runs'/f'seed-{seed}'/file)==digest,file
    selection=json.loads((run/'selection.json').read_text())
    assert rank_losses(selection['before']['probabilities'],selection['after']['probabilities'])==selection['ranking']
    assert {c['name'] for c in selection['selected']}==set(selection['ranking']['selected_names'])
    claims=json.loads((run/'training_claims.json').read_text())
    assert claims['drift']==p['latest']+selection['selected']
    assert claims['random']==p['latest']+selection['random']
    for key in ['steps','examples','input_tokens','supervised_tokens','padded_tokens']:
        assert report['arms']['drift']['training'][key]==report['arms']['random']['training'][key],key
    bad_names={r['name'] for r in report['baseline']['rows'] if not r['correct']}
    selected_names={c['name'] for c in selection['selected']}
    random_names={c['name'] for c in selection['random']}
    selections.append(dict(seed=seed,drift=sorted(selected_names),random=sorted(random_names),overlap=sorted(selected_names&random_names),
        observed_errors=sorted(bad_names),drift_error_coverage=sorted(selected_names&bad_names),random_error_coverage=sorted(random_names&bad_names),
        probe_seconds=report['probe_seconds'],probe_tokens=selection['before']['input_tokens']+selection['after']['input_tokens'],
        ranking=selection['ranking']['ranking']))
    baseline_by_family=dict(evaluation=report['baseline'],transfer=report['baseline_transfer'])
    for arm,result in dict(baseline=baseline_by_family,**report['arms']).items():
        training=result.get('training',{})
        assert all(math.isfinite(x) for x in training.get('losses',[]))
        if training:assert training['steps']==40 and training['examples']==training['supervised_tokens']==160
        for family in ['evaluation','transfer']:
            rows=result[family]['rows'];base=baseline_by_family[family]['rows']
            assert len(rows)==(40 if family=='evaluation' else 80)
            assert [(r['name'],r['template'],r['answer']) for r in rows]==[(r['name'],r['template'],r['answer']) for r in base]
            assert all(math.isfinite(v) for r in rows for v in r['probabilities'])
            assert all(abs(sum(r['probabilities'])-1)<1e-5 for r in rows)
            subset=lambda names:statistics.mean(r['correct'] for r in rows if r['name'] in names)
            target=[r for r in rows if r['name']==p['target_name']]
            failed=[r['correct'] for r,b in zip(rows,base) if not b['correct']]
            kept=[r['correct'] for r,b in zip(rows,base) if b['correct']]
            records.append(dict(seed=seed,arm=arm,family=family,accuracy=statistics.mean(r['correct'] for r in rows),
                recovered_errors=statistics.mean(failed),collateral_retention=statistics.mean(kept),
                target_accuracy=statistics.mean(r['correct'] for r in target),old_error=statistics.mean(r['prediction']==p['old_wrong_color'] for r in target),
                other_original=subset(set(p['original_names'])-{p['target_name']}),
                wave1=subset({c['name'] for c in p['candidates']}-set(p['original_names'])),latest=subset({c['name'] for c in p['latest']}),
                train_seconds=training.get('seconds',0),input_tokens=training.get('input_tokens',0),steps=training.get('steps',0)))
    for file in run.rglob('*'):
        if file.is_file():manifest[str(file.relative_to(ROOT))]=sha(file)
arms=['baseline','drift','random']
means={family:{arm:{k:statistics.mean(r[k] for r in records if r['arm']==arm and r['family']==family)
    for k in records[0] if k not in ('seed','arm','family')} for arm in arms} for family in ['evaluation','transfer']}
summary=dict(records=records,means=means,selections=selections,max_reload_delta=max(deltas),receipt_count=len(manifest))
(ROOT/'SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
(ROOT/'runs'/'receipt_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
lines=['# Qwen3.5-2B: replay selected by measured forgetting','',
    'Three paired recovery trials from the affected small-buffer endpoints. Operational probes rank the increase in loss on sixteen accepted facts. Drift and random arms each replay four candidate facts plus the four latest facts at identical training budgets. Means are across seeds 17,29,43.','',
    'The operational probes included every fact with an observed baseline error in the selected set, and targeted replay recovered all prior incorrect answers on both evaluation sets. It nevertheless created new errors. Overall accuracy was 96.7% versus 98.3% for matched random replay on the original prompts, and 95.4% versus 99.6% on the additional phrasings. These observations do not support a performance advantage for the present targeted policy.',
    '', 'In seed 29, targeted replay recovered the four lost gold-valued facts but damaged the previously repaired Opal fact. In seed 43 it recovered the original losses but damaged first-wave knowledge. Random replay restored some facts it never rehearsed directly. This suggests that an incorrect answer need not mean a fact has been irreversibly erased; this experiment does not identify the internal mechanism.',
    '', 'Both arms used 3,160 input tokens and 40 optimizer steps per seed. Targeted selection additionally used 64 probe forward passes (1,392 input tokens), taking about 15.6 seconds on average, apart from loading the before checkpoint. Training averaged about 55.5 seconds for targeted replay and 54.3 seconds for random. Selection by confidence loss worked as a diagnostic but did not earn an end-to-end efficiency benefit here.',
    '', 'The next useful safeguard to test is checking a candidate update for collateral losses before promoting its weights, with promotion probes separated from final evaluation. That safeguard is proposed, not validated by this trial.','']
for family,title in [('evaluation','Original evaluation phrasings'),('transfer','Four additional evaluation phrasings')]:
    lines+=['## '+title,'','| Arm | Overall | Prior errors recovered | Previously correct answers retained | Repaired target | Latest facts |','|---|---:|---:|---:|---:|---:|']
    for arm in arms:
        m=means[family][arm]
        lines.append('| '+arm+' | '+' | '.join(f'{100*m[k]:.1f}%' for k in ['accuracy','recovered_errors','collateral_retention','target_accuracy','latest'])+' |')
    lines+=['']
lines+=['## Per-seed overall accuracy','','| Seed | Family | Affected checkpoint | Drift replay | Matched random |','|---|---|---:|---:|---:|']
for seed in plan['seeds']:
    for family in ['evaluation','transfer']:
        lines.append(f'| {seed} | {family} | '+' | '.join(f"{100*next(r['accuracy'] for r in records if r['seed']==seed and r['family']==family and r['arm']==arm):.1f}%" for arm in arms)+' |')
lines+=['','## Selection receipts','','| Seed | Drift selection | Random selection | Observed baseline errors | Probe seconds |','|---|---|---|---|---:|']
for s in selections:lines.append(f"| {s['seed']} | {', '.join(s['drift'])} | {', '.join(s['random'])} | {', '.join(s['observed_errors'])} | {s['probe_seconds']:.2f} |")
lines+=['','## Cost and scope','',
    'Ranking uses two operational probe templates that are disjoint from training and final evaluation. Probe answers come from the accepted ledger. Baseline errors listed above are a post-selection diagnostic; they do not enter ranking. This confidence-loss measure is not a truth detector.',
    '', 'The before/after probes and their input tokens are recorded in SUMMARY.json and selection receipts. The random policy would not need those probes or a before-checkpoint reload in deployment. Exact training-budget matching establishes a content-selection comparison, not end-to-end cost superiority. No energy or FLOP saving is claimed.',
    '', 'Frozen sources, fixtures and checkpoint hashes passed. Before and affected checkpoint evaluation reloads passed; final drift-adapter reload passed both prompt sets. All recorded losses/probabilities were finite and every paired training budget field matched. Every arm resets to the same affected weights.',
    '', 'This is an immediate recovery experiment on previously observed failures with twenty fictional four-color facts. There is no further interference wave, learned truth adjudication, independent test cohort, optimized replay capacity, or proof of preventive online adaptation. Known evaluation phrasings are reused. No resident state was changed.']
lines+=['','| Arm | Mean training tokens | Steps | Mean training seconds |','|---|---:|---:|---:|']
for arm in ['drift','random']:
    m=means['evaluation'][arm];lines.append(f"| {arm} | {m['input_tokens']:.0f} | {m['steps']:.0f} | {m['train_seconds']:.2f} |")
(ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(dict(means=means,selections=[{k:v for k,v in s.items() if k!='ranking'} for s in selections]),indent=2))
