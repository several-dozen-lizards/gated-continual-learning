"""Summarize the frozen paired pilot without treating paraphrases as replicates."""
import json
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parent
protocol=json.loads((ROOT/'protocol_v2.json').read_text())
reports=[]
worlds=[]
for seed in protocol['seeds']:
    directory=ROOT/'runs'/f'test-v2-seed-{seed}'
    report=json.loads((directory/'report.json').read_text())
    if report['status']!='complete' or not report['reload_passed']:
        raise RuntimeError(f'Seed {seed} has not passed all execution checks.')
    for key in ('steps','examples','input_tokens','supervised_tokens','padded_tokens'):
        assert report['arms']['gated']['training'][key]==report['arms']['random']['training'][key]
    reports.append(report)
    worlds.append(json.loads((directory/'world.json').read_text()))


def operational(report,arm):
    value=report['arms'][arm]['training']['seconds']
    if arm=='gated':
        value+=report['gate_seconds']+report['storage_seconds']
    if arm=='random':
        value+=report['random_selection_seconds']
    return value


def mean(values):
    return statistics.mean(values)


def percent(value):
    return f'{100*value:.1f}%'


summary={'seeds':protocol['seeds'],'arms':{},'paired_differences':{},'validations':{}}
for arm in ('frozen','all','gated','random'):
    summary['arms'][arm]={
        'accuracy_mean':mean(r['arms'][arm]['evaluation']['accuracy'] for r in reports),
        'accuracy_by_seed':[r['arms'][arm]['evaluation']['accuracy'] for r in reports],
        'group_accuracy':{g:mean(r['arms'][arm]['evaluation']['group_accuracy'][g] for r in reports) for g in ('retention','correction','new')},
        'brier_mean':mean(r['arms'][arm]['evaluation']['brier'] for r in reports),
        'operational_seconds_mean':mean(operational(r,arm) for r in reports),
        'arm_wall_seconds_mean':mean(r['arms'][arm]['arm_wall_seconds'] + (r['gate_seconds']+r['storage_seconds'] if arm=='gated' else r['random_selection_seconds'] if arm=='random' else 0) for r in reports),
        'training_steps_by_seed':[r['arms'][arm]['training']['steps'] for r in reports],
        'training_input_tokens_by_seed':[r['arms'][arm]['training']['input_tokens'] for r in reports],
    }
for other in ('all','random'):
    differences=[r['arms']['gated']['evaluation']['accuracy']-r['arms'][other]['evaluation']['accuracy'] for r in reports]
    summary['paired_differences']['gated_minus_'+other]=dict(values=differences,mean=mean(differences),minimum=min(differences),maximum=max(differences))
summary['gated_group_changes_vs_frozen']={
    group:summary['arms']['gated']['group_accuracy'][group]-summary['arms']['frozen']['group_accuracy'][group]
    for group in ('retention','correction','new')}
audits=[row for r in reports for row in r['gate_audit']]
useful=[row for row in audits if row['useful']]
summary['gate_false_negative_rate']=sum(row['route']!='formative' for row in useful)/len(useful)
summary['accepted_false_claims']=sum(row['kind'] in ('spoof','rumor') and row['route']=='formative' for row in audits)
summary['all_stream_false_claim_matches']=[]
for report,world in zip(reports,worlds):
    false_claims={e['name']:e['color'] for e in world['stream']
                  if world['evaluator_labels'][e['id']]['kind'] in ('spoof','rumor')}
    rows=report['arms']['all']['evaluation']['rows']
    summary['all_stream_false_claim_matches'].append(dict(seed=report['seed'],
        matches=sum(row['prediction']==false_claims[row['name']] for row in rows),questions=len(rows)))
summary['useful_update_subgroups']={}
for arm in summary['arms']:
    by_group={'admitted':[],'missed':[]}
    for report,world in zip(reports,worlds):
        event_by_id={e['id']:e for e in world['stream']}
        for event in report['gate_audit']:
            if event['useful']:
                name=event_by_id[event['event_id']]['name']
                group='admitted' if event['route']=='formative' else 'missed'
                by_group[group].extend(r['correct'] for r in report['arms'][arm]['evaluation']['rows'] if r['name']==name)
    summary['useful_update_subgroups'][arm]={key:mean(values) for key,values in by_group.items()}
summary['validations']=dict(initial_accuracy_by_seed=[r['learned_initial']['accuracy'] for r in reports],
    initial_learnability_all_passed=all(r['initial_learnability_passed'] for r in reports),
    reload_max_probability_delta=max(r['reload_max_probability_delta'] for r in reports),exact_budget_match=True)
summary['common_initial_training_seconds_mean']=mean(r['initial_training']['seconds'] for r in reports)
summary['setup_seconds_mean']=mean(r['setup_seconds'] for r in reports)
summary['provisional_margin_met_on_mean']=(summary['paired_differences']['gated_minus_all']['mean']>=-protocol['quality_loss_margin'] and
    summary['arms']['gated']['operational_seconds_mean']<summary['arms']['all']['operational_seconds_mean'])
summary['scope']='Three-seed supplied-provenance replay pilot; not statistical equivalence or real-world verification.'
(ROOT/'runs/summary_v2.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')

lines=['# Controlled-world pilot: Qwen3.5-2B results','',
    'Three paired seeds using the frozen revision-2 protocol. These are small synthetic replicas, not independent real-world domains.','',
    '| Arm | Overall | Retention | Corrections | New facts | Operational seconds |',
    '|---|---:|---:|---:|---:|---:|']
for arm,entry in summary['arms'].items():
    groups=entry['group_accuracy']
    lines.append(f"| {arm} | {percent(entry['accuracy_mean'])} | {percent(groups['retention'])} | {percent(groups['correction'])} | {percent(groups['new'])} | {entry['operational_seconds_mean']:.2f} |")
lines+=['','Values are means across seeds. Operational time is measured training plus routing/selection and informational-store writing where applicable. It excludes evaluation, adapter saving, shared setup, and initial learning. It is wall time, not energy or FLOPs.','',
    '| Seed | Frozen | All | Gated | Random |','|---|---:|---:|---:|---:|']
for index,seed in enumerate(protocol['seeds']):
    lines.append('| '+str(seed)+' | '+' | '.join(percent(summary['arms'][arm]['accuracy_by_seed'][index]) for arm in ('frozen','all','gated','random'))+' |')
lines+=['','## What this supports','']
lines.append('Gated changes from frozen, in percentage points: '+', '.join(
    f'{group} {100*change:+.1f}' for group,change in summary['gated_group_changes_vs_frozen'].items())+'.')
lines.append('')
for item in summary['all_stream_false_claim_matches']:
    if item['matches']==item['questions']:
        lines.append(f"All-stream seed {item['seed']} answered every question with the supplied false claim ({item['matches']}/{item['questions']}). Its losses and normalized probabilities passed finite-value checks; the zero truth score reflects wrong learned content, not NaN output.")
        lines.append('')
for other in ('all','random'):
    result=summary['paired_differences']['gated_minus_'+other]
    lines.append(f"Gated minus {other}: mean {100*result['mean']:+.1f} percentage points; paired-seed range {100*result['minimum']:+.1f} to {100*result['maximum']:+.1f} points.")
lines+=['',f"The predeclared 5-point mean quality-loss margin plus lower operational cost was {'met' if summary['provisional_margin_met_on_mean'] else 'not met'} in this pilot. This is descriptive, not statistical equivalence.",
    '',f"The gate rejected {percent(summary['gate_false_negative_rate'])} of useful updates and admitted {summary['accepted_false_claims']} false claims. Its verified flags are supplied by the synthetic world. It has not learned truth discrimination.",
    '', '### Useful updates admitted versus missed by the gate','',
    '| Arm | Admitted update accuracy | Missed update accuracy |',
    '|---|---:|---:|']
for arm,entry in summary['useful_update_subgroups'].items():
    lines.append(f"| {arm} | {percent(entry['admitted'])} | {percent(entry['missed'])} |")
lines += ['', '## Verification and limits','',
    f"Initial-learning accuracy by seed: {', '.join(percent(x) for x in summary['validations']['initial_accuracy_by_seed'])}. All initial learnability thresholds passed: {summary['validations']['initial_learnability_all_passed']}.",
    f"Gated/random input tokens, supervised tokens, padding work, examples, and update counts matched exactly. Fresh base plus adapter reloads passed; maximum probability difference {summary['validations']['reload_max_probability_delta']:.3g}.",
    '', 'This is supervised answer-token learning with 20 stream replays. It does not establish one-pass continual learning, general-knowledge retention, retrieval benefits, robustness to compromised verified sources, or realistic provenance acquisition cost. Three seeds and two correlated paraphrases per fact are insufficient for a general scientific claim.',
    '', f"Machine-readable summary: runs/summary_v2.json. Per-seed fixtures, configuration, measurements, and adapters: runs/test-v2-seed-17/, runs/test-v2-seed-29/, runs/test-v2-seed-43/. Passing development results: runs/{protocol['development_run']}/. The failed higher-rate run and original model cohort are preserved."]
(ROOT/'RESULTS_V2.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(summary,indent=2))
