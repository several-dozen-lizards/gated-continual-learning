"""Evaluate bounded replay over two later learning waves."""
import argparse
import copy
import gc
import hashlib
import json
import random
import time
from pathlib import Path
from selection import schedule
COLORS=('red','blue','green','gold')

ROOT=Path(__file__).resolve().parent
PRIOR=ROOT.parent/'repair'
TRAIN=('Beacon color for {name}:','The beacon at {name} has color:')
EVAL=('What color is the beacon at {name}? Answer:','Identify the beacon color at {name}:')
TRANSFER=('Name the color of the beacon located at {name}.','At {name}, which color does the beacon show?',
    'Give the beacon color associated with {name}.','Which color belongs to the {name} beacon?')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def save(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,required=True)
    args=parser.parse_args()
    plan=json.loads((ROOT/'plan.json').read_text())
    assert args.seed in plan['seeds']
    for file,expected in plan['source_hashes'].items():
        assert sha(ROOT/file)==expected,file
    fixture_path=ROOT/'fixtures'/f'{args.seed}.json'
    assert sha(fixture_path)==plan['fixtures'][str(args.seed)]
    payload=json.loads(fixture_path.read_text())
    initial_world=payload['world']
    world=copy.deepcopy(initial_world)
    prior_run=PRIOR/'runs'/f'seed-{args.seed}'
    for file,expected in plan['prior_receipts'][str(args.seed)].items():
        assert sha(prior_run/file)==expected,file
    out=ROOT/'runs'/f'seed-{args.seed}'
    out.mkdir(parents=True,exist_ok=False)
    save(out/'inputs.json',payload)
    save(out/'plan.json',plan)
    import torch
    import transformers
    import peft
    from transformers import AutoTokenizer,Qwen3_5ForCausalLM,BitsAndBytesConfig,set_seed
    from peft import PeftModel,prepare_model_for_kbit_training,get_peft_model_state_dict,set_peft_model_state_dict
    assert torch.cuda.is_available()
    free,_=torch.cuda.mem_get_info()
    if free<3*1024**3:
        raise RuntimeError('Less than 3 GiB free GPU memory; stop before loading.')
    snapshot=json.loads((ROOT.parent/'qwen35'/'model_snapshot.json').read_text())
    assert snapshot['revision']==plan['model_revision']
    started=time.perf_counter()
    tokenizer=AutoTokenizer.from_pretrained(snapshot['path'],local_files_only=True)
    tokenize=lambda text:tokenizer.encode(text,add_special_tokens=False)
    def prompt(text):
        return tokenize(tokenizer.apply_chat_template([dict(role='user',content=text)],tokenize=False,
            add_generation_prompt=True,enable_thinking=False))
    options=[tokenize(' '+c) for c in COLORS]
    assert all(len(x)==1 for x in options)
    def examples(claims):
        rows=[]
        for c in claims:
            for template in TRAIN:
                prefix=prompt(template.format(name=c['name']))
                answer=tokenize(' '+c['color'])
                rows.append((prefix+answer,[-100]*len(prefix)+answer))
        return rows
    def load(adapter,trainable):
        base=Qwen3_5ForCausalLM.from_pretrained(snapshot['path'],local_files_only=True,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16),device_map={'':0},dtype=torch.bfloat16)
        base=prepare_model_for_kbit_training(base,gradient_checkpointing_kwargs={'use_reentrant':False})
        result=PeftModel.from_pretrained(base,adapter,is_trainable=trainable)
        result.config.use_cache=False
        return result
    set_seed(args.seed)
    model=load(prior_run/'corrected_replay_adapter',True)
    setup_seconds=time.perf_counter()-started
    prior={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()}
    started=time.perf_counter()
    arms={arm:schedule(payload['initial_ledger'],payload['waves'],arm,args.seed) for arm in ('none','small','full')}
    selection_seconds=time.perf_counter()-started
    assert arms==payload['schedules']
    width=max(len(ids) for stages in arms.values() for stage in stages for ids,_ in examples(stage['training']))
    save(out/'training_claims.json',arms)
    def advance(current,updates,wave):
        result=copy.deepcopy(current)
        for c in updates:
            result['final'][c['name']]=c['color'];result['groups'][c['name']]='wave'+str(wave)
        return result
    final_world=copy.deepcopy(initial_world)
    for wave,updates in enumerate(payload['waves'],1):final_world=advance(final_world,updates,wave)

    def train(events,label):
        data=examples(events)
        optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=plan['lr'])
        model.train();set_seed(args.seed);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
        started=time.perf_counter();cpu_started=time.process_time();losses=[]
        cuda_start=torch.cuda.Event(enable_timing=True);cuda_end=torch.cuda.Event(enable_timing=True);cuda_start.record()
        for epoch in range(plan['epochs']):
            for offset in range(0,len(data),4):
                batch=data[offset:offset+4]
                ids=torch.full((len(batch),width),tokenizer.eos_token_id,dtype=torch.long,device='cuda')
                labels=torch.full_like(ids,-100);mask=torch.zeros_like(ids)
                for row,(seq,target) in enumerate(batch):
                    ids[row,:len(seq)]=torch.tensor(seq,device='cuda')
                    labels[row,:len(seq)]=torch.tensor(target,device='cuda');mask[row,:len(seq)]=1
                optimizer.zero_grad(set_to_none=True)
                loss=model(input_ids=ids,attention_mask=mask,labels=labels).loss
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError('Nonfinite loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)
                optimizer.step();losses.append(float(loss.detach()));del loss
            if epoch+1 in (1,plan['epochs']//2,plan['epochs']):
                print(f'{label}: {epoch+1}/{plan["epochs"]}, loss {losses[-1]:.5f}',flush=True)
        cuda_end.record();torch.cuda.synchronize()
        result=dict(seconds=time.perf_counter()-started,cpu_seconds=time.process_time()-cpu_started,
            cuda_interval_seconds=cuda_start.elapsed_time(cuda_end)/1000,steps=len(losses),input_tokens=sum(len(x) for x,_ in data)*plan['epochs'],
            supervised_tokens=len(data)*plan['epochs'],padded_tokens=len(data)*width*plan['epochs'],examples=len(data)*plan['epochs'],
            losses=losses,peak_allocated_bytes=torch.cuda.max_memory_allocated())
        optimizer.zero_grad(set_to_none=True);del optimizer
        return result

    def evaluate(templates=EVAL,eval_world=None):
        eval_world=world if eval_world is None else eval_world
        model.eval();rows=[];torch.cuda.synchronize();started=time.perf_counter()
        with torch.inference_mode():
            for name,answer in eval_world['final'].items():
                for template in templates:
                    ids=torch.tensor([prompt(template.format(name=name))],device='cuda')
                    scores=model(input_ids=ids).logits[0,-1].float()[[x[0] for x in options]]
                    probabilities=scores.softmax(0).cpu().tolist()
                    prediction=COLORS[max(range(4),key=probabilities.__getitem__)]
                    rows.append(dict(name=name,template=template,group=eval_world['groups'][name],answer=answer,prediction=prediction,
                        correct=prediction==answer,probabilities=probabilities))
        torch.cuda.synchronize()
        return dict(seconds=time.perf_counter()-started,accuracy=sum(r['correct'] for r in rows)/len(rows),rows=rows,
            groups={g:sum(r['correct'] for r in rows if r['group']==g)/sum(r['group']==g for r in rows) for g in sorted(set(eval_world['groups'].values()))})

    report=dict(status='running',seed=args.seed,setup_seconds=setup_seconds,selection_seconds=selection_seconds,arms={},
        versions={'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__})
    baseline=evaluate()
    old_report=json.loads((prior_run/'report.json').read_text())['arms']['corrected_replay']
    old=old_report['evaluation']['rows']
    assert [(r['name'],r['answer']) for r in old]==[(r['name'],r['answer']) for r in baseline['rows']]
    delta=max(abs(a-b) for x,y in zip(old,baseline['rows']) for a,b in zip(x['probabilities'],y['probabilities']))
    assert delta<=1e-5,delta
    report['prior_reload_delta']=delta
    report['baseline']=baseline
    report['baseline_transfer']=evaluate(TRANSFER)
    report['prior_transfer_reload_delta']=max(abs(a-b) for x,y in zip(old_report['transfer']['rows'],report['baseline_transfer']['rows']) for a,b in zip(x['probabilities'],y['probabilities']))
    assert report['prior_transfer_reload_delta']<=1e-5
    report['frozen_final']=dict(evaluation=evaluate(eval_world=final_world),transfer=evaluate(TRANSFER,final_world))
    save(out/'report.json',report)
    rng=random.Random(args.seed)
    order=list(arms);rng.shuffle(order);report['execution_order']=order
    for arm in order:
        set_peft_model_state_dict(model,copy.deepcopy(prior));model.zero_grad(set_to_none=True)
        current=get_peft_model_state_dict(model)
        assert all(torch.equal(current[k].cpu(),v) for k,v in prior.items());del current
        world=copy.deepcopy(initial_world)
        report['arms'][arm]=dict(waves=[])
        for stage in arms[arm]:
            wave=stage['wave']
            world=advance(world,stage['updates'],wave)
            training=train(stage['training'],arm+' wave '+str(wave))
            evaluation=evaluate()
            transfer=evaluate(TRANSFER) if wave==len(arms[arm]) else None
            started=time.perf_counter();model.save_pretrained(out/(arm+'_wave'+str(wave)+'_adapter'));save_seconds=time.perf_counter()-started
            report['arms'][arm]['waves'].append(dict(wave=wave,training=training,evaluation=evaluation,transfer=transfer,save_seconds=save_seconds))
            save(out/'report.json',report)
            print(arm,'wave',wave,evaluation['accuracy'],evaluation['groups'],flush=True)
    del model,prior;gc.collect();torch.cuda.empty_cache()
    model=load(out/'small_wave2_adapter',False)
    fresh=evaluate();original=report['arms']['small']['waves'][-1]['evaluation']['rows']
    delta=max(abs(a-b) for x,y in zip(original,fresh['rows']) for a,b in zip(x['probabilities'],y['probabilities']))
    report['final_reload_delta']=delta
    fresh_transfer=evaluate(TRANSFER)
    report['transfer_reload_delta']=max(abs(a-b) for x,y in zip(report['arms']['small']['waves'][-1]['transfer']['rows'],fresh_transfer['rows']) for a,b in zip(x['probabilities'],y['probabilities']))
    delta=max(delta,report['transfer_reload_delta'])
    report['status']='complete' if delta<=1e-5 else 'reload_failed'
    save(out/'report.json',report)
    print('Completed',args.seed,'reload delta',delta,flush=True)


if __name__=='__main__':main()
