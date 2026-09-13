"""Revision 2: development learning check and paired synthetic-world trials."""
import argparse
import copy
import gc
import json
import random
import time
from pathlib import Path
from world import COLORS, TRAIN_TEMPLATES, EVAL_TEMPLATES, digest, make_world, matched_random, route

ROOT = Path(__file__).resolve().parent


def write(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['development','test'], required=True)
    p.add_argument('--seed', type=int, default=101)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--protocol', type=Path)
    return p.parse_args()


def main():
    args = arguments()
    if args.mode == 'test':
        if not args.protocol:
            raise ValueError('Test requires a frozen protocol file.')
        protocol = json.loads(args.protocol.read_text())
        assert args.epochs == protocol['epochs'] and args.lr == protocol['lr']
        assert args.seed in protocol['seeds']
        assert protocol['development_passed']
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    import transformers
    import peft
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, get_peft_model_state_dict, set_peft_model_state_dict, PeftModel
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required.')
    free, total = torch.cuda.mem_get_info()
    if free < 3 * 1024**3:
        raise RuntimeError(f'Only {free/1024**3:.2f} GiB free; need 3 GiB before starting.')
    snapshot = json.loads((ROOT/'model_snapshot.json').read_text())
    if args.mode == 'test':
        assert snapshot['revision'] == protocol['model_revision']
        for filename, expected_hash in protocol['source_hashes'].items():
            import hashlib
            assert hashlib.sha256((ROOT/filename).read_bytes()).hexdigest() == expected_hash, filename
    world = make_world(args.mode, args.seed)
    if args.mode == 'test':
        assert digest(world) == protocol['world_hashes'][str(args.seed)]
    write(args.output/'world.json', world)
    write(args.output/'config.json', dict(mode=args.mode,seed=args.seed,epochs=args.epochs,lr=args.lr,model=snapshot,
        protocol_sha256=digest(protocol) if args.mode=='test' else None,world_sha256=digest(world)))
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(snapshot['path'], local_files_only=True)
    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.eos_token
    token = lambda text: tokenizer.encode(text, add_special_tokens=False)
    option_ids = [token(' '+c) for c in COLORS]
    assert all(len(x)==1 for x in option_ids), 'Evaluation assumes one token per color.'

    def encode(name, color, template):
        prefix = token(template.format(name=name))
        answer = token(' '+color)
        return (prefix+answer, [-100]*len(prefix)+answer)

    def examples(events):
        return [encode(e['name'],e['color'],t) for e in events for t in TRAIN_TEMPLATES]

    def cost(e):
        return tuple(len(encode(e['name'],e['color'],t)[0]) for t in TRAIN_TEMPLATES)

    def new_base():
        return AutoModelForCausalLM.from_pretrained(snapshot['path'],local_files_only=True,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16),
            device_map={'':0}, dtype=torch.bfloat16)

    set_seed(args.seed)
    model = prepare_model_for_kbit_training(new_base(), gradient_checkpointing_kwargs={'use_reentrant':False})
    model = get_peft_model(model,LoraConfig(r=16,lora_alpha=32,target_modules=['q_proj','k_proj','v_proj','o_proj'],lora_dropout=0,bias='none',task_type='CAUSAL_LM'))
    model.config.use_cache=False
    setup_seconds = time.perf_counter()-started
    # All training examples share a global padding width, so identical examples,
    # steps, nonpadding tokens and padding work can all be checked explicitly.
    initial_events = [dict(name=n,color=c) for n,c in world['initial'].items()]
    width = max(len(ids) for ids,_ in examples(initial_events+world['stream']))

    def train(events, label):
        data = examples(events)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=args.lr)
        model.train()
        set_seed(args.seed)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        losses=[]
        for epoch in range(args.epochs):
            # Replayed stream order is preserved in every arm. This is a replay
            # pilot, not a claim of one-pass online learning.
            for offset in range(0,len(data),4):
                batch=data[offset:offset+4]
                ids=torch.full((len(batch),width),tokenizer.eos_token_id,dtype=torch.long,device='cuda')
                labels=torch.full_like(ids,-100)
                mask=torch.zeros_like(ids)
                for row,(seq,target) in enumerate(batch):
                    ids[row,:len(seq)]=torch.tensor(seq,device='cuda')
                    labels[row,:len(seq)]=torch.tensor(target,device='cuda')
                    mask[row,:len(seq)]=1
                optimizer.zero_grad(set_to_none=True)
                loss=model(input_ids=ids,attention_mask=mask,labels=labels).loss
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError('Nonfinite training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                optimizer.step()
                losses.append(float(loss.detach()))
                del loss
            if (epoch+1) in {1,args.epochs//2,args.epochs}:
                print(f'{label}: epoch {epoch+1}/{args.epochs}, loss={losses[-1]:.4f}',flush=True)
        torch.cuda.synchronize()
        elapsed=time.perf_counter()-start
        result=dict(seconds=elapsed,steps=len(losses),examples=len(data)*args.epochs,
            input_tokens=sum(len(x) for x,_ in data)*args.epochs,
            supervised_tokens=len(data)*args.epochs,padded_tokens=len(data)*width*args.epochs,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),losses=losses)
        optimizer.zero_grad(set_to_none=True)
        del optimizer
        return result

    def evaluate(targets):
        model.eval()
        rows=[]
        torch.cuda.synchronize()
        start=time.perf_counter()
        with torch.inference_mode():
            for name,answer in targets.items():
                for template in EVAL_TEMPLATES:
                    prompt=template.format(name=name)
                    ids=torch.tensor([token(prompt)],device='cuda')
                    logits=model(input_ids=ids).logits[0,-1].float()
                    scores=logits[[x[0] for x in option_ids]]
                    probs=scores.softmax(0).cpu().tolist()
                    prediction=COLORS[max(range(4),key=probs.__getitem__)]
                    rows.append(dict(name=name,group=world['groups'][name],prompt=prompt,answer=answer,
                        prediction=prediction,correct=prediction==answer,probabilities=probs,
                        brier=sum((p-float(c==answer))**2 for p,c in zip(probs,COLORS))))
        torch.cuda.synchronize()
        metrics={group:sum(r['correct'] for r in rows if r['group']==group)/sum(r['group']==group for r in rows)
                 for group in sorted({r['group'] for r in rows})}
        return dict(seconds=time.perf_counter()-start,accuracy=sum(r['correct'] for r in rows)/len(rows),
                    group_accuracy=metrics,brier=sum(r['brier'] for r in rows)/len(rows),rows=rows)

    report=dict(status='running',mode=args.mode,seed=args.seed,setup_seconds=setup_seconds,
                versions={'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__},
                gpu=torch.cuda.get_device_name(0),arms={})
    report['base_initial']=evaluate(world['initial'])
    report['initial_training']=train(initial_events,'initial')
    report['learned_initial']=evaluate(world['initial'])
    model.save_pretrained(args.output/'initial_adapter')
    prior={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()}
    report['initial_learnability_passed']=report['learned_initial']['accuracy']>=0.875
    write(args.output/'report.json',report)
    print('Initial learned accuracy:',report['learned_initial']['accuracy'],flush=True)
    if args.mode=='development':
        clean=[e for e in world['stream'] if world['evaluator_labels'][e['id']]['useful']]
        report['clean_training']=train(clean,'clean')
        report['clean_evaluation']=evaluate(world['final'])
        metrics=report['clean_evaluation']['group_accuracy']
        report['development_passed']=report['initial_learnability_passed'] and metrics['new']>=0.75 and metrics['correction']>=0.75
        model.save_pretrained(args.output/'clean_adapter')
        report['status']='complete'
        write(args.output/'report.json',report)
        print('Development passed:',report['development_passed'],metrics,flush=True)
        return

    start=time.perf_counter()
    decisions=[route(e) for e in world['stream']]
    gated=[i for i,d in enumerate(decisions) if d['route']=='formative']
    report['gate_seconds']=time.perf_counter()-start
    start=time.perf_counter()
    random_ids=matched_random(world['stream'],gated,cost,args.seed)
    report['random_selection_seconds']=time.perf_counter()-start
    arms={'frozen':[],'all':list(range(len(world['stream']))),'gated':gated,'random':random_ids}
    report['gate_audit']=[dict(event_id=e['id'],**d,**world['evaluator_labels'][e['id']]) for e,d in zip(world['stream'],decisions)]
    execution_order=['all','gated','random']
    random.Random(args.seed).shuffle(execution_order)
    execution_order=['frozen']+execution_order
    report['execution_order']=execution_order
    start=time.perf_counter()
    write(args.output/'informational.json',[e for e,d in zip(world['stream'],decisions) if d['route']=='informational'])
    write(args.output/'forgettable.json',[dict(hash=digest(e),reason=d['reason']) for e,d in zip(world['stream'],decisions) if d['route']=='forgettable'])
    report['storage_seconds']=time.perf_counter()-start
    for arm in execution_order:
        arm_start=time.perf_counter()
        set_peft_model_state_dict(model,copy.deepcopy(prior))
        model.zero_grad(set_to_none=True)
        state=get_peft_model_state_dict(model)
        assert all(torch.equal(state[k].cpu(),v) for k,v in prior.items())
        del state
        training=train([world['stream'][i] for i in arms[arm]],arm) if arms[arm] else dict(seconds=0,steps=0,examples=0,input_tokens=0,supervised_tokens=0,padded_tokens=0)
        evaluation=evaluate(world['final'])
        start=time.perf_counter()
        model.save_pretrained(args.output/(arm+'_adapter'))
        save_seconds=time.perf_counter()-start
        report['arms'][arm]=dict(indices=arms[arm],training=training,evaluation=evaluation,
            save_seconds=save_seconds,arm_wall_seconds=time.perf_counter()-arm_start)
        write(args.output/'report.json',report)
        print(arm,evaluation['group_accuracy'],flush=True)
    for key in ('steps','examples','input_tokens','supervised_tokens','padded_tokens'):
        assert report['arms']['gated']['training'][key]==report['arms']['random']['training'][key],key
    # Fresh base + PEFT reload, rather than merely opening tensor files.
    del model, prior
    gc.collect(); torch.cuda.empty_cache()
    reload_base=prepare_model_for_kbit_training(new_base(), gradient_checkpointing_kwargs={'use_reentrant':False})
    model=PeftModel.from_pretrained(reload_base,args.output/'gated_adapter')
    model.config.use_cache=False
    reloaded=evaluate(world['final'])
    original=report['arms']['gated']['evaluation']['rows']
    report['reload_max_probability_delta']=max(abs(a-b) for x,y in zip(original,reloaded['rows']) for a,b in zip(x['probabilities'],y['probabilities']))
    report['reload_predictions_match']=all(x['prediction']==y['prediction'] for x,y in zip(original,reloaded['rows']))
    report['reload_passed']=report['reload_predictions_match'] and report['reload_max_probability_delta']<=1e-5
    report['status']='complete' if report['reload_passed'] else 'reload_validation_failed'
    write(args.output/'report.json',report)
    print('Complete. Reload delta:',report['reload_max_probability_delta'],flush=True)


if __name__=='__main__':
    main()
