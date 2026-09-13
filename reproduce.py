"""Create a fresh working tree and run the recorded experimental sequence."""
import argparse,json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PHASES=['initial','qwen35','reassessment','repair','buffer_stability','drift_replay','promotion_gate']

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workdir',type=Path,default=ROOT/'work')
    parser.add_argument('--stage',choices=PHASES+['all'])
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args();work=args.workdir.resolve()
    if not args.prepare_only and args.stage is None:parser.error('Choose --stage or --prepare-only')
    if work==ROOT or ROOT/'study'==work or ROOT/'study' in work.parents:parser.error('Use a separate working directory')
    marker=work/'.gcl-workspace'
    if not marker.exists():
        if work.exists() and any(work.iterdir()):parser.error('Work directory must be empty or created by this script')
        work.mkdir(parents=True,exist_ok=True)
        for src in (ROOT/'study').rglob('*'):
            rel=src.relative_to(ROOT/'study')
            if any(p in ('runs','fixtures','__pycache__') for p in rel.parts):continue
            if src.is_file() and src.suffix in ('.py','.md','.ps1'):
                dst=work/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
        marker.write_text('Fresh reproduction; archived receipts are not copied.\n',encoding='utf-8')
    if args.prepare_only:
        print('Prepared',work);return
    env=dict(os.environ,PYTHONUTF8='1')
    def run(script,*arguments):
        subprocess.run([sys.executable,'-X','utf8',str(script),*map(str,arguments)],cwd=script.parent,env=env,check=True)
    for phase in (PHASES if args.stage=='all' else [args.stage]):
        directory=work if phase=='initial' else work/phase
        if (directory/'runs').exists():raise RuntimeError(f'{phase} already has runs; use a fresh workdir')
        if phase in ('initial','qwen35'):
            from huggingface_hub import snapshot_download
            archive=ROOT/'study' if phase=='initial' else ROOT/'study/qwen35'
            snapshot=json.loads((archive/'model_snapshot.json').read_text(encoding='utf-8'))
            snapshot['path']=snapshot_download(repo_id=snapshot['repo'],revision=snapshot['revision'],cache_dir=str(work/'model_cache'))
            (directory/'model_snapshot.json').write_text(json.dumps(snapshot,indent=2),encoding='utf-8')
            lr=.001 if phase=='initial' else .0002
            development='dev-001' if phase=='initial' else 'dev-002'
            run(directory/'benchmark.py','--mode','development','--seed',101,'--epochs',20,'--lr',lr,'--output',directory/'runs'/development)
            run(directory/'freeze_protocol.py',*(['--development',development] if phase=='qwen35' else []))
            for seed in (17,29,43):
                run(directory/'benchmark.py','--mode','test','--seed',seed,'--epochs',20,'--lr',lr,
                    '--protocol',directory/'protocol_v2.json','--output',directory/'runs'/f'test-v2-seed-{seed}')
        else:
            run(directory/'prepare.py')
            for seed in (17,29,43):run(directory/'runner.py','--seed',seed)
        run(directory/'summarize.py')
        print('Completed',phase,flush=True)

if __name__=='__main__':main()
