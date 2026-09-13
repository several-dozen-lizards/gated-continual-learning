"""Run each stage's CPU tests in its own process to isolate module names."""
import os,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parent/'study'
for directory in [root]+sorted(p for p in root.iterdir() if p.is_dir()):
    if not list(directory.glob('test_*.py')):continue
    subprocess.run([sys.executable,'-X','utf8','-m','unittest','discover','-s',str(directory),'-p','test_*.py','-v'],
        env=dict(os.environ,PYTHONUTF8='1'),check=True)
print('All archived CPU tests passed.')
