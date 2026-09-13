param([string]$RunName = ('canary-' + (Get-Date -Format 'yyyyMMdd-HHmmss')))
$ErrorActionPreference = 'Stop'
$experimentRoot = $PSScriptRoot
$gpuPython = if ($env:GCL_PYTHON) { $env:GCL_PYTHON } else { 'python' }
if ($RunName -notmatch '^[a-zA-Z0-9_-]+$') { throw 'RunName must be a simple directory name.' }
# The embedded interpreter ignores PYTHONPATH: add only the experiment-local packages explicitly.
& $gpuPython -c 'import sys,runpy; root=sys.argv.pop(1); sys.path.insert(0,root+"/packages"); runpy.run_path(root+"/canary.py",run_name="__main__")' $experimentRoot --model unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit --local-only --output (Join-Path $experimentRoot "runs\$RunName")
exit $LASTEXITCODE
