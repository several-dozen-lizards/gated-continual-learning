param(
    [ValidateSet('development','test')][string]$Mode='development',
    [int]$Seed=101,
    [int]$Epochs=20,
    [double]$LearningRate=0.001,
    [string]$RunName=('benchmark-'+(Get-Date -Format 'yyyyMMdd-HHmmss')),
    [string]$Protocol=''
)
$ErrorActionPreference='Stop'
if ($RunName -notmatch '^[a-zA-Z0-9_-]+$') { throw 'Use a simple run directory name.' }
$gpuPython = if ($env:GCL_PYTHON) { $env:GCL_PYTHON } else { 'python' }
$benchmarkArgs=@('--mode',$Mode,'--seed',"$Seed",'--epochs',"$Epochs",'--lr',$LearningRate.ToString([System.Globalization.CultureInfo]::InvariantCulture),'--output',(Join-Path $PSScriptRoot "runs\$RunName"))
if ($Protocol) { $benchmarkArgs+=@('--protocol',$Protocol) }
& $gpuPython -u -c 'import os,sys,runpy; root=sys.argv.pop(1); os.environ["HF_HUB_OFFLINE"]="1"; sys.path[:0]=[root,root+"/packages"]; runpy.run_path(root+"/benchmark.py",run_name="__main__")' $PSScriptRoot @benchmarkArgs
exit $LASTEXITCODE
