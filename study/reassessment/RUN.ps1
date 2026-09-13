param([int[]]$Seeds=@(17,29,43))
$ErrorActionPreference='Stop'
$gpuPython = if ($env:GCL_PYTHON) { $env:GCL_PYTHON } else { 'python' }
foreach ($trialSeed in $Seeds) {
    & $gpuPython -u -c 'import os,sys,runpy; root=sys.argv.pop(1); os.environ["HF_HUB_OFFLINE"]="1"; sys.path[:0]=[root,root+"/../packages"]; runpy.run_path(root+"/runner.py",run_name="__main__")' $PSScriptRoot --seed $trialSeed
    if ($LASTEXITCODE -ne 0) { throw "Trial $trialSeed failed: $LASTEXITCODE" }
}
