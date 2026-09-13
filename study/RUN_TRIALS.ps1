$ErrorActionPreference='Stop'
foreach ($trialSeed in @(17,29,43)) {
    & (Join-Path $PSScriptRoot 'RUN_BENCHMARK.ps1') -Mode test -Seed $trialSeed -RunName "test-v2-seed-$trialSeed" -Protocol (Join-Path $PSScriptRoot 'protocol_v2.json')
    if ($LASTEXITCODE -ne 0) { throw "Trial $trialSeed failed with exit code $LASTEXITCODE" }
}
