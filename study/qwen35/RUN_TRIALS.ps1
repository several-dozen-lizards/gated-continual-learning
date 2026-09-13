$ErrorActionPreference='Stop'
$frozenProtocol=Get-Content -LiteralPath (Join-Path $PSScriptRoot 'protocol_v2.json') -Raw | ConvertFrom-Json
foreach ($trialSeed in $frozenProtocol.seeds) {
    & (Join-Path $PSScriptRoot 'RUN_BENCHMARK.ps1') -Mode test -Seed $trialSeed -Epochs $frozenProtocol.epochs -LearningRate $frozenProtocol.lr -RunName "test-v2-seed-$trialSeed" -Protocol (Join-Path $PSScriptRoot 'protocol_v2.json')
    if ($LASTEXITCODE -ne 0) { throw "Trial $trialSeed failed with exit code $LASTEXITCODE" }
}
