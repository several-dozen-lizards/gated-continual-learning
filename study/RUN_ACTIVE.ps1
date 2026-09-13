# User-selected Qwen3.5-2B experiment. Arguments pass through to its launcher.
& (Join-Path $PSScriptRoot 'qwen35\RUN_BENCHMARK.ps1') @args
exit $LASTEXITCODE
