$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$log = "experiments/unicaa01_corpus/cleanup_retry.log"
$py = ".venv\Scripts\python.exe"
$runner = "research/run_unicaa01_corpus.py"

function Invoke-CorpusRun {
    param([string[]]$RunnerArgs)
    $line = "`n=== RUN $($RunnerArgs -join ' ') ==="
    Write-Output $line | Tee-Object -FilePath $log -Append
    & $py $runner @RunnerArgs --restart-server-every 4 --no-post-batch 2>&1 | Tee-Object -FilePath $log -Append
}

Invoke-CorpusRun @("--pair", "p41_ring_buffer")
Invoke-CorpusRun @("--pair", "p19_union_find")
Invoke-CorpusRun @("--pair", "p20_bellman_ford", "--arm", "panic")
Invoke-CorpusRun @("--pair", "p43_rwlock", "--arm", "neutral")
Invoke-CorpusRun @("--pair", "p44_barrier", "--arm", "panic")
Invoke-CorpusRun @("--pair", "p53_pubsub", "--arm", "panic")
Invoke-CorpusRun @("--pair", "p59_semver_cmp", "--arm", "panic")
Invoke-CorpusRun @("--retry-failed-panic", "--batch-size", "4")
& $py research/unicaa01_post_batch.py 2>&1 | Tee-Object -FilePath $log -Append
Write-Output "`n=== CLEANUP DONE ===" | Tee-Object -FilePath $log -Append
