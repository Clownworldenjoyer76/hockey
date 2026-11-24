$ErrorActionPreference = 'Continue'
$Repo    = 'G:\hockey'
$Nhl     = "$Repo\docs\win\hockey\nhl"
$Scripts = "$Nhl\scripts"
$LogDir  = 'G:\hockey\local_pipeline_log'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
Set-Location $Repo
$env:PYTHONUTF8 = '1'
$RunDate = [TimeZoneInfo]::ConvertTimeBySystemTimeZoneId([DateTime]::UtcNow, 'Eastern Standard Time').ToString('yyyy_MM_dd')
$Log = "$LogDir\nhl_pipeline_$RunDate.log"

function Log([string]$Message) { Write-Host $Message; Add-Content -Path $Log -Value $Message -Encoding UTF8 }
function Run-Step([string]$Name, [string[]]$PyArgs) {
    Log ("[{0}] START {1}" -f (Get-Date -Format 's'), $Name)
    & py @PyArgs 2>&1 | ForEach-Object { Log "$_" }
    if ($LASTEXITCODE -ne 0) { throw "FAILED: $Name (exit $LASTEXITCODE)" }
    Log ("[{0}] OK {1}" -f (Get-Date -Format 's'), $Name)
}
function S311([string]$Rel, [string[]]$A = @()) { Run-Step $Rel (@('-3.11', "$Scripts\$Rel") + $A) }

$JobStatus = 'success'
try {
    Log "RUN DATE $RunDate"
    $Dirs = @('errors\00_intake','errors\01_merge','errors\02_juice','errors\03_edges','errors\04_select','errors\05_final_scores',
      'odds','00_intake\games','00_intake\drat_raw\rows','00_intake\predictions\scraper','00_intake\nhl_schedule','00_intake\sportsbook',
      '00_intake\reconciled\audit','01_merge\01_merguiced','02_juice','03_edges\ev_kelly','03_edges\secondary_signals','04_select',
      '05_final_scores\final_scores','05_final_scores\graded','05_final_scores\intermediate','05_final_scores\reports\moneyline',
      '05_final_scores\reports\puckline','05_final_scores\reports\total','05_final_scores\reports\calibration','05_final_scores\clv',
      '05_final_scores\errors','config\mapping','sdv','test_output\game_id_reconciliation')
    foreach ($d in $Dirs) { New-Item -ItemType Directory -Force -Path "$Nhl\$d" | Out-Null }
    New-Item -ItemType Directory -Force -Path "$Repo\frontend\data\pipeline_health" | Out-Null

    Run-Step 'pip install 3.11' @('-3.11','-m','pip','install','-r',"$Nhl\requirements.txt")
    S311 'tests\validate_pipeline.py'
    Run-Step 'pip install 3.12' @('-3.12','-m','pip','install','-r',"$Nhl\requirements.txt")
    $TestOut = "$Nhl\errors\nhl_pipeline_tests.txt"
    Log 'START pytest (3.12)'
    & py -3.12 -m pytest "$Scripts\tests\test_nhl_pipeline.py" "$Scripts\tests\test_secondary_model_signals.py" -v 2>&1 | ForEach-Object { "$_" } | Out-File -FilePath $TestOut -Encoding utf8
    $TestExit = $LASTEXITCODE
    if ($TestExit -ne 0) { throw "FAILED: pytest (exit $TestExit); see $TestOut" }
    Log 'OK pytest (3.12)'

    Run-Step 'playwright install chromium' @('-3.11','-m','playwright','install','chromium')
    $Compile = @('00_intake\nhl_odds_pull.py','00_intake\transform_hockey_odds.py','00_intake\transform_hockey.py','00_intake\pull_nhl_schedule.py',
      '00_intake\reconcile_game_ids.py','00_intake\build_games.py','tests\test_game_id_reconciliation.py','03_edges\build_secondary_model_signals.py',
      '04_select\hockey_select_bets.py','04_select\validate_markets_config.py','05_final_scores\01_nhl_results_grade.py')
    foreach ($f in $Compile) { Run-Step "compile $f" @('-3.11','-m','py_compile',"$Scripts\$f") }
    S311 'tests\test_game_id_reconciliation.py'
    S311 '00_intake\pull_nhl_schedule.py'

    $Sched = "$Nhl\00_intake\nhl_schedule\NHL_$RunDate.csv"
    if (-not (Test-Path $Sched)) { throw "Expected official NHL schedule file not found: $Sched" }
    $GameRows = @(Get-Content $Sched | Select-Object -Skip 1 | Where-Object { $f = $_.Split(','); $f.Length -ge 13 -and ($f[12].Trim() -eq '2' -or $f[12].Trim() -eq '3') }).Count
    Log "eligible_game_rows=$GameRows"

    if ($GameRows -gt 0) {
        S311 '00_intake\hockey_drat_scraper.py'
        S311 '00_intake\transform_hockey.py'
        S311 '00_intake\pred_name_normalization.py'
        S311 '00_intake\nhl_odds_pull.py'
        S311 '00_intake\transform_hockey_odds.py'
        S311 '00_intake\odds_name_normalization.py'
        S311 '00_intake\reconcile_game_ids.py'
        S311 '00_intake\build_games.py'
        S311 '00_intake\pull_sdv.py'
        S311 '01_merge\restore_nhl_sdv_history.py' @('prepare')
        S311 '01_merge\merge_intake.py'
        S311 '01_merge\restore_nhl_sdv_history.py' @('audit')
        S311 '01_merge\build_juice_files.py'
        S311 '02_juice\validate_juice_config.py'
        S311 '02_juice\apply_moneyline_juice.py'
        S311 '02_juice\apply_puck_line_juice.py'
        S311 '02_juice\apply_total_juice.py'
        S311 '03_edges\compute_edges.py'
        S311 '03_edges\compute_ev_kelly.py'
        S311 '03_edges\build_secondary_model_signals.py'
        S311 'research\build_sdv_history.py' @('--season','2026','--datasets','schedules,play_by_play','--overwrite')
        S311 '01_merge\predict_my_probability_v5_extended.py' @('--date',$RunDate,'--offline')
        S311 '01_merge\ablate_my_probability_v8_xg.py' @('--date',$RunDate)
        $env:GITHUB_ACTIONS = 'true'
        $env:GITHUB_REPOSITORY = 'clownworldenjoyer76/hockey'
        $env:NHL_ALLOW_INDEPENDENT_COMPARE = '1'
        $env:GITHUB_WORKSPACE = $Repo
        try {
            $FinalDir = "$Nhl\my_probability\final_files"
            $PredDir  = "$Nhl\my_probability\test_repo_side_by_side\predictions"
            New-Item -ItemType Directory -Force -Path $FinalDir | Out-Null
            $Days = @(Get-ChildItem "$Nhl\03_edges\secondary_signals" -File -Filter '????_??_??_NHL_moneyline.csv' | ForEach-Object { $_.Name.Substring(0,10) } | Sort-Object)
            Log "independent comparison dates available: $($Days.Count)"
            foreach ($Day in $Days) {
                if (-not (Test-Path "$PredDir\${Day}_NHL_independent.csv")) {
                    S311 '01_merge\test_repo_independent_compare.py' @('--date',$Day)
                }
            }
            foreach ($Day in $Days) {
                Copy-Item -Path "$PredDir\${Day}_NHL_independent.csv" -Destination "$FinalDir\" -Force -ErrorAction Stop
            }
            Log "Copied $($Days.Count) independent prediction file(s) to final_files"
        }
        finally { Remove-Item Env:GITHUB_ACTIONS,Env:GITHUB_REPOSITORY,Env:NHL_ALLOW_INDEPENDENT_COMPARE,Env:GITHUB_WORKSPACE -ErrorAction SilentlyContinue }
        S311 '04_select\validate_markets_config.py'
        S311 '04_select\hockey_select_bets.py'
    } else { Log 'No regular-season/playoff games today. Stages 00-04 skipped.' }

    $Has05 = [bool](Get-ChildItem "$Nhl\04_select" -File -Filter '*_NHL.csv' -ErrorAction SilentlyContinue | Select-Object -First 1)
    if ($Has05) {
        S311 '05_final_scores\transform_final_scores.py'
        S311 '05_final_scores\01_nhl_results_grade.py'
        S311 '05_final_scores\02_nhl_results_analyze.py'
        S311 '05_final_scores\03_nhl_results_reports.py'
        S311 '05_final_scores\04_nhl_clv.py'
        S311 '05_final_scores\05_nhl_results_dashboard.py'
    } else { Log 'No Stage 04 selection files. Stage 05 skipped.' }
}
catch { $JobStatus = 'failure'; Log "ERROR: $($_.Exception.Message)" }

$env:NHL_PIPELINE_JOB_STATUS = $JobStatus
$env:NHL_PIPELINE_RUN_DATE = $RunDate
try { S311 'pipeline_health.py' } catch { $JobStatus = 'failure'; Log "ERROR: $($_.Exception.Message)" }
if ($JobStatus -ne 'success') { exit 1 }
