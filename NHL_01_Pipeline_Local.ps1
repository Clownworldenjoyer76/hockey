# NHL 01 Pipeline - Windows PowerShell 5.1 / PowerShell 7
# Source: .github/workflows/nhl_01_pipeline.yml (user-provided)
# Executes all steps against the user's existing G:\hockey repository.
# Like the workflow, this script commits and pushes generated results to origin/main.
# Required tools (Git, Python 3.11, Python 3.12) are installed with winget if absent.
# Python packages, Windows time-zone data, and Playwright Chromium are installed.
# Git reset --hard and git clean -fd are part of the original YAML and retained.

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$RepositoryPath = 'G:\hockey'
$PythonRequirements = 'docs/win/hockey/nhl/requirements.txt'
$NhlBase = 'docs/win/hockey/nhl'
$script:GitExe = $null
$script:PyLauncher = $null
$script:Python311 = $null
$script:Python312 = $null
$script:Workspace = $null
$script:JobFailed = $false
$script:HasGames = $false
$script:HasGradingInputs = $false
$script:IntakeLogsShown = $false
$script:StageLogsShown = $false
$script:GradingLogsShown = $false
$script:GradingFilesShown = $false
$script:BasePath = $env:PATH

function Find-CommandPath {
    param([string]$Name, [string[]]$Fallbacks = @())
    $found = Get-Command -Name $Name -ErrorAction SilentlyContinue
    if ($null -ne $found) { return $found.Source }
    foreach ($path in $Fallbacks) {
        if ($path -and (Test-Path -LiteralPath $path -PathType Leaf)) { return $path }
    }
    return $null
}

function Install-WindowsDependency {
    param([string]$PackageId, [string]$Description)
    $winget = Find-CommandPath -Name 'winget.exe'
    if (-not $winget) {
        throw "$Description is missing. Windows Package Manager (winget) is also unavailable, so it cannot be installed automatically."
    }
    Write-Host "Installing required dependency: $Description ($PackageId)"
    & $winget install --id $PackageId --exact --scope user --silent --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) {
        throw "Installation failed for $Description (winget exit code $LASTEXITCODE)."
    }
}

function Ensure-Git {
    $locations = @()
    if ($env:ProgramFiles) { $locations += (Join-Path $env:ProgramFiles 'Git\cmd\git.exe') }
    $programFilesX86 = [Environment]::GetEnvironmentVariable('ProgramFiles(x86)')
    if ($programFilesX86) { $locations += (Join-Path $programFilesX86 'Git\cmd\git.exe') }
    if ($env:LOCALAPPDATA) { $locations += (Join-Path $env:LOCALAPPDATA 'Programs\Git\cmd\git.exe') }
    $script:GitExe = Find-CommandPath -Name 'git.exe' -Fallbacks $locations
    if (-not $script:GitExe) {
        Install-WindowsDependency -PackageId 'Git.Git' -Description 'Git for Windows'
        $script:GitExe = Find-CommandPath -Name 'git.exe' -Fallbacks $locations
    }
    if (-not $script:GitExe) {
        throw 'Git was installed but could not be located. Open a new PowerShell session and run this file again.'
    }
    # Keep Git visible to child programs even if winget installed it in this same session.
    $env:PATH = (Split-Path -Parent $script:GitExe) + ';' + $env:PATH
}

function Get-PythonLauncher {
    $locations = @(
        (Join-Path $env:WINDIR 'py.exe'),
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Launcher\py.exe')
    )
    return Find-CommandPath -Name 'py.exe' -Fallbacks $locations
}

function Has-PythonVersion {
    param([string]$Version)
    if (-not $script:PyLauncher) { return $false }
    $previous = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $script:PyLauncher "-$Version" -c 'import sys; print(sys.version)' *> $null
        return ($LASTEXITCODE -eq 0)
    }
    catch { return $false }
    finally { $ErrorActionPreference = $previous }
}

function Ensure-PythonVersion {
    param([string]$Version)
    $script:PyLauncher = Get-PythonLauncher
    if (Has-PythonVersion -Version $Version) { return }
    Install-WindowsDependency -PackageId "Python.Python.$Version" -Description "Python $Version"
    $script:PyLauncher = Get-PythonLauncher
    if (-not (Has-PythonVersion -Version $Version)) {
        throw "Python $Version could not be started through the Python launcher after installation."
    }
}

function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments)
    $previous = $ErrorActionPreference
    try {
        # Native tools sometimes write harmless progress to stderr on Windows.
        # Display all output without returning it as a PowerShell function value.
        $ErrorActionPreference = 'Continue'
        & $Executable @Arguments 2>&1 | Out-Host
        $exitCode = $LASTEXITCODE
    }
    finally { $ErrorActionPreference = $previous }
    if ($exitCode -ne 0) {
        throw "Command failed (exit $exitCode): $Executable $($Arguments -join ' ')"
    }
}

function Invoke-Git {
    param([string[]]$Arguments)
    Invoke-Checked -Executable $script:GitExe -Arguments $Arguments
}

function Invoke-Python {
    param([string]$Version, [string[]]$Arguments)
    $executable = if ($Version -eq '3.12') { $script:Python312 } else { $script:Python311 }
    if (-not $executable) { throw "Python $Version environment has not been set up." }
    Invoke-Checked -Executable $executable -Arguments $Arguments
}

function Select-PythonEnvironment {
    param([string]$Version)
    $executable = if ($Version -eq '3.12') { $script:Python312 } else { $script:Python311 }
    if (-not $executable) { throw "Cannot select Python $Version before installation." }
    $env:VIRTUAL_ENV = Split-Path -Parent (Split-Path -Parent $executable)
    $env:PATH = (Split-Path -Parent $executable) + ';' + $script:BasePath
}

function Start-Step {
    param([string]$Name)
    Write-Host "`n===== $Name ====="
}

function Initialize-PythonEnvironment {
    param([string]$Version, [string]$Destination)
    Start-Step "Set up Python $Version and install NHL pipeline dependencies"
    Invoke-Checked -Executable $script:PyLauncher -Arguments @("-$Version", '-m', 'venv', $Destination)
    $pythonExe = Join-Path $Destination 'Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $pythonExe -PathType Leaf)) {
        throw "Python $Version virtual environment is missing: $pythonExe"
    }
    Invoke-Checked -Executable $pythonExe -Arguments @('-m', 'pip', 'install', '--upgrade', 'pip')
    # Windows does not ship the IANA timezone files used by ZoneInfo('America/New_York').
    # Installing tzdata supplies the equivalent of the Ubuntu runner's system timezone data.
    Invoke-Checked -Executable $pythonExe -Arguments @('-m', 'pip', 'install', '-r', $PythonRequirements, 'tzdata')
    return $pythonExe
}

function Commit-StagedIfChanged {
    param([string]$CommitMessage, [string]$UnchangedMessage, [switch]$Push)
    & $script:GitExe diff --cached --quiet
    $diffCode = $LASTEXITCODE
    if ($diffCode -eq 0) {
        if ($UnchangedMessage) { Write-Host $UnchangedMessage }
        return $false
    }
    if ($diffCode -ne 1) { throw "git diff --cached failed (exit $diffCode)." }
    Invoke-Git -Arguments @('commit', '-m', $CommitMessage)
    if ($Push) {
        Invoke-Git -Arguments @('pull', '--rebase', 'origin', 'main')
        Invoke-Git -Arguments @('push')
    }
    return $true
}

function Stage-Paths {
    param([string[]]$Paths)
    # Git paths are relative to the repository root, matching the YAML.
    Invoke-Git -Arguments (@('add', '-A', '--') + $Paths)
}

function Show-TextLogs {
    param([string]$Caption, [string]$Directory)
    Write-Host "===== $Caption ====="
    if (-not (Test-Path -LiteralPath $Directory -PathType Container)) { return }
    Get-ChildItem -LiteralPath $Directory -Filter '*.txt' -File -Recurse -ErrorAction SilentlyContinue |
        Sort-Object -Property FullName | ForEach-Object {
            Write-Host "----- $($_.FullName) -----"
            try { Get-Content -LiteralPath $_.FullName -ErrorAction Stop | ForEach-Object { Write-Host $_ } }
            catch { Write-Warning "Cannot read log $($_.FullName): $_" }
        }
}

function Show-IntakeLogs {
    if ($script:IntakeLogsShown) { return }
    $script:IntakeLogsShown = $true
    Show-TextLogs -Caption 'NHL 00 Intake Logs' -Directory "$NhlBase/errors/00_intake"
}

function Show-StageLogs {
    if ($script:StageLogsShown -or -not $script:HasGames) { return }
    $script:StageLogsShown = $true
    Show-TextLogs -Caption 'NHL 01 Merge Logs' -Directory "$NhlBase/errors/01_merge"
    Show-TextLogs -Caption 'NHL 02 Juice Logs' -Directory "$NhlBase/errors/02_juice"
    Show-TextLogs -Caption 'NHL 03 Edges Logs' -Directory "$NhlBase/errors/03_edges"
    Show-TextLogs -Caption 'NHL 04 Select Logs' -Directory "$NhlBase/errors/04_select"
}

function Show-GradingLogs {
    if ($script:GradingLogsShown -or -not $script:HasGradingInputs) { return }
    $script:GradingLogsShown = $true
    Write-Host '===== transform_final_scores log ====='
    $file = "$NhlBase/errors/05_final_scores/transform_final_scores.txt"
    if (Test-Path -LiteralPath $file -PathType Leaf) {
        Get-Content -LiteralPath $file | ForEach-Object { Write-Host $_ }
    }
    else { Write-Host 'No transform_final_scores.txt log found.' }
    Show-TextLogs -Caption 'NHL Stage 05 result-script logs' -Directory "$NhlBase/05_final_scores/errors"
}

function Show-GeneratedCsv {
    param([string]$Caption, [string]$Directory, [switch]$TopLevelOnly)
    Write-Host "===== $Caption ====="
    if (-not (Test-Path -LiteralPath $Directory -PathType Container)) { return }
    $files = if ($TopLevelOnly) {
        Get-ChildItem -LiteralPath $Directory -Filter '*.csv' -File -ErrorAction SilentlyContinue
    } else {
        Get-ChildItem -LiteralPath $Directory -Filter '*.csv' -File -Recurse -ErrorAction SilentlyContinue
    }
    foreach ($file in $files) { Write-Host $file.FullName }
}

function Show-GradingFiles {
    if ($script:GradingFilesShown -or -not $script:HasGradingInputs) { return }
    $script:GradingFilesShown = $true
    Show-GeneratedCsv 'Final score outputs' "$NhlBase/05_final_scores/final_scores"
    Show-GeneratedCsv 'Graded outputs' "$NhlBase/05_final_scores/graded"
    Show-GeneratedCsv 'Intermediate outputs' "$NhlBase/05_final_scores/intermediate"
    Show-GeneratedCsv 'Report outputs' "$NhlBase/05_final_scores/reports"
    Show-GeneratedCsv 'CLV outputs' "$NhlBase/05_final_scores/clv"
    Write-Host '===== Market tally ====='
    $tally = "$NhlBase/05_final_scores/nhl_market_tally.csv"
    if (Test-Path -LiteralPath $tally -PathType Leaf) { Write-Host (Resolve-Path $tally).Path }
}

function Invoke-NhlPipeline {
    Start-Step 'Use existing hockey repository on main'
    Ensure-Git
    $script:BasePath = $env:PATH
    if (-not (Test-Path -LiteralPath $RepositoryPath -PathType Container)) {
        throw "Repository folder does not exist: $RepositoryPath"
    }
    Set-Location -LiteralPath $RepositoryPath
    $actualRoot = (& $script:GitExe rev-parse --show-toplevel 2>$null)
    if ($LASTEXITCODE -ne 0 -or -not $actualRoot) {
        throw "$RepositoryPath is not a Git repository."
    }
    $actualFullPath = [IO.Path]::GetFullPath(([string]$actualRoot).Trim()).TrimEnd([char[]]@('\','/'))
    $requestedFullPath = [IO.Path]::GetFullPath($RepositoryPath).TrimEnd([char[]]@('\','/'))
    if ($actualFullPath -ine $requestedFullPath) {
        throw "Repository root differs from $RepositoryPath. Actual: $actualFullPath"
    }
    $branch = (& $script:GitExe branch --show-current).Trim()
    if ($LASTEXITCODE -ne 0 -or $branch -cne 'main') {
        throw 'G:\hockey must be on the main branch to reproduce the YAML checkout of main.'
    }
    # GitHub always starts with a clean checkout. Refuse to destroy user edits.
    # The YAML's reset/clean steps are preserved later, after test outputs are staged.
    $workingChanges = @(& $script:GitExe status --porcelain=v1 --untracked-files=all)
    if ($LASTEXITCODE -ne 0) { throw 'Could not inspect repository changes.' }
    if ($workingChanges.Count -gt 0) {
        throw 'G:\hockey contains uncommitted changes or untracked files. GitHub starts clean; this run stops rather than deleting them. No files were changed.'
    }
    Start-Step 'Update existing main branch to the GitHub checkout revision'
    Invoke-Git -Arguments @('fetch', 'origin', 'main')
    $localSha = (& $script:GitExe rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0) { throw 'Could not determine local commit.' }
    $remoteSha = (& $script:GitExe rev-parse refs/remotes/origin/main).Trim()
    if ($LASTEXITCODE -ne 0) { throw 'Could not determine origin/main commit.' }
    if ($localSha -cne $remoteSha) {
        & $script:GitExe merge-base --is-ancestor HEAD refs/remotes/origin/main
        if ($LASTEXITCODE -ne 0) {
            throw 'Local main contains commits not in origin/main. Refusing to overwrite them.'
        }
        Invoke-Git -Arguments @('merge', '--ff-only', 'origin/main')
    }
    $script:Workspace = $RepositoryPath
    Write-Host "Pipeline folder: $RepositoryPath"
    Ensure-PythonVersion -Version '3.11'
    Ensure-PythonVersion -Version '3.12'
    $runRoot = Join-Path ([System.IO.Path]::GetTempPath()) ('nhl_pipeline_python_' + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $runRoot -Force | Out-Null

    Start-Step 'Configure Git'
    Invoke-Git -Arguments @('config', 'user.name', 'github-actions[bot]')
    Invoke-Git -Arguments @('config', 'user.email', '41898282+github-actions[bot]@users.noreply.github.com')

    Start-Step 'Set date (America/New_York)'
    $easternZone = [System.TimeZoneInfo]::FindSystemTimeZoneById('Eastern Standard Time')
    $easternNow = [System.TimeZoneInfo]::ConvertTimeFromUtc([DateTime]::UtcNow, $easternZone)
    $env:DATE = $easternNow.ToString('yyyy_MM_dd', [System.Globalization.CultureInfo]::InvariantCulture)
    $env:TZ = 'America/New_York'
    $env:PYTHONUTF8 = '1'
    $env:PYTHONIOENCODING = 'utf-8'
    $env:GITHUB_REF_NAME = 'main'
    $env:GITHUB_SHA = (& $script:GitExe rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0) { throw 'Could not determine initial Git commit.' }
    # These two values only exist for actual GitHub Actions jobs; do not fabricate them.
    Remove-Item Env:GITHUB_RUN_ID -ErrorAction SilentlyContinue
    Remove-Item Env:GITHUB_RUN_ATTEMPT -ErrorAction SilentlyContinue
    Write-Host "DATE=$env:DATE"

    # GitHub uses Python 3.11 for validation, 3.12 for tests, 3.11 for pipeline.
    $script:Python311 = Initialize-PythonEnvironment -Version '3.11' -Destination (Join-Path $runRoot 'venv311')
    Select-PythonEnvironment -Version '3.11'
    Start-Step 'Validate NHL pipeline (Python 3.11)'
    Invoke-Python -Version '3.11' -Arguments @("$NhlBase/scripts/tests/validate_pipeline.py")

    $script:Python312 = Initialize-PythonEnvironment -Version '3.12' -Destination (Join-Path $runRoot 'venv312')
    Select-PythonEnvironment -Version '3.12'
    Start-Step 'Create NHL test output directories'
    foreach ($p in @("$NhlBase/errors", "$NhlBase/test_output/game_id_reconciliation")) {
        New-Item -ItemType Directory -Path $p -Force | Out-Null
    }

    Start-Step 'Run NHL pipeline tests (Python 3.12; continue on failure to save results)'
    $testReport = "$NhlBase/errors/nhl_pipeline_tests.txt"
    $oldPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $testOutput = @(& $script:Python312 -m pytest `
            "$NhlBase/scripts/tests/test_nhl_pipeline.py" `
            "$NhlBase/scripts/tests/test_secondary_model_signals.py" -v 2>&1 |
            ForEach-Object { $_.ToString() })
        $testCode = $LASTEXITCODE
    }
    finally { $ErrorActionPreference = $oldPreference }
    [System.IO.File]::WriteAllLines(
        (Join-Path (Get-Location).Path $testReport),
        [string[]]$testOutput,
        ([System.Text.UTF8Encoding]::new($false))
    )
    Write-Host "Pytest exit code: $testCode; log: $testReport"

    Start-Step 'Commit NHL pipeline test results'
    Stage-Paths -Paths @($testReport)
    $testCommitted = Commit-StagedIfChanged -CommitMessage 'test: update NHL pipeline test results' `
        -UnchangedMessage 'No NHL pipeline test result changes to commit.'
    # Original YAML: reset/clean generated test files after committing the test result.
    Invoke-Git -Arguments @('reset', '--hard', 'HEAD')
    Invoke-Git -Arguments @('clean', '-fd')
    if ($testCommitted) {
        Invoke-Git -Arguments @('pull', '--rebase', 'origin', 'main')
        Invoke-Git -Arguments @('push')
    }
    if ($testCode -ne 0) { throw "NHL pipeline tests failed (pytest exit $testCode)." }

    Start-Step 'Set up Python 3.11 for NHL pipeline'
    Select-PythonEnvironment -Version '3.11'
    # The 3.11 virtual environment above remains independent of the 3.12 test environment.
    Invoke-Python -Version '3.11' -Arguments @('--version')

    Start-Step 'Cache Playwright / SDV models / full-season SDV datasets'
    # Browser, xG and SDV caches persist in the user's Windows profile, not the fresh checkout.
    # Windows does not use the Ubuntu-specific `playwright install-deps` command.
    Start-Step 'Install Playwright Chromium'
    Invoke-Python -Version '3.11' -Arguments @('-m', 'playwright', 'install', 'chromium')

    Start-Step 'Create required NHL directories'
    $directories = @(
        'errors/00_intake', 'errors/01_merge', 'errors/02_juice',
        'errors/03_edges', 'errors/04_select', 'errors/05_final_scores',
        'odds',
        '00_intake/games', '00_intake/drat_raw/rows',
        '00_intake/predictions/scraper', '00_intake/nhl_schedule',
        '00_intake/sportsbook', '00_intake/reconciled/audit',
        '01_merge/01_merguiced', '02_juice', '03_edges/ev_kelly',
        '03_edges/secondary_signals', '04_select',
        '05_final_scores/final_scores', '05_final_scores/graded',
        '05_final_scores/intermediate', '05_final_scores/reports/moneyline',
        '05_final_scores/reports/puckline', '05_final_scores/reports/total',
        '05_final_scores/reports/calibration', '05_final_scores/clv',
        '05_final_scores/errors', 'config/mapping', 'sdv',
        'test_output/game_id_reconciliation'
    )
    foreach ($folder in $directories) {
        New-Item -ItemType Directory -Path "$NhlBase/$folder" -Force | Out-Null
    }
    New-Item -ItemType Directory -Path 'frontend/data/pipeline_health' -Force | Out-Null

    Start-Step 'Syntax check NHL intake and game ID scripts'
    $compileTargets = @(
        '00_intake/nhl_odds_pull.py',
        '00_intake/transform_hockey_odds.py',
        '00_intake/transform_hockey.py',
        '00_intake/pull_nhl_schedule.py',
        '00_intake/reconcile_game_ids.py',
        '00_intake/build_games.py',
        'tests/test_game_id_reconciliation.py',
        '03_edges/build_secondary_model_signals.py',
        '04_select/hockey_select_bets.py',
        '04_select/validate_markets_config.py',
        '05_final_scores/01_nhl_results_grade.py'
    )
    foreach ($target in $compileTargets) {
        Invoke-Python -Version '3.11' -Arguments @('-m', 'py_compile', "$NhlBase/scripts/$target")
    }

    Start-Step 'Run fixture-only NHL game ID reconciliation test'
    Invoke-Python -Version '3.11' -Arguments @("$NhlBase/scripts/tests/test_game_id_reconciliation.py")

    Start-Step 'Commit NHL game ID test outputs'
    Stage-Paths -Paths @("$NhlBase/test_output/game_id_reconciliation")
    [void](Commit-StagedIfChanged -CommitMessage 'test: NHL fixture game ID reconciliation' `
        -UnchangedMessage 'No NHL game ID test output changes to commit.' -Push)

    Start-Step 'Pull official NHL schedule'
    Invoke-Python -Version '3.11' -Arguments @("$NhlBase/scripts/00_intake/pull_nhl_schedule.py")

    Start-Step 'Detect current NHL slate'
    $schedulePath = "$NhlBase/00_intake/nhl_schedule/NHL_$env:DATE.csv"
    if (-not (Test-Path -LiteralPath $schedulePath -PathType Leaf)) {
        throw "Expected official NHL schedule file not found: $schedulePath"
    }
    # The YAML checks the 13th column: game_type = 2 or 3.
    $eligible = @(Import-Csv -LiteralPath $schedulePath | Where-Object {
        $_.game_type -eq '2' -or $_.game_type -eq '3'
    }).Count
    Write-Host "eligible_game_rows=$eligible"
    if ($eligible -gt 0) {
        $script:HasGames = $true
        Write-Host "Current NHL slate contains $eligible regular-season/playoff game(s)."
    }
    else {
        Write-Host 'No regular-season/playoff games today. Stages 00-04 will be skipped.'
    }

    if ($script:HasGames) {
        $intakeSteps = @(
            [pscustomobject]@{ Name = 'Run NHL D-Ratings scraper'; Script = '00_intake/hockey_drat_scraper.py' },
            [pscustomobject]@{ Name = 'Transform NHL predictions'; Script = '00_intake/transform_hockey.py' },
            [pscustomobject]@{ Name = 'Normalize NHL prediction team names'; Script = '00_intake/pred_name_normalization.py' },
            [pscustomobject]@{ Name = 'Pull ESPN NHL odds'; Script = '00_intake/nhl_odds_pull.py' },
            [pscustomobject]@{ Name = 'Transform ESPN NHL odds'; Script = '00_intake/transform_hockey_odds.py' },
            [pscustomobject]@{ Name = 'Normalize NHL odds team names'; Script = '00_intake/odds_name_normalization.py' },
            [pscustomobject]@{ Name = 'Reconcile NHL game IDs'; Script = '00_intake/reconcile_game_ids.py' },
            [pscustomobject]@{ Name = 'Build NHL games'; Script = '00_intake/build_games.py' },
            [pscustomobject]@{ Name = 'Pull SportsDataverse NHL data'; Script = '00_intake/pull_sdv.py' }
        )
        foreach ($step in $intakeSteps) {
            Start-Step $step.Name
            Invoke-Python -Version '3.11' -Arguments @("$NhlBase/scripts/$($step.Script)")
        }
    }

    Start-Step 'Commit NHL intake, odds, and SDV outputs'
    Stage-Paths -Paths @(
        "$NhlBase/odds", "$NhlBase/00_intake", "$NhlBase/sdv",
        "$NhlBase/errors/00_intake", "$NhlBase/config/mapping"
    )
    $utcNow = [DateTime]::UtcNow.ToString('yyyy-MM-dd HH:mm', [Globalization.CultureInfo]::InvariantCulture)
    [void](Commit-StagedIfChanged -CommitMessage "chore: NHL intake odds SDV run $utcNow UTC" `
        -UnchangedMessage 'No NHL intake, odds, or SDV changes to commit.' -Push)

    if ($script:HasGames) {
        $modelSteps = @(
            [pscustomobject]@{ Name = '01 Merge - Restore historical NHL SDV features'; Script = '01_merge/restore_nhl_sdv_history.py'; Arg = 'prepare' },
            [pscustomobject]@{ Name = '01 Merge - NHL merge intake'; Script = '01_merge/merge_intake.py'; Arg = '' },
            [pscustomobject]@{ Name = '01 Merge - Audit NHL SDV feature coverage'; Script = '01_merge/restore_nhl_sdv_history.py'; Arg = 'audit' },
            [pscustomobject]@{ Name = '01 Merge - NHL build juice files'; Script = '01_merge/build_juice_files.py'; Arg = '' },
            [pscustomobject]@{ Name = '02 Juice - Validate calibration config'; Script = '02_juice/validate_juice_config.py'; Arg = '' },
            [pscustomobject]@{ Name = '02 Juice - Apply moneyline juice'; Script = '02_juice/apply_moneyline_juice.py'; Arg = '' },
            [pscustomobject]@{ Name = '02 Juice - Apply puck line juice'; Script = '02_juice/apply_puck_line_juice.py'; Arg = '' },
            [pscustomobject]@{ Name = '02 Juice - Apply total juice'; Script = '02_juice/apply_total_juice.py'; Arg = '' },
            [pscustomobject]@{ Name = '03 Edges - Compute NHL edges'; Script = '03_edges/compute_edges.py'; Arg = '' },
            [pscustomobject]@{ Name = '03 Edges - Compute NHL EV and Kelly'; Script = '03_edges/compute_ev_kelly.py'; Arg = '' },
            [pscustomobject]@{ Name = '03 Edges - Build NHL secondary model signals'; Script = '03_edges/build_secondary_model_signals.py'; Arg = '' },
            [pscustomobject]@{ Name = '04 Select - Validate markets config'; Script = '04_select/validate_markets_config.py'; Arg = '' },
            [pscustomobject]@{ Name = '04 Select - Select NHL bets'; Script = '04_select/hockey_select_bets.py'; Arg = '' }
        )
        foreach ($step in $modelSteps) {
            Start-Step $step.Name
            $argv = @("$NhlBase/scripts/$($step.Script)")
            if ($step.Arg) { $argv += $step.Arg }
            Invoke-Python -Version '3.11' -Arguments $argv
        }
        Show-StageLogs

        Start-Step 'Commit NHL Stage 01-04 outputs'
        Stage-Paths -Paths @(
            "$NhlBase/sdv", "$NhlBase/01_merge", "$NhlBase/02_juice",
            "$NhlBase/03_edges", "$NhlBase/04_select",
            "$NhlBase/errors/01_merge", "$NhlBase/errors/02_juice",
            "$NhlBase/errors/03_edges", "$NhlBase/errors/04_select",
            "$NhlBase/config/mapping"
        )
        [void](Commit-StagedIfChanged -CommitMessage 'chore: update NHL Stage 01-04 outputs' `
            -UnchangedMessage 'No NHL Stage 01-04 changes to commit.' -Push)
    }

    Start-Step 'Detect NHL Stage 05 grading inputs'
    $selectedDirectory = "$NhlBase/04_select"
    $script:HasGradingInputs = [bool](@(Get-ChildItem -LiteralPath $selectedDirectory `
        -Filter '*_NHL.csv' -File -ErrorAction SilentlyContinue).Count -gt 0)
    if ($script:HasGradingInputs) {
        Write-Host 'Stage 04 selection files found. Stage 05 grading/reporting will run.'
    }
    else {
        Write-Host 'No Stage 04 selection files found. Stage 05 grading/reporting will be skipped.'
    }

    if ($script:HasGradingInputs) {
        $gradingSteps = @(
            [pscustomobject]@{ Name = '05 Final Scores - Transform final scores'; Script = '05_final_scores/transform_final_scores.py' },
            [pscustomobject]@{ Name = '05 Final Scores - Grade NHL results'; Script = '05_final_scores/01_nhl_results_grade.py' },
            [pscustomobject]@{ Name = '05 Final Scores - Analyze NHL results'; Script = '05_final_scores/02_nhl_results_analyze.py' },
            [pscustomobject]@{ Name = '05 Final Scores - Build NHL reports'; Script = '05_final_scores/03_nhl_results_reports.py' },
            [pscustomobject]@{ Name = '05 Final Scores - Calculate NHL CLV'; Script = '05_final_scores/04_nhl_clv.py' },
            [pscustomobject]@{ Name = '05 Final Scores - Build NHL dashboard'; Script = '05_final_scores/05_nhl_results_dashboard.py' }
        )
        foreach ($step in $gradingSteps) {
            Start-Step $step.Name
            Invoke-Python -Version '3.11' -Arguments @("$NhlBase/scripts/$($step.Script)")
        }
        Show-GradingLogs
        Show-GradingFiles

        Start-Step 'Commit NHL Stage 05 outputs'
        Stage-Paths -Paths @("$NhlBase/05_final_scores", "$NhlBase/errors/05_final_scores")
        Invoke-Git -Arguments @('add', '--', 'frontend/data/dashboards/nhl.json')
        [void](Commit-StagedIfChanged -CommitMessage 'chore: update NHL Stage 05 final scores and reports' `
            -UnchangedMessage 'No NHL Stage 05 changes to commit.' -Push)
    }
}

$originalLocation = Get-Location
try {
    Invoke-NhlPipeline
}
catch {
    $script:JobFailed = $true
    Write-Host "`nPIPELINE FAILED: $($_.Exception.Message)" -ForegroundColor Red
}
finally {
    if ($script:Workspace -and (Test-Path -LiteralPath (Join-Path $script:Workspace '.git'))) {
        try { Set-Location -LiteralPath $script:Workspace } catch { }
        if ($script:JobFailed) {
            try { Show-IntakeLogs } catch { Write-Warning "Could not print intake logs: $_" }
        }
        try { Show-StageLogs } catch { Write-Warning "Could not print Stage 01-04 logs: $_" }
        try { Show-GradingLogs } catch { Write-Warning "Could not print Stage 05 logs: $_" }
        try { Show-GradingFiles } catch { Write-Warning "Could not list Stage 05 files: $_" }

        # GitHub's if: always() health steps run even when a previous step failed.
        if ($script:Python311 -and (Test-Path -LiteralPath $script:Python311 -PathType Leaf)) {
            $env:NHL_PIPELINE_JOB_STATUS = if ($script:JobFailed) { 'failure' } else { 'success' }
            $env:NHL_PIPELINE_RUN_DATE = $env:DATE
            try {
                Start-Step 'NHL Pipeline Health (always)'
                Invoke-Python -Version '3.11' -Arguments @("$NhlBase/scripts/pipeline_health.py")
            }
            catch {
                $script:JobFailed = $true
                Write-Warning "NHL Pipeline Health failed: $_"
            }
        }
        else {
            Write-Warning 'Could not run pipeline health: Python 3.11 environment was not initialized.'
            $script:JobFailed = $true
        }

        try {
            Start-Step 'Commit NHL pipeline health (always)'
            Invoke-Git -Arguments @('config', 'user.name', 'github-actions[bot]')
            Invoke-Git -Arguments @('config', 'user.email', '41898282+github-actions[bot]@users.noreply.github.com')
            Invoke-Git -Arguments @('add', '--', "$NhlBase/pipeline_health.json")
            Invoke-Git -Arguments @('add', '--', "$NhlBase/errors/pipeline_health.txt")
            Invoke-Git -Arguments @('add', '--', 'frontend/data/pipeline_health/nhl.json')
            & $script:GitExe diff --cached --quiet
            $diff = $LASTEXITCODE
            if ($diff -eq 1) { Invoke-Git -Arguments @('commit', '-m', 'chore: update NHL pipeline health') }
            elseif ($diff -ne 0) { throw "git diff failed (exit $diff)." }
            Invoke-Git -Arguments @('pull', '--rebase', 'origin', 'main')
            Invoke-Git -Arguments @('push')
        }
        catch {
            $script:JobFailed = $true
            Write-Warning "Commit NHL pipeline health failed: $_"
        }
    }
    try { Set-Location -LiteralPath $originalLocation } catch { }
    if ($script:Workspace) { Write-Host "`nLocal pipeline repository: $script:Workspace" }
}

if ($script:JobFailed) { exit 1 }
Write-Host "`nNHL 01 Pipeline completed successfully." -ForegroundColor Green
exit 0
