param(
    [Parameter(Position = 0)]
    [ValidateSet("start", "status", "tail", "stop")]
    [string]$Action = "start",

    [Parameter(Position = 1)]
    [string]$Exp = "hmap-barcode-qroi-v3",

    [string]$Remote = "yjunj@10.80.31.40",
    [string]$RemoteRepo = "/home/yjunj/projects/heatmap-model",
    [string]$Branch = "",
    [string]$RunName = "",
    [string]$CondaEnv = "hmap",
    [string]$CondaSh = "",
    [switch]$NoPush,
    [switch]$AllowDirty,
    [switch]$Follow,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$TrainArgs
)

$ErrorActionPreference = "Stop"

function Invoke-GitText {
    param([string[]]$GitArgs)
    $output = & git @GitArgs
    if ($LASTEXITCODE -ne 0) {
        throw "git $($GitArgs -join ' ') failed"
    }
    return ($output | Out-String).Trim()
}

function Quote-Bash {
    param([string]$Value)
    return "'" + $Value.Replace("'", "'`"'`"'") + "'"
}

function Invoke-RemoteBash {
    param([string]$Script)
    ssh $Remote $Script
    if ($LASTEXITCODE -ne 0) {
        throw "Remote command failed"
    }
}

$repoRoot = Invoke-GitText @("rev-parse", "--show-toplevel")
Set-Location $repoRoot

if (-not $Branch) {
    $Branch = Invoke-GitText @("branch", "--show-current")
}
if (-not $Branch) {
    throw "Current checkout is detached. Pass -Branch explicitly."
}

$session = "hmap-$Exp"

if ($Action -eq "status") {
    $remoteScript = @"
set -euo pipefail
cd $(Quote-Bash $RemoteRepo)
echo "repo: $(Quote-Bash $RemoteRepo)"
echo "branch: `$(git branch --show-current)"
echo
tmux has-session -t $(Quote-Bash $session) 2>/dev/null && echo "session: running ($(Quote-Bash $session))" || echo "session: not running ($(Quote-Bash $session))"
echo
find python/runs -maxdepth 2 -name train.log -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -5 | cut -d' ' -f2-
"@
    Invoke-RemoteBash $remoteScript
    exit
}

if ($Action -eq "tail") {
    $tailTarget = if ($RunName) { "python/runs/$RunName/train.log" } else { '$(find python/runs -maxdepth 2 -name train.log -printf "%T@ %p\n" 2>/dev/null | sort -nr | head -1 | cut -d" " -f2-)' }
    $tailFlag = if ($Follow) { "-f" } else { "-n 120" }
    $remoteScript = @"
set -euo pipefail
cd $(Quote-Bash $RemoteRepo)
log_path="$tailTarget"
if [[ -z "`$log_path" || ! -f "`$log_path" ]]; then
  echo "No train.log found. Pass -RunName or start a run first." >&2
  exit 1
fi
tail $tailFlag "`$log_path"
"@
    Invoke-RemoteBash $remoteScript
    exit
}

if ($Action -eq "stop") {
    $remoteScript = @"
set -euo pipefail
tmux kill-session -t $(Quote-Bash $session) 2>/dev/null && echo "Stopped $(Quote-Bash $session)" || echo "No running session: $(Quote-Bash $session)"
"@
    Invoke-RemoteBash $remoteScript
    exit
}

$dirty = Invoke-GitText @("status", "--porcelain")
if ($dirty -and -not $AllowDirty) {
    throw "Working tree has uncommitted changes. Commit/stash them, or pass -AllowDirty if they are unrelated to this run."
}

if (-not $RunName) {
    $RunName = "$(Get-Date -Format 'yyyyMMdd_HHmmss')_$Exp"
}

if (-not $NoPush) {
    git push origin $Branch
    if ($LASTEXITCODE -ne 0) {
        throw "git push failed"
    }
}

$extraArgs = ""
if ($TrainArgs) {
    $extraArgs = ($TrainArgs | ForEach-Object { Quote-Bash $_ }) -join " "
}
$condaShExport = ""
if ($CondaSh) {
    $condaShExport = "export CONDA_SH=$(Quote-Bash $CondaSh)"
}

$remoteStart = @"
set -euo pipefail
cd $(Quote-Bash $RemoteRepo)
git fetch origin $(Quote-Bash $Branch)
git checkout $(Quote-Bash $Branch)
git pull --ff-only origin $(Quote-Bash $Branch)
export CONDA_ENV=$(Quote-Bash $CondaEnv)
$condaShExport
export HMAP_RUN_NAME=$(Quote-Bash $RunName)
bash python/scripts/train_daemon.sh $(Quote-Bash $Exp) $extraArgs
"@

Invoke-RemoteBash $remoteStart
Write-Host ""
Write-Host "Remote training submitted."
Write-Host "Run name: $RunName"
Write-Host "Status:   powershell -ExecutionPolicy Bypass -File tools/remote_train.ps1 status $Exp"
Write-Host "Log:      powershell -ExecutionPolicy Bypass -File tools/remote_train.ps1 tail $Exp -RunName $RunName -Follow"
