Param(
    [string]$Script = "src/v5_neural_vae.py",
    [string]$Config = "configs/v5_base.txt",
    [string]$RunId = "",
    [string]$Note = "",
    [string]$LlmSummaryCmd = ""
)

$runner = Join-Path $PSScriptRoot "run_experiment.py"

if ($LlmSummaryCmd) {
    $env:LLM_SUMMARY_CMD = $LlmSummaryCmd
}

$argsList = @($runner, "--script", $Script, "--config", $Config)
if ($RunId) { $argsList += @("--run-id", $RunId) }
if ($Note) { $argsList += @("--note", $Note) }

& python @argsList
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
