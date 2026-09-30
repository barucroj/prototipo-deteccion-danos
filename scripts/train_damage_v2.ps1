# Entrena damage/v2 (Mask R-CNN sobre CarDD) y abre el monitor de avance en otra ventana.
#
# El monitor es el mismo de car_parts (src.detection.common.watch): epoca, batch, barras de
# avance, loss por termino, LR, memoria GPU, ETA, mejor epoca e historial de box/mask AP.
#
# Uso, desde la raiz del proyecto:
#   powershell -ExecutionPolicy Bypass -File scripts\train_damage_v2.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\train_damage_v2.ps1 -Output models\checkpoints\damage\v3
#
# Receta: la de damage/v1 (12 epocas, StepLR cada 8) + augmentation (default) y seleccion del
# mejor checkpoint por mask AP. Unas 5.5 h en la RTX 3050: conviene lanzarlo de noche.
param(
    [string]$Output = "models\checkpoints\damage\v2",
    [int]$Epochs = 12,
    [int]$LrStepSize = 8,
    [switch]$NoWatch,
    # Flags extra para el trainer, p. ej. -Extra "--max-train-images","40" (solo pruebas).
    [string[]]$Extra = @()
)
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Python = Join-Path $Root "venv_cuda\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "No se encontro $Python (el entorno canonico es venv_cuda)." }

if (Test-Path (Join-Path $Output "best_model.pth")) {
    throw "$Output ya tiene un best_model.pth. Reentrenar ahi lo sobrescribe: usa otra carpeta con -Output."
}

$dirty = git status --porcelain --untracked-files=no
if ($dirty) {
    Write-Warning "Hay cambios sin commitear; run_info.json registrara dirty=true:`n$($dirty -join "`n")"
}

if (-not $NoWatch) {
    $watch = "Set-Location '$Root'; & '$Python' -m src.detection.common.watch '$Output'"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $watch
}

& $Python -m src.detection.damage.train --epochs $Epochs --lr-step-size $LrStepSize `
    --select-by segm --output $Output @Extra
exit $LASTEXITCODE
