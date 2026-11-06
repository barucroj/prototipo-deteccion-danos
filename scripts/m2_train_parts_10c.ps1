# Entrena los modelos de partes en 10 clases (car_parts/v3_10c y car_parts/v3_mix), uno tras
# otro, y abre para cada uno el monitor de avance en otra ventana.
#
# En esta ventana se ve lo del trainer: [Epoch N/20], la barra de Training y la de evaluacion,
# loss total y por termino, box/mask AP global y por clase, tiempo y ETA. El monitor
# (src.detection_common.watch) muestra epoca, batch, fase, losses, LR, GPU, ETA, mejor epoca e
# historial. Todo queda en la carpeta de cada modelo: metrics.json (por epoca, con AP por
# clase), batches.jsonl (loss por batch), status.json, train.log y run_info.json.
#
# Uso, desde la raiz del proyecto:
#   powershell -ExecutionPolicy Bypass -File scripts\m2_train_parts_10c.ps1             # los dos
#   powershell -ExecutionPolicy Bypass -File scripts\m2_train_parts_10c.ps1 -Model mix  # uno
#
# Receta (decidida 2026-11-06): 20 epocas, LR 0.005 que baja x0.1 en la epoca 15
# (--lr-step-size 14), batch 2, AMP, augmentation, semilla 42, seleccion por mask AP en el val
# de Roboflow + HITL. Estimado en la RTX 3050: v3_10c ~1 h, v3_mix ~2 h 40 min.
param(
    [ValidateSet("both", "10c", "mix")]
    [string]$Model = "both",
    [int]$Epochs = 20,
    [int]$LrStepSize = 14,
    [string]$Output10c = "models\checkpoints\car_parts\v3_10c",
    [string]$OutputMix = "models\checkpoints\car_parts\v3_mix",
    [switch]$NoWatch,
    # Cualquier flag del trainer puesto al final pasa tal cual, p. ej. (solo pruebas):
    #   ... -Model mix -Epochs 2 -OutputMix models\checkpoints\car_parts\smoke_x --max-train-images 40
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Extra = @()
)
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Python = Join-Path $Root "venv_cuda\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "No se encontro $Python (el entorno canonico es venv_cuda)." }

$runs = @()
if ($Model -in @("both", "10c")) { $runs += , @("src.m2_parts.train_masks_10c", $Output10c) }
if ($Model -in @("both", "mix")) { $runs += , @("src.m2_parts.train_masks_mix", $OutputMix) }

# Se revisan todas las carpetas antes de empezar, para no fallar a mitad de la noche.
foreach ($run in $runs) {
    if (Test-Path (Join-Path $run[1] "best_model.pth")) {
        throw "$($run[1]) ya tiene un best_model.pth. Reentrenar ahi lo sobrescribe: usa otra carpeta."
    }
}
if (-not (Test-Path "data\raw\parts_mix_coco\instances_train.json")) {
    throw "Falta data\raw\parts_mix_coco: corre antes  python -m src.m2_parts.mix_dataset"
}

$dirty = git status --porcelain --untracked-files=no
if ($dirty) {
    Write-Warning "Hay cambios sin commitear; run_info.json registrara dirty=true:`n$($dirty -join "`n")"
}

foreach ($run in $runs) {
    $module, $output = $run
    Write-Host "`n=== $module -> $output ($Epochs epocas, LR baja en la $($LrStepSize + 1)) ===`n"
    if (-not $NoWatch) {
        $watch = "Set-Location '$Root'; & '$Python' -m src.detection_common.watch '$output'"
        Start-Process powershell -ArgumentList "-NoExit", "-Command", $watch
    }
    & $Python -m $module --epochs $Epochs --lr-step-size $LrStepSize --batch-size 2 --lr 0.005 `
        --seed 42 --select-by segm --output $output @Extra
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "$module termino con codigo $LASTEXITCODE; no se lanza el siguiente."
        exit $LASTEXITCODE
    }
}
exit 0
