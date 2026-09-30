# prototipo-deteccion-danos

Prototipo de detección de daños vehiculares para renta de autos en el mercado mexicano
(TT 27-1-0005, ESCOM-IPN). Compara un par de fotos del mismo vehículo — antes (A) y después
(B) de la renta — y localiza los daños nuevos que aparecieron en B: golpes (dent), rayones
(scratch), fisuras (crack), además de cristal roto, faros rotos y llanta ponchada.

**No se usan redes siamesas.** Cualquier mención a Siamese, a un backbone de features con
comparación por distancia o a un módulo de atención de cambios inspirado en remote sensing
quedó **obsoleta el 2026-09-29** y no debe reintroducirse. `README` (sin extensión) describía
el diseño siamés viejo y el roadmap de 5 sprints; ambos son obsoletos. `setup.py` **no**
menciona siamesas: solo tenía una descripción genérica ("detectar tallones y rayaduras").

## Arquitectura vigente (decidida 2026-09-29): 4 módulos + orquestador + interfaz

1. **M1. Preprocesamiento óptico** (`src/preprocessing/`)
   - Recibe el par A/B en JPG/PNG. Valida formato y resolución mínima 1920×1080, corrige
     orientación EXIF, reduce reflejos especulares (`specular_removal.py`), aplica CLAHE de
     forma **opcional** y devuelve metadatos (tamaño original, transformaciones aplicadas).
   - **NO redimensiona a 512×512 ni normaliza con media/desviación de ImageNet.** Los R-CNN de
     torchvision esperan RGB en [0,1] al tamaño original y hacen su propio resize (lado corto
     800, largo máx. 1333) y normalización dentro de `GeneralizedRCNNTransform`.
2. **M2. Segmentación de partes** (`src/detection/car_parts/`)
   - Segmentación de **instancias** con Mask R-CNN sobre Roboflow Car-Parts. Las 47 clases
     originales se reducen con `src/detection/car_parts/class_map.py` (ver abajo).
   - Salida por imagen: lista de `{clase_parte, confianza, bbox, máscara}`.
   - **Decisión (2026-09-29): cada par izquierda/derecha se fusiona en una sola clase**
     (`left_front_door` + `right_front_door` → `front_door`, etc.). Consecuencia: el nombre de
     la parte ya no dice el lado. M3 asigna cada daño a la **instancia** de parte con mayor
     solapamiento, y M4 debe emparejar partes de A y B **por clase y posición en la imagen**,
     no solo por nombre. Esto no está implementado todavía.
3. **M3. Detección de daños** (`src/detection/damage/` + paquete propio para la lógica)
   - Mask R-CNN entrenada en CarDD (6 clases), detecta sobre la **imagen completa**, no sobre
     recortes por parte.
   - Asigna cada daño a la parte con mayor `área(daño ∩ parte) / área(daño)`. Si ninguna
     llega a 0.5, el daño queda como "sin parte asignada" y **no se descarta**.
   - Calcula `pos_relativa`: coordenadas normalizadas del daño dentro de la caja de su parte.
   - Salida por imagen: JSON, una máscara PNG por daño y una imagen anotada:
     ```json
     {"imagen": "B", "danos": [{"id": "B-03", "tipo": "rayon", "confianza": 0.87,
       "bbox": [812, 440, 1030, 505], "area_px": 9120, "parte": "puerta_delantera_izq",
       "solapamiento": 0.93, "pos_relativa": [0.41, 0.62]}]}
     ```
4. **M4. Comparación antes/después** — **lógica determinista, no una red neuronal.**
   - Empareja primero las partes de A y B por clase y posición en la imagen (izq./der. ya no
     se distinguen por nombre).
   - Dentro de cada parte, empareja daños de A y B del mismo tipo con `pos_relativa` cercana
     (umbral de distancia) usando el método húngaro (`scipy.optimize.linear_sum_assignment`),
     no greedy.
   - Clasifica cada daño como **nuevo** (en B sin pareja en A), **preexistente** (en A y B) o
     **no visible** (en A y no en B; se marca para revisión, no se descarta).
   - Salida: reporte JSON y mapa de cambios PNG sobre B.
   - Supone que A y B son de la misma vista; lo garantiza la metodología de captura, no el código.
5. **Orquestador** A→B (M1→M2→M3→M4, cargando cada modelo una sola vez; p. ej. `src/pipeline/`)
   e **interfaz web mínima en Streamlit** (carga del par A/B y visualización del reporte).

`src/detection/common/` es el pipeline compartido de entrenamiento/inferencia de M2 y M3. La
lógica de negocio de M3 (asignación a partes, JSON) y de M4 (emparejamiento) va en paquetes
propios, sin duplicar ese pipeline.

## Metodología y calendario
Espiral de Boehm con 4 vueltas (una por módulo M1-M4), seguidas de integración, despliegue y
cierre. **Fecha final: 23 nov 2026.** El roadmap de 5 sprints (Sprint 1 Preprocesamiento …
Sprint 5 Testing) es **obsoleto**; no reflejarlo como vigente en ningún resumen.

## Estado real (rama `feature/car-parts-detection`, 2026-09-29)
| Pieza | Estado |
|---|---|
| M1 | **Parcial.** `specular_removal.py` funciona y tiene pruebas. `normalization.py` hace resize 512×512 + ImageNet (incompatible, ver abajo). Falta validación de formato/resolución, EXIF, metadatos, pipeline encadenado y pruebas. Ningún script de entrenamiento o inferencia lo usa. |
| M2 | `car_parts/v1` = Faster R-CNN, **solo cajas** (no cumple la spec). `car_parts/v2` (Mask R-CNN) tiene config, entrypoint (`train_masks.py`) y `class_map` cerrado (29 clases finales), **pero no hay modelo entrenado**. Umbral de confianza de partes y manejo de partes solapadas: **no definidos**. |
| M3 | Hay modelo: `damage/v1` (val: bbox AP 0.5191, mask AP 0.5020, mejor época 11, sin augmentation). **No existe** el módulo de asignación a partes ni el JSON. |
| M4 | No existe. |
| Orquestador / Streamlit | No existen. |
| Pares A/B con ground truth | No existen (ver "Gap conocido"). |

## Hallazgos técnicos (medidos, no supuestos)
- **(a) BatchNorm no congelada.** `maskrcnn_resnet50_fpn_v2` usa `nn.BatchNorm2d` (no
  `FrozenBatchNorm2d`): 69 capas (53 backbone, 8 FPN, 4 box head, 4 mask head), igual para
  48 y 7 clases. El trainer solo llama `model.train()`, así que con batch 2 las estadísticas
  se recalculan con muy pocas imágenes. `conv1`/`layer1` tienen pesos congelados
  (`requires_grad=False`) pero **sus `running_mean`/`running_var` sí cambian** en cada forward
  de entrenamiento (verificado). **No está medido si esto afecta el AP.** `--freeze-bn`
  (apagado por defecto) deja las BN en `eval()`; se decidirá con una corrida A/B en car_parts.
- **(b) Semilla fija, no reproducible.** `--seed` fija `random`, `numpy`, `torch` y
  `torch.cuda`, pero no activa `cudnn.deterministic` ni `torch.use_deterministic_algorithms`.
  Descríbase como "semilla fija", nunca como "reproducible".
- **(c) mAP viejo de car_parts/v1 no comparable.** Su `val_map50 = 0.7878` es la métrica casera
  (mAP@0.5, `engine.evaluate`), no COCO AP. La comparación válida de car_parts es en **test**
  con `tests/detection/evaluate_test_set.ipynb`: v1 → AP 0.5389, AP50 0.7689.
- **(d) Splits de partes pequeños.** val = 32 imágenes con 5 clases sin instancias
  (`air_intake`, `left_windowark`, `right_glass`, `right_windowark`, `shield`); test = 16 con 6
  (`fog_lights`, `left_side_door`, `left_windowark`, `right_glass`, `right_windowark`, `shield`).
  Los AP de partes son **indicativos**.
- **(e) Reentrenar sobre la misma carpeta pierde el anterior.** `best_model.pth` se sobrescribe
  al terminar la primera época; `last_model.pth`, `metrics.json`, `status.json` y
  `run_info.json` se reemplazan; `train.log` se **anexa** (quedan corridas mezcladas).
  `models/checkpoints/` está en `.gitignore`: no hay recuperación. `damage/train.py` y
  `car_parts/train.py` escriben por defecto en `damage/v1` y `car_parts/v1` — **siempre pasar
  `--output` a una carpeta nueva**.
- **(f) Train de partes = 111 fotos × 3 copias aumentadas por Roboflow** (ruido "sal",
  recortes negros; mismo prefijo antes de `_jpg.rf.`). Los conteos de train están inflados ×3:
  "3 instancias" es una sola foto. valid (32) y test (16) no tienen copias, y ninguna foto base
  se repite entre splits (sin fuga).
- **(g) AP de v1 en test según el conjunto de clases.** Sobre las 47 clases (41 con instancias
  en test): AP 0.5389 / AP50 0.7689, reproducido exactamente por `compare_versions`.
  Excluyendo las 9 clases que el `class_map` descarta: 0.5746 / 0.8075. Mapeado a las 29
  clases finales: 0.5757 / 0.8240 (diferencia +0.0011, IC 95 % [−0.026, +0.033]: reducir
  clases no cambia el AP de v1 de forma distinguible).
- **(h) En Roboflow Car-Parts, `windshield` son los limpiaparabrisas**, no el parabrisas (franja
  delgada al pie del cristal; nunca se solapa con `front_glass`, que es el cristal). El
  `class_map` descarta la etiqueta original `windshield` y renombra `front_glass` → `windshield`.

## Métricas que pide el protocolo
Precision, Recall, F1, mAP, IoU y Accuracy. Para los detectores (M2, M3), COCO AP vía
`pycocotools` es la métrica principal (`src/detection/common/coco_eval.py`). Para M4 y el
sistema end-to-end (nuevo/preexistente/no visible por daño) hacen falta P, R, F1 y Accuracy
contra ground truth propio.

## Environment
**Hay dos virtualenvs. `venv_cuda/` es el que se usa.**

- `venv_cuda/` — **entorno canónico.** Python 3.9.13, `torch 2.7.1+cu118` /
  `torchvision 0.22.1+cu118` / `torchaudio 2.7.1+cu118`, `numpy 2.0.2`,
  `opencv-python 5.0.0.93`, `matplotlib 3.9.4`, `pytest 8.4.2`, `pycocotools 2.0.11` y Jupyter.
  Todo se corre con `./venv_cuda/Scripts/python.exe`.
- **GPU disponible**: NVIDIA GeForce RTX 3050 6GB Laptop, CUDA 11.8. Verificar con
  `./venv_cuda/Scripts/python.exe check_cuda.py`. Mask R-CNN ResNet50-FPN v2 entrena con
  `--batch-size 2` + AMP (pico medido ~2.25 GB en partes).
- `venv/` — **legado, no usar.** Python 3.14.6, `torch 2.13.0+cpu`, sin CUDA.
- **`requirements-cuda.txt` es el único archivo de dependencias.** `requirements.txt` se
  eliminó (sus pins, `torch==2.0.1` etc., no correspondían a nada instalado).
- `setup.py` declaraba `python_requires=">=3.10"` con el entorno en 3.9.13; ya dice `">=3.9"`
  (no se ha vuelto a correr `pip install -e .`; el editable install actual es anterior).
- Gotcha histórico: `NotImplementedError: Could not run 'torchvision::...'` = par
  torch/torchvision desalineado, no un bug del código.
- Pruebas desde la raíz: `./venv_cuda/Scripts/python.exe -m pytest -q`. `from src...` resuelve
  por el editable install; **no** hay `conftest.py` raíz. Las pruebas de dataset se saltan
  solas si el dataset no está en `data/raw/`.

## Datasets (`data/raw/`, en `.gitignore`)
1. **`CarDD_release/CarDD_release/CarDD_COCO/`** — daños, 6 categorías (ids 1-6: dent,
   scratch, crack, glass shatter, lamp broken, tire flat). `train2017/` (2816 imgs / 6211
   anns), `val2017/` (810 / 1744), `test2017/` (374 / 785) — el split se llama `val`. JSONs en
   `annotations/instances_{split}.json`. Un polígono por anotación, sin RLE ni crowds.
   Instancias en train: scratch 2560, dent 1806, crack 651, lamp broken 494, glass shatter
   475, tire flat 225. Hay un `CarDD_SOD/` que nada usa.
2. **`Car parts coco-segmentation/`** — partes (Roboflow), 47 categorías + id 0 raíz
   excluido. `train/` 333 imgs / 8439 anns (6 imágenes sin anotaciones, que el dataset
   descarta → 327 efectivas), `valid/` 32 / 827, `test/` 16 / 419. **El 100% de las
   anotaciones son polígonos válidos** (sin RLE, sin solo-caja, sin polígonos <3 puntos, sin
   máscaras vacías al rasterizar).
3. **`Car-Parts-Segmentation/`** — otro dataset de partes (19 categorías), repo vendorizado de
   terceros; no modificar. Solo `trainingset/` y `testset/`.

**Gap conocido: no hay pares A/B con ground truth.** Para medir M4 y el sistema completo hace
falta capturar y anotar 20-40 pares propios marcando qué daños son nuevos en B.

## Notebooks
- `notebooks/samples_visualization*.ipynb` — uno por dataset; comparten `show_samples()` /
  `load_coco_annotations()` duplicados (un bug en uno probablemente está en los tres). Usar
  `matplotlib.colormaps[name].resampled(n)`, no `cm.get_cmap`.
- `notebooks/demo_avances.ipynb` — demo de avances; importa `apply_clahe_rgb`,
  `resize_with_padding` y `specular_removal` solo para visualizar.
- `tests/detection/evaluate_test_set.ipynb` — evalúa `car_parts/v1` en test y guarda, para las
  primeras 6 imágenes, un JSON de M2 por imagen en `tests/detection/outputsM2/` (en
  `.gitignore`): `{imagen, image_id, width, height, checkpoint, score_threshold,
  partes: [{clase_parte, confianza, bbox, mascara: null}]}` con umbral 0.5. Es un prototipo
  del formato de M2 en notebook, no un módulo.
- `tests/preprocessing/*_before_after.ipynb` — demos de M1.

Las rutas en notebooks se resuelven relativas a la raíz del proyecto, no hardcodeadas.

## `src/preprocessing/` — M1
- `specular_removal.py` — Shen & Cai (2009), "maximum diffuse chromaticity", más un gating por
  brillo propio (no del paper) para no oscurecer superficies grises/negras (llantas, asfalto).
  Parámetros documentados en `ShenParams`. CLI:
  `python -m src.preprocessing.specular_removal <input_dir> <output_dir> [-n N]`. Probado en
  `tests/preprocessing/test_specular_removal.py`.
- `normalization.py` — `preprocess_pipeline()` = CLAHE (canal L en LAB) → letterbox a 512×512
  → ImageNet mean/std + CHW. **Contradice la spec de M1**: no conectarlo a los detectores. Se
  debe quitar el resize y la normalización del camino hacia los detectores (queda CLAHE como
  paso opcional). Sin pruebas.
- Los modelos actuales se entrenaron con imágenes **crudas** (`coco_dataset.py`:
  `Image.open(...).convert("RGB")` + `to_tensor`). Pendiente: medir AP con y sin
  preprocesamiento.

## `src/detection/`
**Dos detectores, un pipeline compartido.** Lo dataset-agnóstico vive en `common/`; cada
detector es un `DetectorConfig` más wrappers de CLI. No agregar otra copia del pipeline:
agregar una config.

```
src/detection/
├── common/     config.py · coco_dataset.py · model.py · engine.py · trainer.py
│               coco_eval.py · transforms.py · predict.py · visualize.py
│               audit.py · watch.py
├── car_parts/  config.py (CONFIG = v1 Faster R-CNN; MASK_CONFIG = v2 Mask R-CNN)
│               train.py (v1) · train_masks.py (v2) · predict.py
└── damage/     config.py · train.py · predict.py   (CarDD, Mask R-CNN)
```

CLIs (siempre con `./venv_cuda/Scripts/python.exe`):
`-m src.detection.car_parts.train`, `-m src.detection.car_parts.train_masks`,
`-m src.detection.damage.train`, `-m src.detection.{car_parts,damage}.predict`.
Runbook completo: `docs/reentrenar_modelos.txt`.

### `common/`
- `config.py` — `DetectorConfig` (frozen dataclass): rutas, splits, `exclude_category_ids`,
  `with_masks`, `arch`, defaults de CLI y `default_select_by` (`"bbox"` por defecto).
- `coco_dataset.py` — `CocoDetectionDataset`; `image_id` es el id COCO original; ids de
  categoría usados tal cual (0 = fondo), lo que requiere ids contiguos desde 1. Máscaras
  rasterizadas con `cv2.fillPoly`; RLE lanza `TypeError`. `skip_empty=True` por defecto.
  - `class_map=` (nombre original → nombre final o `None`) se aplica al cargar vía
    `apply_class_map()`: renombra, fusiona y descarta; ids finales = nombres finales en orden
    alfabético → 1..N (`final_categories()`). Falla si el mapa no cubre exactamente las
    categorías del archivo. Las imágenes que quedan sin anotaciones se tratan como vacías y se
    reportan en `class_map_stats["images_emptied"]` (hoy 0 en los tres splits).
  - `dataset.coco_gt` es el dict COCO (remapeado si hay mapa): **usarlo como ground truth de
    COCOeval**, no el JSON en disco, cuyos ids ya no coinciden con los del modelo.
  - `build_dataset(cfg, split, use_class_map=True)` aplica `cfg.class_map`.
- `model.py` — `build_model(num_classes, arch)` → `fasterrcnn_resnet50_fpn_v2` o
  `maskrcnn_resnet50_fpn_v2`, preentrenados en COCO con cabezas reemplazadas.
- `engine.py` — `train_one_epoch()` (AMP opcional, callback `on_batch`) y `evaluate()`,
  mAP@0.5 casero de respaldo, **no comparable** con COCO AP.
- `coco_eval.py` — COCO AP (`bbox` y `segm`) vía pycocotools; **la métrica de la tesis**.
  `per_category_ap()` da AP por clase. `load_coco_gt()` acepta ruta o dict; el trainer evalúa
  contra `valid_dataset.coco_gt`. **El dict se copia (deepcopy)**: `COCOeval` con
  `iouType="segm"` reescribe in situ los polígonos del GT a RLE, y el dataset comparte esas
  anotaciones; sin la copia, la época 2 fallaba con "RLE segmentation is not supported"
  (pasó en la primera corrida de car_parts/v2, 2026-09-29).
- `trainer.py` — CLI y loop compartidos. Flags relevantes:
  - `--seed N` (default 42; `-1` = sin semilla) → ver hallazgo (b).
  - `--select-by bbox|segm` — qué COCO AP@0.5:0.95 elige `best_model.pth`. Default de la
    config: `segm` en `car_parts` v2 (`MASK_CONFIG`), `bbox` en `car_parts` v1 y en `damage`.
    `validate_args()` rechaza `segm` sin máscaras o con `--metric simple`.
  - `--freeze-bn` (default off) — pone las `BatchNorm2d` en `eval()` tras cada
    `model.train()` (`model.freeze_batchnorm_stats`, llamado en `engine.train_one_epoch`).
    Se eligió `eval()` y no `FrozenBatchNorm2d` porque este último cambia las claves del
    `state_dict` y rompería `load_checkpoint`. Queda en `run_info.json` y en el checkpoint.
    Probado en `tests/detection/test_freeze_bn.py`.
  - `--no-class-map` — ignora `cfg.class_map` y entrena con las categorías originales (para
    reproducir las 47 clases de v1). El mapa aplicado (o `None`) y sus estadísticas por split
    quedan en `run_info.json` (`class_map`, `class_map_stats`) y el mapa en el checkpoint
    (`class_map`); el payload lo arma `build_checkpoint_payload()`.
  - `--eval-every`, `--lr-step-size`, `--max-train-images`, `--max-val-images`, `--arch`,
    `--no-amp`, `--no-augment`, `--hflip-prob`, `--jitter-prob`, `--metric coco|simple`.
  - Checkpoints autodescriptivos: `num_classes`, `categories`, `arch`, `with_masks`,
    `class_map`, `detector`, `metric`, `select_by`, `seed`, `freeze_bn`, `augmented`, `epoch`, `val_map`, `val_map50`,
    `coco`. Los anteriores al refactor no tienen `arch` y se cargan como `faster_rcnn`.
- `audit.py` — `RunAuditor`. Cada corrida escribe en `--output`:
  - `run_info.json` (una vez): args, config, commit de git + si el árbol estaba sucio,
    versiones, GPU, AMP, augmentation, tamaños de dataset e instancias por clase.
  - `status.json` (cada ~2 s, escritura atómica): fase, época, batch, % de época y total,
    loss por término, LR, memoria GPU, ETAs, mejor época; estado final
    `completed`/`interrupted`/`failed` (con traceback).
  - `train.log`: todo lo impreso, con fecha y hora.
  - `metrics.json` por época: loss total y por término, LR, tiempo, pico de GPU, `is_best`,
    bloque `coco`.
- `watch.py` — `python -m src.detection.common.watch <output_dir> [--interval N] [--once]`:
  muestra época, batch, barras de avance, losses, ETA, mejor época e historial; avisa si
  `status.json` lleva >2 min sin actualizarse o si hay losses NaN/inf.
- `transforms.py` — augmentation de train (flip horizontal + jitter de color), activa por
  defecto; nunca en validación.
- `predict.py` / `visualize.py` — `load_checkpoint()` reconstruye la arquitectura desde el
  checkpoint; `run_prediction()` filtra por `--score-threshold` (default 0.5) y **solo dibuja**
  imágenes anotadas. No produce el JSON de M2 ni de M3.

### `car_parts/` — M2
Dataset #2; id 0 excluido → 47 clases originales. `CONFIG` (v1, Faster R-CNN, solo cajas,
sin `class_map`) se conserva para reproducir v1. `MASK_CONFIG` (v2): `with_masks=True`,
`arch="mask_rcnn"`, salida por defecto `car_parts/v2`, selección por mask AP y
`class_map=CLASS_MAP`. Entrenar con `train_masks.py`.

- `class_map.py` — `CLASS_MAP`, tabla **por nombre** (auditable) de las 47 clases originales
  a su clase final o `None`. **Cerrado el 2026-09-29 → 29 clases finales** (30 salidas con
  fondo); `PENDING_DECISIONS` está vacío.
  - Descartadas (9): por tener <10 instancias en train, `air_intake`, `left_side_door`,
    `left_windowark`, `right_windowark`, `right_glass`, `shield`; por decisión del usuario con
    la muestra visual, `back_light` (anotación inconsistente: a veces el medallón trasero, a
    veces un reflejante en la defensa), `front_mirror` y `windshield` (limpiaparabrisas, ver
    hallazgo (h)).
  - Fusionadas: 8 pares izq./der. (`front_door`, `back_door`, `front_door_glass`,
    `back_door_glass`, `quarter_glass`, `headlight`, `taillight`, `fog_light`), más
    `fog_lights` → `fog_light`.
  - Renombradas: `front_glass` → `windshield` (el cristal), `side_steps` → `side_step`.
  - `bumper` y `back_bumper` se mantienen separadas; el resto queda igual.
  - Clases finales con <10 instancias en val o test (AP poco fiable): `back_bumper`,
    `back_glass`, `emblem`, `fog_light`, `hood`, `indicator_light`, `mudguard`,
    `roof_trunk`, `step`, `trunk`.
- `compare_versions.py` — `python -m src.detection.car_parts.compare_versions --v1 V1.pth
  --v2 V2.pth --split test --output DIR`. Experimento A: v1 en sus clases originales contra v1
  con predicciones mapeadas por el `class_map` (NMS dentro de cada clase final, IoU 0.5).
  Experimento B: v1 mapeado contra v2, mismas imágenes y clases finales, más el mask AP de v2.
  AP y AP50 promediados solo sobre las clases del experimento (`params.catIds`), AP por clase
  con instancias, bootstrap pareado por imagen (1000, semilla 0, IC 95 % percentil) del AP de
  cada modelo y de las diferencias; clases con <10 instancias marcadas "poco fiable". Escribe
  `compare_{split}.json` y `.md`. **Selección con val, veredicto con test**: con `--split valid`
  lo advierte. Reimplementa `COCOeval.accumulate` con pesos por imagen porque
  `COCOeval.evaluate` elimina `imgIds` repetidos; las pruebas verifican igualdad exacta con
  pycocotools (bbox y segm). Probado con v1 contra sí mismo en test: diferencia B = 0 exacta.

### `damage/` — M3 (solo el detector)
Dataset #1 (CarDD), 7 clases. Mask R-CNN porque un rayón es largo, delgado y diagonal: su caja
es casi todo fondo. Selecciona por bbox AP salvo `--select-by segm`.

### Pendiente
- **M3**: asignación daño→parte (umbral 0.5), `pos_relativa`, JSON + PNGs, como paquete propio
  con pruebas sobre máscaras sintéticas.
- **M4**: emparejamiento húngaro + reporte + mapa de cambios, paquete propio con pruebas sobre
  JSON sintéticos.
- **M2**: definir umbral de confianza de partes y cómo resolver partes que se solapan (hoy el
  NMS de torchvision es por clase y no resuelve solapes entre clases distintas).
- **Orquestador** y **Streamlit**.

## Checkpoints (`models/checkpoints/`, en `.gitignore`)
- `car_parts/v1/` — Faster R-CNN, solo cajas, 20 épocas, sin augmentation. Checkpoint viejo:
  solo `categories, epoch (19), num_classes, val_map50 (0.7878, métrica casera)`. En **test**
  (16 imgs, COCO): AP 0.5389, AP50 0.7689, AP75 0.6133, AR@100 0.6164. Peores clases:
  `air_intake`, `back_light` (0.000), `step` (0.100). Se estanca en val desde la época 11-12.
- `car_parts/smoke_test/`, `car_parts/smoke_test_masks/` — pruebas de pipeline, no modelos.
- `car_parts/v2_interrumpido_smoke/` — dos corridas de v2 interrumpidas y mezcladas (17 s, y
  1 época: bbox AP 0.3082, mask AP 0.2958, 105 s/época). No es un modelo usable.
- `car_parts/v2_fallido_rle/` — primera corrida real de v2 (29 clases), falló en la época 2 por
  el bug de RLE ya corregido en `7bbf1b9`. Época 1 en val: box AP 0.487, mask AP 0.477.
  `car_parts/v2/` no existe: queda libre para la corrida real.
- `car_parts/_muestra_clases/` — muestra visual (4 imágenes de train por clase dudosa, con
  máscaras y nombre) con la que se cerró el `class_map`. Una hoja de contacto por grupo.
- `car_parts/_compare_v1_vs_v1/` — salida de `compare_versions` con v1 contra sí mismo en test.
- `damage/v1/` — Mask R-CNN, 12 épocas, `--lr-step-size 8`, AMP, batch 2, **sin
  augmentation**, 315 min. Seleccionado por bbox AP; mejor época 11 (LR 0.0005, deducido del
  StepLR; `metrics.json` de v1 no guarda LR), que también es la mejor por mask AP.
  - **Val**: bbox AP 0.5191 / AP50 0.6873; mask AP 0.5020 / AP50 0.6735.
  - **Test** (374 imgs): bbox AP 0.5331, AP50 0.7138; mask AP 0.4949, AP50 0.6830.
  - AP bbox por clase (test): glass shatter 0.867, tire flat 0.837, lamp broken 0.570,
    **dent 0.333, scratch 0.321, crack 0.271** — las tres clases objetivo de la tesis son las
    peores. AP small 0.256 / medium 0.229 / large 0.539.
  - Curva: meseta ~0.45 en épocas 4-8; la bajada de LR en la 8 la sube a ~0.51; plana desde
    la 9. Más épocas con esta receta no ayudan.
- `damage/smoke_test/` — prueba de pipeline.

## Pruebas
`tests/detection/` (`test_coco_dataset.py`, `test_model.py`, `test_config.py`,
`test_coco_eval.py`, `test_transforms.py`, `test_audit.py`, `test_freeze_bn.py`,
`test_class_map.py`, `test_compare_versions.py`) y
`tests/preprocessing/test_specular_removal.py`. Resultado actual: ver `pytest -q`.
`tests/detection/manual_inference.py` es un demo con GUI, no se colecta.

## Plan de trabajo acordado (registrado, no ejecutado salvo donde se indica)
0. ~~Cerrar el `class_map`~~ — hecho (29 clases).
1. Entrenamientos: `car_parts/v2` (Mask R-CNN, con augmentation, carpeta nueva; comparar con
   v1 usando `compare_versions` en test) y `damage/v2`
   (augmentation, `--epochs 12 --lr-step-size 8`, `--output damage/v2`). Considerar una
   corrida A/B de BatchNorm en car_parts antes de fijar la receta.
2. M3: asignación a partes + JSON, con pruebas.
3. M4: emparejamiento + reporte, con pruebas.
4. M1 compatible con los detectores (sin resize 512 ni ImageNet) + experimento de AP con y sin
   preprocesamiento.
5. Orquestador + Streamlit.
6. Evaluación end-to-end con pares A/B propios (P, R, F1, Accuracy).
7. Limpieza de `README`, `setup.py` y `requirements.txt`: hecha.

## Reglas de trabajo
- Preguntar antes de editar código; no hacer push, entrenamientos ni borrados sin
  confirmación explícita del usuario.
- Commits de una sola línea y **sin** trailers de coautoría/atribución.
- Todo lo común a los dos detectores va en `src/detection/common/`; M3 y M4 en paquetes propios.
- Todo módulo nuevo lleva pruebas con pytest.
- Mantener `docs/reentrenar_modelos.txt` al día con cada entrenamiento.
- Cada entrenamiento en una carpeta `--output` nueva (hallazgo (e)).
- `AGENTS.md` se versiona; `CLAUDE.md` es una copia idéntica que queda fuera del repo
  (`.gitignore`). Al cambiar uno, copiar al otro.
