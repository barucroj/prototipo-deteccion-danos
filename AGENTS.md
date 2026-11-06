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

1. **M1. Preprocesamiento óptico** (`src/m1_preprocessing/`)
   - Recibe el par A/B en JPG/PNG. Valida formato y resolución mínima 1920×1080, corrige
     orientación EXIF, reduce reflejos especulares (`specular_removal.py`), aplica CLAHE de
     forma **opcional** y devuelve metadatos (tamaño original, transformaciones aplicadas).
   - **NO redimensiona a 512×512 ni normaliza con media/desviación de ImageNet.** Los R-CNN de
     torchvision esperan RGB en [0,1] al tamaño original y hacen su propio resize (lado corto
     800, largo máx. 1333) y normalización dentro de `GeneralizedRCNNTransform`.
2. **M2. Segmentación de partes** (`src/m2_parts/`)
   - Segmentación de **instancias** con Mask R-CNN sobre Roboflow Car-Parts. Las 47 clases
     originales se reducen con `src/m2_parts/class_map.py` (ver abajo).
   - Salida por imagen: lista de `{clase_parte, confianza, bbox, máscara}`.
   - **Decisión (2026-09-29): cada par izquierda/derecha se fusiona en una sola clase**
     (`left_front_door` + `right_front_door` → `front_door`, etc.). Consecuencia: el nombre de
     la parte ya no dice el lado. M3 asigna cada daño a la **instancia** de parte con mayor
     solapamiento, y M4 debe emparejar partes de A y B **por clase y posición en la imagen**,
     no solo por nombre. Esto no está implementado todavía.
3. **M3. Detección de daños** (`src/m3_damage/detector/` + `src/m3_damage/assignment/` para la lógica)
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
   - A y B deben ser la misma vista: lo pide la metodología de captura y lo comprueba
     `view_check.py`; un par que no pasa no se clasifica salvo que el agente lo fuerce.
5. **Orquestador** A→B (M1→M2→M3→M4, cargando cada modelo una sola vez; p. ej. `src/pipeline/`)
   e **interfaz web mínima en Streamlit** (carga del par A/B y visualización del reporte).

`src/detection_common/` es el pipeline compartido de entrenamiento/inferencia de M2 y M3. La
lógica de negocio de M3 (asignación a partes, JSON) y de M4 (emparejamiento) va en paquetes
propios, sin duplicar ese pipeline.

## Metodología y calendario
Espiral de Boehm con 4 vueltas (una por módulo M1-M4), seguidas de integración, despliegue y
cierre. **Fecha final: 23 nov 2026.** El roadmap de 5 sprints (Sprint 1 Preprocesamiento …
Sprint 5 Testing) es **obsoleto**; no reflejarlo como vigente en ningún resumen.

## Estado real (rama `feature/m4-report`, 2026-11-03)
| Pieza | Estado |
|---|---|
| M1 | **Hecho.** Validación de formato/resolución, carga con EXIF, reflejos y CLAHE opcionales, metadatos (`pipeline.py`), sin resize ni ImageNet; transform para entrenar/evaluar con M1 y experimento de AP, con pruebas. Resultado: con los modelos vigentes, M1 **no mejora** el AP (los reflejos lo empeoran; ver la sección de M1). `normalization.py` (512×512 + ImageNet) sigue en el repo pero fuera del camino hacia los detectores. |
| M2 | `car_parts/v1` = Faster R-CNN, **solo cajas** (no cumple la spec). `car_parts/v2` (Mask R-CNN, 29 clases finales) **entrenado** el 2026-09-29 (ver Checkpoints): test box AP 0.6041, mask AP 0.5779; supera a v1 en AP de caja (+0.028, IC 95 % [+0.008, +0.047]) pero no en AP50. Umbral de confianza de partes y manejo de partes solapadas: **no definidos**. |
| M3 | Modelo vigente: `damage/v2` (test: box AP 0.5275, mask AP 0.5075; frente a `damage/v1` las diferencias no son distinguibles del ruido, ver Checkpoints). Asignación a partes **hecha** (`src/m3_damage/assignment/`): parte con mayor solapamiento (mín. 0.5, partes con confianza ≥ 0.7), `pos_relativa`, JSON con `parte_id` por instancia, PNG por daño e imagen anotada, con pruebas. |
| M4 | **Código completo** (2026-10-30): `src/m4_comparison/` empareja partes, clasifica cada daño como nuevo/preexistente/no visible, arma el reporte JSON de la sección 4.9.2 y dibuja el mapa de cambios PNG; conectado al orquestador. **Primera medición** (2026-11-03) con `batch1` (11 pares, 13 daños, todos nuevos; ver "Evaluación con pares A/B"): con umbral de daños 0.5, nuevos P 0.500 / R 0.615 / F1 0.552; el barrido da el mejor F1 en 0.75 (P 0.889 / R 0.615 / F1 0.727), **elegido sobre el mismo lote** (calibración, no prueba). Las distancias (0.25 partes, 0.15 daños) siguen **sin calibrar**: `batch1` no tiene daños preexistentes. |
| Orquestador / Streamlit | **Flujo completo** (2026-10-30): `src/pipeline/orchestrator.py` corre M1→M2→M3→M4 y `src/app/streamlit_app.py` muestra resumen, mapa y tabla por daño. La interfaz se renderizó con `AppTest` y un resultado real, pero la carga de archivos sigue **sin probarse a mano en navegador**. |
| Pares A/B con ground truth | `batch1` (2026-11-03): 11 pares semisintéticos (daño agregado con IA en B), CVAT → COCO 1.0. Faltan pares sin cambios, con daños preexistentes y un lote de prueba separado. |
| Evaluación | `src/evaluation/` + `scripts/evaluate_pairs.py` + `scripts/sweep_damage_threshold.py` + modo "Evaluar un lote etiquetado" en Streamlit, con pruebas. **Sin commitear** al 2026-11-03 (el usuario lo prueba antes). |

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
  (mAP@0.5, `engine.evaluate`), no COCO AP. La comparación válida de car_parts es COCO AP en
  **test** (`compare_versions` o `tests/m2_parts/evaluate_test_set_v2.ipynb`, que coinciden):
  v1 → AP 0.5389, AP50 0.7689. El notebook viejo `evaluate_test_set.ipynb` **no calcula AP**
  (solo conteos) y no es la fuente de ese número.
- **(d) Splits de partes pequeños.** val = 32 imágenes con 5 clases sin instancias
  (`air_intake`, `left_windowark`, `right_glass`, `right_windowark`, `shield`); test = 16 con 6
  (`fog_lights`, `left_side_door`, `left_windowark`, `right_glass`, `right_windowark`, `shield`).
  Los AP de partes son **indicativos**.
- **(e) Reentrenar sobre la misma carpeta pierde el anterior.** `best_model.pth` se sobrescribe
  al terminar la primera época; `last_model.pth`, `metrics.json`, `status.json` y
  `run_info.json` se reemplazan; `train.log` se **anexa** (quedan corridas mezcladas).
  `models/checkpoints/` está en `.gitignore`: no hay recuperación. `m2_parts/train.py`
  escribe por defecto en `car_parts/v1`; `m3_damage/detector/train.py` en `damage/v2` y `train_masks.py`
  en `car_parts/v2` (las dos ya existen o existirán) — **siempre pasar `--output` a una
  carpeta nueva**. `scripts/m3_train_damage_v2.ps1` se niega a arrancar si la carpeta ya tiene
  `best_model.pth`.
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
`pycocotools` es la métrica principal (`src/detection_common/coco_eval.py`). Para M4 y el
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
- `notebooks/datos_*_muestras.ipynb` (`datos_cardd`, `datos_car_parts_roboflow`,
  `datos_car_parts_segmentation`) — uno por dataset; comparten `show_samples()` /
  `load_coco_annotations()` duplicados (un bug en uno probablemente está en los tres). Usar
  `matplotlib.colormaps[name].resampled(n)`, no `cm.get_cmap`.
- `notebooks/demo_avances.ipynb` — demo de avances; importa `apply_clahe_rgb`,
  `resize_with_padding` y `specular_removal` solo para visualizar.
- `tests/m2_parts/evaluate_test_set.ipynb` — **solo para v1**: carga test con las 47 clases
  originales, cuenta predicciones contra GT (no calcula AP) y guarda JSON de M2 con
  `mascara: null`. Con un checkpoint de v2 compara ids de clases distintas sin error: no usarlo
  para v2.
- `tests/m2_parts/evaluate_test_set_v2.ipynb` — evaluación de un checkpoint de car_parts (v2
  por defecto; v1 con `M2_CHECKPOINT=...`) con **las clases del checkpoint** (aplica su
  `class_map` al GT y verifica que las categorías coinciden). Una sola pasada de inferencia:
  COCO AP de caja y máscara global y por clase (con instancias y "poco fiable" < 10),
  tiempo de inferencia (4b), figuras predicción/GT en dos secciones (7a máscaras, 7b cajas),
  y salida de M2 para M3 en
  `tests/m2_parts/outputs/<carpeta del checkpoint>/`: un JSON por imagen
  `{imagen, image_id, width, height, checkpoint, score_threshold, clases, partes: [{id,
  clase_parte, confianza, bbox [x1,y1,x2,y2], area_px, mascara: "masks/<img>_pNN.png"}]}` con
  umbral 0.5. Variables de entorno: `M2_CHECKPOINT`, `M2_SPLIT`, `M2_DEVICE`. Guardado con las
  salidas de la corrida sobre `car_parts/v2`. Sus números coinciden con `compare_versions`.
- `tests/m3_damage/detector/evaluate_damage_test_set.ipynb` — lo mismo para el
  detector de daños (`M3_CHECKPOINT`, default `damage/v2`; `M3_REFERENCE`, default `damage/v1`,
  vacío para no comparar; `M3_SPLIT`, `M3_DEVICE`): COCO AP de caja/máscara global, por tamaño
  y por clase; tiempo de inferencia (4b); curva de entrenamiento contra la referencia (7);
  comparación en las mismas imágenes con bootstrap pareado e IC 95 % (8, usa las funciones de
  `compare_versions`); figuras de máscaras (9a) y cajas (9b) con una imagen por clase; y salida
  cruda del detector en `tests/m3_damage/detector/outputs/<carpeta>/` (en `.gitignore`): un JSON por
  imagen `{imagen, image_id, width, height, checkpoint, score_threshold, danos: [{id, tipo,
  confianza, bbox, area_px, mascara}]}` + PNG por daño, sin `parte`/`solapamiento`/
  `pos_relativa` (el notebook evalúa solo el detector; esos campos los agrega
  `src/m3_damage/assignment/`). Reproduce los números de test de v1.
- `tests/m2_parts/manual_inference.py` y
  `tests/m3_damage/detector/manual_inference.py` — demos con GUI (selector de
  archivo, ventana ajustada a la pantalla), no se colectan. Partes: default `car_parts/v2`,
  umbral 0.7. Daños: default `damage/v2`, umbral 0.5, lista cada daño con su área de máscara.
  Ambos buscan la raíz del proyecto subiendo hasta `src/` (el de partes se rompió al moverlo
  cuando la calculaba por profundidad).
- `tests/m1_preprocessing/*_before_after.ipynb` — demos de M1.

Las rutas en notebooks se resuelven relativas a la raíz del proyecto, no hardcodeadas.

## `src/m1_preprocessing/` — M1
- `specular_removal.py` — Shen & Cai (2009), "maximum diffuse chromaticity", más un gating por
  brillo propio (no del paper) para no oscurecer superficies grises/negras (llantas, asfalto).
  Parámetros documentados en `ShenParams`. CLI:
  `python -m src.m1_preprocessing.specular_removal <input_dir> <output_dir> [-n N]`. Probado en
  `tests/m1_preprocessing/test_specular_removal.py`.
- `validation.py` — `validate_image()`: JPG/PNG decidido por contenido, no por extensión;
  resolución mínima 1920×1080 sin importar la orientación (lado largo ≥ 1920, corto ≥ 1080).
  `ImageValidationError.code` ∈ `no_encontrado`, `ilegible`, `formato`, `resolucion`; el
  mensaje está en español para el usuario.
- `pipeline.py` — `load_image()` (BGR derecho según EXIF), `preprocess_image()` (reflejos →
  CLAHE) y `preprocess_file()` (validación + carga + pasos + metadatos). `PreprocessConfig`:
  reflejos **activados** y CLAHE apagado por defecto. Conserva el tamaño salvo con `max_long_side` (ver `pair_size.py`).
- `torch_transform.py` — M1 como transform de dataset (variantes `raw`, `specular`, `clahe`,
  `both`); lo usa `--preprocess` del trainer (train **y** val) y `ap_experiment.py`.
- `ap_experiment.py` — COCO AP de un detector con cada variante de M1 contra imágenes crudas
  (bootstrap pareado, IC 95 %); con `--retrained` compara un modelo reentrenado con M1.
- `pair_size.py` — iguala la escala del par: la foto con lado largo mayor se reduce (INTER_AREA) al
  lado largo de la otra; la chica **nunca** se agranda. Lo aplica el orquestador
  (`PipelineSettings.equalize_size`, activado; casilla en Streamlit) vía
  `preprocess_file(max_long_side=…)`, después de EXIF y antes de los pasos ópticos; queda como paso
  `igualar_tamano_par` (escala, tamaño antes/después) en los metadatos. Las cajas de M3 quedan en
  píxeles de la imagen reducida: la evaluación escala el GT con el tamaño de la imagen etiquetada
  (`PairGroundTruth.size_a/size_b`, `metrics.scale_boxes`), así `bbox_real_*` está en píxeles de la
  imagen analizada. Medido el 2026-11-04 en `batch2` (p01, p04, p08, p14): igualar no cambia el
  veredicto de `view_check` (puntos 13→12, 10→9, 19→20, 0→0).
- `denoise.py` — `cv2.fastNlMeansDenoisingColored` (h=5) sobre una carpeta. Herramienta suelta,
  **fuera del pipeline por decisión del usuario**; quita ruido, no sube la resolución.
- `normalization.py` — `preprocess_pipeline()` = CLAHE (canal L en LAB) → letterbox a 512×512
  → ImageNet mean/std + CHW. **Contradice la spec de M1**: no conectarlo a los detectores
  (`pipeline.py` solo reutiliza su `apply_clahe_rgb`). Probado en `test_normalization.py`.
- Los modelos vigentes se entrenaron con imágenes **crudas**. Resultados en **test**
  (`<checkpoint>/m1_effect/` y `car_parts/v3_m1_specular_b/m1_retrain/`):
  - Solo inferencia, `damage/v2`: reflejos box AP −0.085 [−0.104, −0.064], CLAHE −0.025
    [−0.040, −0.010], ambos −0.106; **todas distinguibles** (empeoran), sobre todo dent,
    scratch y crack.
  - Solo inferencia, `car_parts/v2`: reflejos −0.032 [−0.055, −0.015] (empeora), CLAHE −0.002
    (no distinguible).
  - Reentrenado con reflejos (`car_parts/v3_m1_specular_b`) contra `car_parts/v2`: box AP
    −0.0035 [−0.032, +0.021], mask AP −0.011; **no distinguible**.
  - Por eso el orquestador corre M1 con reflejos **apagados** por defecto (solo validación,
    EXIF y metadatos). Reflejos cuestan ~140-280 ms por imagen en CPU.

## Nombres de carpetas (desde 2026-10-13, rama `feature/module-naming`)
Cada paquete de `src/` lleva el número de su módulo y `tests/` replica la misma estructura.
Lo compartido por varios módulos no lleva número. Los módulos nuevos siguen la regla:
`src/m4_comparison/`, y sin número `src/pipeline/` (orquestador) y `src/app/` (Streamlit).
**`models/checkpoints/car_parts/` y `models/checkpoints/damage/` NO se renombraron** (sus rutas
están en `run_info.json`, en los `train.log` y en los resultados ya citados), y el `name` de
cada `DetectorConfig` sigue siendo `car_parts` / `damage`.

```
src/
├── m1_preprocessing/   validation.py · pipeline.py · specular_removal.py · normalization.py
│                       torch_transform.py · ap_experiment.py
├── m2_parts/           config.py (CONFIG = v1 Faster R-CNN; MASK_CONFIG = v2 Mask R-CNN)
│                       class_map.py · train.py (v1) · train_masks.py (v2) · predict.py
│                       compare_versions.py
├── m3_damage/
│   ├── detector/       config.py · train.py · predict.py   (CarDD, Mask R-CNN)
│   └── assignment/     assign.py · analyze.py · output.py · render.py
├── detection_common/   config.py · coco_dataset.py · model.py · engine.py · trainer.py
│                       coco_eval.py · transforms.py · predict.py · visualize.py
│                       audit.py · watch.py
├── m4_comparison/      matching.py · parts.py · damages.py · report.py · render.py
│                       view_check.py
├── evaluation/         ground_truth.py · metrics.py · report.py · run.py · sweep.py · plots.py
├── pipeline/           orchestrator.py   (orquestador A/B)
└── app/                streamlit_app.py · view.py   (interfaz web)

tests/
├── m1_preprocessing/   test_validation · test_pipeline · test_specular_removal
│                       test_normalization · test_torch_transform · test_ap_experiment
│                       + notebooks antes/después
├── m2_parts/           test_class_map.py · test_compare_versions.py · notebooks de
│                       evaluación · manual_inference.py · outputs/ (ignorado)
├── m3_damage/
│   ├── detector/       notebook de evaluación · manual_inference.py · outputs/ (ignorado)
│   └── assignment/     test_assign.py · test_analyze.py · manual_inference.py · outputs/
├── detection_common/   test_coco_dataset · test_model · test_config · test_coco_eval
│                       test_transforms · test_audit · test_freeze_bn
├── m4_comparison/      test_match_parts.py · test_classify_damages.py · test_report.py
│                       test_render.py · test_view_check.py
├── pipeline/           test_orchestrator.py
└── app/                test_view.py
```

Nombres anteriores, por si aparecen en salidas guardadas de notebooks o en registros viejos:
`src/preprocessing` → `m1_preprocessing`; `src/detection/car_parts` → `m2_parts`;
`src/detection/damage` → `m3_damage/detector`; `src/damage_assignment` →
`m3_damage/assignment`; `src/detection/common` → `detection_common`;
`tests/detection/outputsM2` y `outputsM3` → `tests/m2_parts/outputs` y
`tests/m3_damage/detector/outputs`.

## `src/detection_common/`, `src/m2_parts/`, `src/m3_damage/detector/`
**Dos detectores, un pipeline compartido.** Lo dataset-agnóstico vive en `detection_common/`;
cada detector es un `DetectorConfig` más wrappers de CLI. No agregar otra copia del pipeline:
agregar una config.

CLIs (siempre con `./venv_cuda/Scripts/python.exe`):
`-m src.m2_parts.train`, `-m src.m2_parts.train_masks`,
`-m src.m3_damage.detector.train`, `-m src.m2_parts.predict`,
`-m src.m3_damage.detector.predict`.
Runbook completo: `docs/m2_m3_reentrenar_modelos.txt`.

### `detection_common/`
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
    config: `segm` en `car_parts` v2 (`MASK_CONFIG`) y en `damage` (desde 2026-09-30),
    `bbox` en `car_parts` v1.
    `validate_args()` rechaza `segm` sin máscaras o con `--metric simple`.
  - `--freeze-bn` (default off) — pone las `BatchNorm2d` en `eval()` tras cada
    `model.train()` (`model.freeze_batchnorm_stats`, llamado en `engine.train_one_epoch`).
    Se eligió `eval()` y no `FrozenBatchNorm2d` porque este último cambia las claves del
    `state_dict` y rompería `load_checkpoint`. Queda en `run_info.json` y en el checkpoint.
    Probado en `tests/detection_common/test_freeze_bn.py`.
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
    bloque `coco` y, desde 2026-11-06, `per_class` (`{bbox|segm: {clase: AP, AP50, AP75, AR_100,
    instances}}`, de la misma evaluación con `coco_eval.category_stats`; `None` sin GT); el
    trainer imprime también el AP por clase de cada época.
  - `batches.jsonl`: una línea por batch de train (época, batch, paso global, loss y términos
    —`null` si no es finito—, LR, segundos desde el inicio).
- `watch.py` — `python -m src.detection_common.watch <output_dir> [--interval N] [--once]`:
  muestra época, batch, barras de avance, losses, ETA, mejor época e historial; avisa si
  `status.json` lleva >2 min sin actualizarse o si hay losses NaN/inf.
- `transforms.py` — augmentation de train (flip horizontal + jitter de color), activa por
  defecto; nunca en validación.
- `predict.py` / `visualize.py` — `load_checkpoint()` reconstruye la arquitectura desde el
  checkpoint; `run_prediction()` filtra por `--score-threshold` (default 0.5) y **solo dibuja**
  imágenes anotadas. No produce el JSON de M2 ni de M3.

### `m2_parts/` — M2
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
- `compare_versions.py` — `python -m src.m2_parts.compare_versions --v1 V1.pth
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

### `car_parts/v3_10c` y `car_parts/v3_mix` — partes en 10 clases (2026-11-06, sin entrenar)
Clases: bumper, door, fender, glass, grille, hood, light, mirror, trunk, wheel (ids 1..10 en
orden alfabético). Se fusionan izq./der., delantera/trasera y todos los cristales y luces; se
descartan partes chicas (manija, emblema, tapa de gasolina) y accesorios (techo, canastilla,
estribos, guardafango). **Dos modelos con la misma receta salvo los datos de train**
(decidido 2026-11-06): `v3_10c` (`train_masks_10c`, solo Roboflow train) y `v3_mix`
(`train_masks_mix`, Roboflow train 327 + HITL train 802, una etapa). Ambos eligen la época con
el val de los dos datasets (Roboflow 32 + HITL 98) por mask AP. Receta (2026-11-06): 20 épocas,
`--lr-step-size 14`, lr 0.005, batch 2, semilla 42; se lanzan con
`scripts/m2_train_parts_10c.ps1` (los dos seguidos, monitor en otra ventana). Runbook y
comandos: sección 8 de `docs/m2_m3_reentrenar_modelos.txt`.
- **light = solo faros y calaveras** (decidido 2026-11-06, porque HITL no etiqueta faros de
  niebla ni direccionales): `left_fog_light`, `right_fog_lights`, `fog_lights` e
  `indicator_light` → `None` en todos los mapas (y por tanto `fog_light`/`indicator_light` de v2
  en `MAP_V2_TO_10`). `back_light` → `None`: la muestra muestra un reflejante de la defensa o el
  medallón, no una calavera.
- `mix_dataset.py`: `python -m src.m2_parts.mix_dataset` escribe `data/raw/parts_mix_coco/`
  (`train`, `train_roboflow`, `val`, `test`), categorías `roboflow/<nombre>` y `hitl/<nombre>`
  con `CLASS_MAP_MIX` (en `class_map_hitl.py`), `file_name` relativo a `data/raw`, orden de
  imágenes barajado con semilla 0 (para que `--max-train-images` tome ambas fuentes).
  Configs: `MASK_10_CONFIG` (train `train_roboflow`), `MASK_MIX_CONFIG` (train `train`);
  para evaluar, `ROBOFLOW_10_CONFIG`, `HITL_10_CONFIG` y `MIX_10_DATA`.
- **Regla de adopción** (`decide_mix.py`, sobre los reportes de `compare_10c` en test con
  referencia `v3_10c`): se adopta `v3_mix` si en mask AP el IC 95 % de la diferencia en el test
  de Roboflow no queda por debajo de cero (solo no inferioridad). El test de HITL (IC por encima
  de cero) solo cuenta con `--hitl-aligned`, cuando el criterio esté alineado; mientras, se
  reporta aparte. La caja se reporta sin decidir. **No entrenar hasta que el usuario confirme.**
- `class_map_10.py`: `CLASS_MAP_ROBOFLOW` (47 originales; decisiones aplicadas el 2026-11-06 con
  la muestra `car_parts/_muestra_clases_10c`: `windshield` = limpiaparabrisas → `None`,
  `front_mirror` → `None`, luces como arriba), `CLASS_MAP_CARPARTS_SEG` (23, solo prueba
  externa) y `MAP_V2_TO_10` (29 clases de v2 → 10, derivado con `compose_maps`).
- **Rueda** (decidido 2026-11-06): wheel = rueda completa, **una instancia por rueda**. En Roboflow
  `wheel` es el rin y `tire` la rueda completa (615 de 624 rines de train dentro de una llanta;
  el rin ocupa ~1/3 de la caja); para igualar a HITL el rin (`wheel`) → `None` y `tire` → wheel
  (también en `MAP_V2_TO_10`: el `wheel` de v2 se descarta). v2 en wheel: test de Roboflow caja
  0.714 → 0.712 (71 → 41 instancias), test de HITL caja 0.398 → 0.592 (sus rines ya no son falsos
  positivos).
- `final_classes` en `DetectorConfig` / `final_categories()` / `apply_class_map()` fuerza los ids
  aunque el dataset no tenga ejemplos de alguna clase.
- Instancias por clase final (train inflado ×3 / val / test) en Roboflow: bumper 321/33/18,
  door 603/62/31, fender 618/62/33, glass 1200/116/58, grille 354/31/16, hood 189/17/**9**,
  light 615/61/31, mirror 447/48/21, trunk 156/16/**7**, wheel (solo `tire`) 882/86/41 (hood y trunk
  poco fiables en el test de Roboflow; en el test conjunto con HITL, hood 84 y trunk 42).
- **Carparts-Seg no se usa para entrenar** (decidido 2026-11-06). `data/raw/carparts_seg`
  (de `ndjson_a_coco.py`, 640×640), 3,156/401/276 imágenes de **579 fotos base**. **Val y test
  son copias aumentadas de fotos de train**: con SIFT + RANSAC, 392 de 401 de val y 264 de 276 de
  test son la misma foto que una de train (rotada, volteada, recortada; mediana ~250 puntos);
  el hash de imagen no lo detectaba. **Sin fuga hacia el val/test de Roboflow** (las
  coincidencias más altas, ≤ 76 puntos, son falsos positivos por textura). Criterio distinto:
  la puerta incluye su ventana, los cristales laterales no se etiquetan, no hay faros de
  niebla, no hay fender ni grille, y 35 de 115 ruedas del test vienen unidas en un solo
  polígono. Muestras y hojas en `car_parts/_muestra_carparts_seg_10c/`.
- `CARPARTS_SEG_10_CONFIG`: mismos ids, solo para la prueba externa.
- `compare_10c.py`: `--model NOMBRE=CKPT` repetible, `--reference` (default el primero),
  `--dataset roboflow|hitl|mix|carparts_seg` (`hitl` y `mix` advierten que el test de HITL puede
  ser optimista: SIFT no distingue el mismo coche por debajo de 15 puntos). AP/AP50 de caja y máscara con IC, AP por clase con
  instancias e IC, grupos, diferencias contra la referencia con bootstrap pareado por imagen y
  veredicto (mejora/empeora solo si el IC excluye cero), y v2 en sus 29 clases como dato
  histórico. v2 contra sí mismo (`car_parts/v2/compare_10c/`, con el mapa de luces anterior):
  diferencia 0 exacta; en 29 clases reproduce 0.6041 / 0.5779. Con light = faros y calaveras:
  `car_parts/v2/compare_10c_b/`: test de Roboflow caja 0.680 [0.618, 0.764], máscara 0.666
  [0.598, 0.751]; HITL caja 0.262, máscara 0.272 (door 0.04/0.06: el GT de HITL incluye la
  ventana); ambos 0.335 / 0.341.
  - **Prueba externa** (`--dataset carparts_seg`): solo light, bumper, hood, wheel, mirror, trunk
    (no door ni glass); advierte que el IC es optimista y que no elige modelo. v2 mapeado (268
    imágenes): caja 0.217 [0.193, 0.255], máscara 0.223 [0.193, 0.265]; por clase (caja) bumper
    0.244, hood 0.323, light 0.168, mirror 0.302, trunk 0.217 [0.111, 0.367] (26), wheel 0.050
    (no interpretable: GT con ruedas unidas y rin/llanta).
- `trainer --init-from CKPT [--reset-predictors]`: hecho y probado (`test_init_from.py`), sin uso
  en el plan actual.
- Medido (RTX 3050): Roboflow ~0.28 s por imagen de train (60 imágenes, pico 2.2 GB); mezcla
  (`car_parts/smoke_mix_tiempo`, 160 imágenes, 68 % HITL) 0.36 s por imagen, val de 130 imágenes
  ~79 s, pico 2.84 GB. Estimado con 20 épocas: `v3_10c` ~1 h, `v3_mix` ~2 h 40 min (~8 min/época).

### Dataset HITL "Car parts and car damages" (2026-11-06, se usa en `v3_mix`)
Humans in the Loop, CC0 1.0 según su página (descargado el 2026-11-06 de una re-subida en Roboflow
Universe cuya licencia no se revisó; URL pendiente de confirmar);
detalle completo en `docs/dataset_hitl_partes.txt`. **Las carpetas
vienen invertidas**: las 998 imágenes de partes están en `Car damages dataset/`. Formato
Supervisely (polígonos con huecos: 140, casi todos puertas con la ventana como hueco); imágenes
de 562-916 px de lado largo; coches siniestrados en vista 3/4. 441 de las 998 también están en
el conjunto de daños.
- `hitl_dataset.py`: conversor a COCO (huecos → varias piezas sin hueco, IoU ≥ 0.957 contra la
  máscara exacta), `assign_splits()` por grupo y CLI. `class_map_hitl.py`: 21 clases → las 10
  (ids iguales a Roboflow); descarta License-plate, Roof y Rocker-panel.
- Criterio: en las etiquetas originales **la puerta incluye su ventana y el espejo** (hallado el
  2026-11-06 midiendo solapes: 1069 de 1244 puertas de train contienen una ventana, en Roboflow 12
  de 585). **Opción A aplicada** en la conversión (`fix_door`): se restan ventanas y espejos con
  >50 % dentro de la puerta, fragmentos <5 % fuera, puerta descartada si conserva <40 %.
  **Marco de la ventana quitado** (decidido 2026-11-06, `cut_frame`): se borra la puerta por
  encima de la línea de cintura = recta ajustada al borde inferior de las ventanas restadas
  (solo columnas con borde en o bajo su mediana, para que los lados inclinados no dejen un pilar),
  limitada al rango de esos bordes y aplicada a todas las columnas; el espejo no define el borde.
  Resultado: 1384 de 1638 puertas modificadas, 9 descartadas (recortes que eran solo marco o
  primeros planos con la ventana casi entera), área conservada mediana 0.63 (marco quitado
  mediana 7.5 % del área original). Muestra: `car_parts/_muestra_hitl_10c/door_sin_marco*.jpg`.
  v2 en door del test de HITL (antes → opción A → sin marco): caja 0.037 → 0.038 → **0.296**,
  máscara 0.059 → 0.292 → **0.397** (Roboflow 0.679 / 0.719). Sigue debajo de Roboflow, pero ya
  al nivel del resto de HITL (v2 en todo HITL: caja 0.307, máscara 0.327): lo que queda es cambio
  de dominio, no criterio. Matriz de solapes recalculada
  (`car_parts/_muestra_solapes_10c/matriz_solapes_b.md`): **ningún par con diferencia > 20
  puntos** (máx. 4). `class_map_hitl.PENDING_DECISIONS` vacío. Ventanas
  laterales etiquetadas aparte;
  Quarter-panel = `back_fender`; wheel = rueda completa; light = Headlight + Tail-light (no
  etiqueta faros de niebla ni direccionales; por eso se quitaron de Roboflow).
- Duplicados (SIFT): sin copias internas ni contra Roboflow/Carparts-Seg (máx. 19 puntos); 17
  pares de mismo vehículo desde otro ángulo → 982 grupos (probablemente hay más no detectados).
- Particiones (`data/raw/hitl_parts_coco/`): 80/10/10 por grupo, semilla 0 → 802/98/98
  imágenes; ninguna clase con menos de 10 instancias en val o test.

### `m3_damage/detector/` — M3 (solo el detector)
Dataset #1 (CarDD), 7 clases. Mask R-CNN porque un rayón es largo, delgado y diagonal: su caja
es casi todo fondo. Desde 2026-09-30 la config selecciona `best_model.pth` por **mask AP**
(`default_select_by="segm"`, porque M3 usa las máscaras de daño) y su salida por defecto es
`damage/v2` (antes `damage/v1`, que un `damage.train` sin `--output` sobrescribía).

**Lanzar damage/v2**: `powershell -ExecutionPolicy Bypass -File scripts/m3_train_damage_v2.ps1`.
Corre `damage.train --epochs 12 --lr-step-size 8 --select-by segm --output
models/checkpoints/damage/v2` (augmentation por defecto) y abre en otra ventana el mismo
monitor que car_parts (`common.watch`). Parámetros: `-Output`, `-Epochs`, `-LrStepSize`,
`-NoWatch`; cualquier flag del trainer puesto al final pasa tal cual (p. ej.
`--max-train-images 40`). Se niega si `-Output` ya tiene `best_model.pth` y avisa si el árbol
tiene cambios sin commitear. Probado el 2026-09-30 con 2 épocas sobre 40/20 imágenes
(`damage/smoke_test_v2`): completa, selecciona por mask AP, pico de GPU 3.10 GB.

Auditoría de CarDD (2026-09-30): sin archivos faltantes, tamaños coherentes con el JSON, un
polígono válido por anotación, sin máscaras vacías ni cajas inválidas, sin imágenes vacías.
Sin duplicados exactos entre splits; 2 casi-duplicados (misma foto con otra edición):
`test/001843` ≈ `train/000457` y `val/000790` ≈ `train/000473` — fuga despreciable, anotada.
`crack` es 39 % objetos pequeños (<32² px), lo que explica su AP bajo en v1.

### Pendiente
- **M4**: confirmar el umbral de daños (0.75 en `batch1`) con un lote de prueba y calibrar las
  distancias con pares que tengan daños preexistentes (ver "Evaluación con pares A/B").
- **M2**: el umbral de partes es 0.7 en la asignación (`DEFAULT_PART_THRESHOLD`) y 0.5 en los
  demos, sin justificación medida; falta resolver partes que se solapan (hoy el NMS de
  torchvision es por clase y no resuelve solapes entre clases distintas).

## `src/m4_comparison/` — M4
Lógica determinista sobre los reportes JSON de M3 (`partes` con `id`/`clase_parte`/`bbox`,
`danos` con `id`/`tipo`/`bbox`/`parte_id`/`pos_relativa`); sin torch ni modelos.
- `matching.py` — `match_points()`: método húngaro (`linear_sum_assignment`) con umbral de
  distancia aplicado **antes** de resolver (los pares inválidos cuestan más que cualquier suma
  válida), así maximiza el número de pares válidos y luego minimiza la distancia. Filtrar
  después del húngaro sin umbral puede perder pares (caso en las pruebas).
- `parts.py` — `match_parts()`: empareja partes **de la misma clase** por el centro de su caja
  normalizado al tamaño de cada imagen (A y B pueden tener otra resolución).
  `DEFAULT_PART_MAX_DISTANCE = 0.25`. Devuelve `PartMatching(pares, solo_a, solo_b)`.
- `damages.py` — `classify_damages()` / `compare_reports()`; daños del mismo tipo en dos
  etapas: (1) `"parte"`: dentro de cada par de partes, por `pos_relativa`; (2) `"imagen"`: lo
  que queda (sin parte, parte vista en una sola foto, daño en una unión asignado a partes
  distintas) por centro de caja normalizado a la imagen. `DEFAULT_DAMAGE_MAX_DISTANCE = 0.15`
  en ambas etapas. Estados `nuevo`, `preexistente` (con `distancia` y `metodo`) y
  `no_visible`; cada daño aparece exactamente una vez (primero los de B en su orden, luego los
  de A sin pareja).
  - **Histéresis** (2026-11-05): un daño sin pareja se busca otra vez entre las detecciones de la otra
    foto con confianza entre el umbral bajo (`DEFAULT_HYSTERESIS_THRESHOLD = 0.3`) y el umbral de daños
    ("apoyo"), con las mismas dos etapas. Si la encuentra es `preexistente` con `histeresis: true` (no
    nuevo ni no visible). El apoyo nunca se reporta como daño propio; el reporte de M4 lista en
    `histeresis` las detecciones de apoyo usadas, y la evaluación las cuenta como detecciones. M3:
    `split_support()` y `restrict_damages()` en `assignment/analyze.py`. Activada por defecto
    (`PipelineSettings.hysteresis_threshold`, casilla y deslizador en Streamlit, `--hysteresis` /
    `--no-hysteresis` en los scripts). Medido con barrido forzado e IoU 0.3: `batch1` sin cambio en
    recall (0.75: P 0.875 / R 0.538, igual; 0.6: FP 4 → 3), `batch2` (sin daños nuevos) nuevos falsos
    12 → 8 en 0.75 y p14 deja de marcar su golpe preexistente como nuevo (preexistentes bien
    clasificados 2 → 3 de 6). Los 8 falsos restantes son daños que M3 ve solo en B (rayones por
    reflejos, sobre todo), que la histéresis no puede corregir. Umbral bajo **sin calibrar**.
- `report.py` — `build_report()` arma el `ReporteComparacion` de la sección 4.9.2 (`imagen_a`,
  `imagen_b`, `umbral_distancia`, `resumen`, `danos`, `mapa_cambios`), más lo que piden RF09 y
  RNF05 y el diseño no traía: `umbral_distancia_partes`, `modelos` y, por daño, `confianza`,
  `bbox`, `bbox_en_b`, `pos_relativa`, `metodo` y `parte_id`. `requiere_revision` es verdadero
  para no visibles y daños sin parte. Un no visible no tiene píxeles en B: `bbox_en_b` es una
  **estimación** (por la parte emparejada o, si no hay, escalando a B). `save_report()` escribe
  `reporte_par.json` y `mapa_cambios.png`.
- `render.py` — `draw_change_map()`: nuevo con relleno fuerte y caja gruesa, preexistente con
  relleno tenue y caja delgada, no visible solo con caja punteada; resumen arriba a la izquierda.
- `view_check.py` — `check_same_view()`: SIFT + homografía con RANSAC **solo sobre el coche**
  (unión de máscaras de M2, porque en un patio el fondo es común). Exige `puntos` ≥ 40 y
  `solape` (IoU del coche de A movido sobre el de B) ≥ 0.5. Si falla, `build_report()` no
  clasifica nada y no hay mapa (`mapa_cambios: null`); con `forzada` compara, lo registra en
  `comparacion_forzada` y manda todos los daños a revisión. Comparar nombres de partes **no**
  sirve: frente de Duster contra frente de Ibiza da 0.80. Medido el 2026-10-30 con 12 fotos
  propias: 12/12 de misma vista **simulada** aceptadas y 66/66 de vistas o coches distintos
  rechazadas; los umbrales salieron de esos datos y **están sin calibrar**. Un par real tendrá
  menos puntos que uno simulado. No demuestra que sea el mismo vehículo: dos unidades idénticas
  podrían pasar.
- Sin pares A/B no se pueden calibrar los umbrales. Probado con modelos reales el 2026-10-30:
  la misma foto como A y B da todo preexistente. Las fotos `rigth` a `rigth4` de la Duster
  **no** forman pares válidos (son encuadres distintos).

## `scripts/analyze_folder.py`
M2+M3 sobre todas las fotos de una carpeta, sin GUI: por imagen el JSON de M3, imágenes
anotadas (daños con parte, todas las partes, todos los daños) y una máscara por daño, más
`resumen.json`. Default: `data/raw/own-data/TESTS/duster-condicion media-multiples angulos`.
`docs/etiquetas_modelos.csv`: clases de ambos modelos con nombre en español y etiquetas
originales.

## `scripts/analyze_pair.py`
M1→M4 sobre un par, sin GUI: `analyze_pair.py A.jpg B.jpg -o salida/ [--no-validate] [--force]`. Escribe
los reportes de M3 (`A.json`, `B.json`), imágenes anotadas, máscaras, `reporte_par.json` y
`mapa_cambios.png`.

## Evaluación con pares A/B (`src/evaluation/`, 2026-11-03)
- **Lote**: `<lote>/images/pNN_<coche>-<vista>_A|B.jpg` y `<lote>/annotations/instances_images.json`
  (CVAT, export **COCO 1.0**, polígonos). Atributo opcional `dano_id` para ligar el mismo daño en
  A y B: solo en B → nuevo, en ambas → preexistente, solo en A → no visible; sin `dano_id`, cada
  anotación es un daño propio. `ground_truth.load_batch()` valida nombres, pares incompletos y
  `dano_id` repetidos.
- **`batch1`** (`data/raw/own-data/batch1/`): 11 pares (Duster p01-p05, Ibiza p06-p11), 13 daños,
  **todos nuevos**, agregados con IA en B y con fondo quitado. Respaldo del JSON original en
  `annotations/instances_images.orig.json`. p03 lo bloquea `view_check` (12 puntos, solape 0.09:
  B tiene otro recorte). p06: el polígono de `lamp_broken` cubre también la pintura agrietada y la
  detección solo el faro (IoU 0.28) → **pendiente reetiquetar** (faro ajustado + `crack` aparte).
- **Métricas** (`metrics.py`, solo daños nuevos): un daño real cuenta como detectado si una
  detección lo cubre con IoU ≥ 0.3 (principal; también 0.5), emparejamiento húngaro con umbral; el
  tipo **no** tiene que coincidir. P/R/F1 de nuevo: FP = detecciones falsas marcadas nuevo + daños
  reales no nuevos marcados nuevo. Por par: "¿B tiene un daño nuevo?" (accuracy; `batch1` no tiene
  negativos) y pares exactos. Un par bloqueado por `view_check` cuenta como `sin_clasificar` salvo
  `--force`.
- **`scripts/evaluate_pairs.py <lote> -o <salida> [--force] [--iou 0.3 0.5] [--damage-threshold …]`**
  → `evaluacion.json`/`.md` y `pares/<pNN>/` (reportes de M3, anotadas, `reporte_par.json`, mapa).
  `run.load_evaluation()` recalcula los resúmenes al abrir un archivo viejo.
- **`scripts/sweep_damage_threshold.py <lote> -o <salida> [--thresholds …]`**: los modelos corren
  una vez por par con el umbral más bajo; cada umbral filtra esas detecciones (`restrict_damages`)
  y repite M4 reutilizando el veredicto de vista. Mejor = máx. F1 de nuevos en el IoU principal
  (empate: menos FP, luego umbral mayor). Escribe `barrido_umbral.csv/.json/.md` y
  `figuras/` (`prf_vs_umbral`, `errores_vs_umbral`, `precision_recall`, `f1_por_iou`, PNG 200 dpi).
- **Resultados en `batch1`** (`data/processed/eval/`, IoU 0.3): umbral 0.5 → P 0.500, R 0.615,
  F1 0.552, 8 nuevos falsos, 3/11 pares exactos (con `--force`: 0.529/0.692/0.600). Barrido
  (`batch1_umbral/`): **0.75 → P 0.889, R 0.615, F1 0.727, 1 falso, 6/11 exactos**; de 0.5 a 0.75
  se quitan 7 falsos sin perder aciertos; 0.80 ya pierde 2 (0.75 está al borde de la meseta
  0.60-0.75). Con IoU 0.5 el mejor sería 0.30. **Umbral de calibración**: 13 daños del mismo lote;
  `DEFAULT_DAMAGE_THRESHOLD` sigue en 0.5 hasta confirmarlo con un lote de prueba.
- **Streamlit**: barra lateral "Modo" → "Evaluar un lote etiquetado" (`app/evaluation_page.py`,
  lógica sin Streamlit en `app/evaluation_view.py`): evalúa o abre una evaluación guardada,
  métricas, tabla por par y las imágenes con el GT en verde.

## `src/pipeline/` y `src/app/` — orquestador e interfaz
- `pipeline/orchestrator.py` — `PairPipeline.from_checkpoints()` carga `car_parts/v2` y
  `damage/v2` una sola vez; `analyze_pair(path_a, path_b, PipelineSettings)` valida **ambas**
  fotos antes de cualquier inferencia y corre M1 → M2+M3 sobre A y B y luego M4.
  `PipelineSettings`: pasos de M1 (reflejos apagados, ver M1), validación de resolución,
  igualar el tamaño del par (activado),
  umbrales de partes (0.7), daños (0.5), solapamiento (0.5), distancias de M4 (0.25 y 0.15) y
  histéresis de M4 (0.3) y verificación de vista (40 puntos, solape 0.5). `PairResult` trae `comparison` (reporte de M4)
  y `change_map` (BGR, `None` si el par quedó bloqueado). `compare(result_a, result_b, settings,
  force)` repite solo M4, sin inferencia: lo usa el botón "Comparar de todos modos".
  `PairResult.to_json()` = `{A: {m1, m3, segundos}, B: …, m4: {…}}`. Probado con modelos falsos
  en `tests/pipeline/`.
- `app/streamlit_app.py` — `./venv_cuda/Scripts/python.exe -m streamlit run
  src/app/streamlit_app.py`. Carga de A y B, ajustes en la barra lateral, una pestaña por
  imagen (vista daños+parte / todas las partes / todos los daños, tabla de daños, partes y
  metadatos de M1), pestaña de M4 primero (conteos por estado, mapa de cambios, tabla por daño y
  descarga del PNG) y descarga del JSON del par.
- `app/view.py` — tablas, nombres en español de las clases de CarDD y estados de M4, y reducción de la
  imagen para mostrarla (lado largo ≤ 1600); sin Streamlit ni torch, probado en `tests/app/`.
- Medido el 2026-10-16 (RTX 3050) con fotos propias de 1600×1200 (hay que desactivar la
  validación): ~4-5.5 s de M2+M3 por imagen, incluido el calentamiento de CUDA.
- Medido el 2026-10-30 (RTX 3050), par completo M1→M4 con los dos modelos cargados, foto propia
  **reescalada** a 4000×3000 (vale para memoria y tiempo, no para calidad de detección): mediana
  ~5 s por par (4.2-6.0 s, 4 corridas). **Pico de GPU 5.5 GB asignados y 6.5 GB reservados**: el
  margen frente a los 6 GB de RNF02 es mínimo, y las cifras de 4.07 y 2.72 GB de cada modelo por
  separado no lo muestran.

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
- `car_parts/v2/` — **modelo de partes vigente para M2/M3.** Mask R-CNN, 29 clases finales
  (`class_map`), 20 épocas, lr 0.005 sin step, batch 2, AMP, augmentation, semilla 42,
  `freeze_bn` off, selección por mask AP. Commit `895f096` (árbol limpio). 33 min (~99 s/época).
  - Mejor época 16 (val): box AP 0.6664, mask AP 0.6435. Val se estanca desde la época ~6-8
    (box AP 0.65-0.67) mientras el loss sigue bajando 0.53 → 0.27; sin step de LR.
  - **Test** (16 imgs): box AP 0.6041 [IC 95 % 0.561-0.685], AP50 0.8043, AP75 0.7021;
    mask AP 0.5779 [0.533-0.659], AP50 0.8047.
  - Contra v1 mapeado a las mismas 29 clases (`v2/compare/compare_test.md`): box AP +0.0284
    [+0.0075, +0.0473]; AP50 −0.0197 [−0.0535, +0.0140]. Mejora la precisión de localización
    (AP75), no la detección a IoU 0.5. Solo 16 imágenes de test.
  - Peores clases en test (box AP): `step` 0.100 (1 instancia), `fog_light` 0.269 (6),
    `indicator_light` 0.359 (5), `back_bumper` 0.396 (8), todas "poco fiables".
  - **Tiempo de inferencia** (RTX 3050, FP32, lote 1, medido en la sección 4b del notebook
    v2, estable en dos corridas): imagen de 640×640 → modelo ~178 ms, total ~184 ms. Foto
    del protocolo 4000×3000 → preparación ~36 ms, modelo ~265 ms, **salida ~250-370 ms**
    (copiar a la CPU las máscaras a resolución completa, 12 MP por parte; es la fase más
    variable, 90-1100 ms), total ~575-590 ms (mediana). Pico de GPU 4.07 GB con 12 MP.
    Una inspección de 4 vistas ≈ 2.4 s de M2; un par A/B ≈ 4.7 s.
- `car_parts/_muestra_clases/` — muestra visual (4 imágenes de train por clase dudosa, con
  máscaras y nombre) con la que se cerró el `class_map`. Una hoja de contacto por grupo.
- `car_parts/_compare_v1_vs_v1/` — salida de `compare_versions` con v1 contra sí mismo en test.
- `car_parts/v3_m1_specular/` — primer reentrenamiento con `--preprocess specular`, cortado en
  la época 16 (`status.json` quedó en `running`, sin estado final). No usar.
- `car_parts/v3_m1_specular_b/` — reentrenamiento con `--preprocess specular` completo (20
  épocas, mejor época 18 por mask AP, misma receta que v2). En test no se distingue de v2 (ver
  la sección de M1). `car_parts/smoke_test_m1_specular/` es la prueba de pipeline.
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
- `damage/v2/` — **modelo de daños vigente.** Mask R-CNN, 12 épocas, `--lr-step-size 8`, AMP,
  batch 2, augmentation, semilla 42, selección por mask AP. Commit `df5948c` (árbol limpio).
  4.6 h (~20-28 min/época).
  - Val: mejor época 10, box AP 0.5297, mask AP 0.5124 (v1: 0.5191 / 0.5020). Misma forma de
    curva que v1: meseta ~0.45 en épocas 4-8, salto con la bajada de LR, plano desde la 9. El
    loss de entrenamiento queda más alto que en v1 (0.296 contra 0.202 en la época 12): efecto
    esperado de la augmentation.
  - **Test** (374 imgs): box AP 0.5275, AP50 0.7297, AP75 0.5736; mask AP 0.5075, AP50 0.7066.
    APs/APm/APl caja 0.354 / 0.291 / 0.540.
  - **Contra v1 en test, bootstrap pareado (1000, IC 95 %)**: box AP −0.0056 [−0.028, +0.020];
    box AP50 +0.016 [−0.015, +0.048]; mask AP +0.0125 [−0.006, +0.033]; mask AP50 +0.024
    [−0.006, +0.054]. **Ninguna diferencia es distinguible del ruido**: la augmentation no
    mejoró el detector de forma demostrable.
  - Por clase (caja, v1 → v2): glass shatter 0.867 → 0.840, tire flat 0.837 → 0.798, lamp broken
    0.570 → 0.605, dent 0.333 → 0.352, scratch 0.321 → 0.326, crack 0.271 → 0.244 (máscara de
    crack 0.089 → 0.123). Las tres clases objetivo siguen siendo las peores (~0.31).
  - Inferencia (RTX 3050, FP32, lote 1): ~205 ms por imagen de CarDD; con 4000×3000 la mediana
    es ~810 ms pero muy variable (260-2980 ms; el pegado de hasta 100 máscaras a 12 MP ocurre
    dentro del modelo). Pico 2.72 GB.
- `damage/v2_corrupta/` — primer intento de damage/v2 (2026-10-01), interrumpido a mano en la
  validación de la época 1 tras 2 h 16 min, unas 5× más lento de lo esperado (~28 min/época).
  Sin error en el log ni checkpoints; la causa probable es la laptop en suspensión o con
  energía limitada. Para detectarlo a tiempo: el ETA del monitor en los primeros minutos debe
  rondar 5.5 h en total.

## Pruebas
`tests/detection_common/` (`test_coco_dataset.py`, `test_model.py`, `test_config.py`,
`test_coco_eval.py`, `test_transforms.py`, `test_audit.py`, `test_freeze_bn.py`),
`tests/m2_parts/` (`test_class_map.py`, `test_compare_versions.py`),
`tests/m3_damage/assignment/` (`test_assign.py`, `test_analyze.py`),
`tests/m1_preprocessing/` (`test_validation.py`, `test_pipeline.py`,
`test_specular_removal.py`, `test_normalization.py`, `test_torch_transform.py`,
`test_ap_experiment.py`), `tests/m4_comparison/` (`test_match_parts.py`,
`test_classify_damages.py`, `test_report.py`, `test_render.py`, `test_view_check.py`),
`tests/pipeline/test_orchestrator.py`, `tests/app/` (`test_view.py`, `test_evaluation_view.py`) y
`tests/evaluation/` (`test_ground_truth.py`, `test_metrics.py`, `test_evaluation_report.py`,
`test_run.py`, `test_sweep.py`), más `tests/m2_parts/test_class_map_10.py`,
`tests/m2_parts/test_compare_10c.py`, `tests/detection_common/test_final_classes.py` y
`test_init_from.py`, `tests/m2_parts/test_hitl_dataset.py`, `test_class_map_hitl.py`,
`test_mix_dataset.py` y `test_decide_mix.py`.
Resultado al 2026-11-06: 432 pasan.
Los `manual_inference.py` de `tests/m2_parts/`, `tests/m3_damage/detector/` y
`tests/m3_damage/assignment/` son demos con GUI, no se colectan.

## Plan de trabajo acordado (registrado, no ejecutado salvo donde se indica)
0. ~~Cerrar el `class_map`~~ — hecho (29 clases).
1. Entrenamientos: ~~`car_parts/v2`~~ y ~~`damage/v2`~~ (hechos, ver Checkpoints). La
   augmentation no mejoró `damage` de forma demostrable; lo que más limita a M3 con fotos del
   protocolo es la escala (ver auditoría de CarDD). Pendiente opcional: corrida A/B de
   BatchNorm en car_parts.
2. ~~M3: asignación a partes + JSON, con pruebas~~ — hecho.
3. ~~M4: emparejamiento, clasificación, reporte JSON y mapa de cambios PNG~~ — hecho
   (2026-10-30); falta calibrar y medir.
4. ~~M1 compatible con los detectores + experimento de AP con y sin preprocesamiento~~ —
   hecho; M1 no mejora el AP (ver la sección de M1).
5. Orquestador + Streamlit: flujo completo conectado; falta probar en navegador.
6. Evaluación end-to-end con pares A/B propios (P, R, F1, Accuracy): herramientas hechas y
   primera medición con `batch1` (2026-11-03); faltan lotes con preexistentes, sin cambios y de prueba.
7. Limpieza de `README`, `setup.py` y `requirements.txt`: hecha.

## Reglas de trabajo
- Preguntar antes de editar código; no hacer push, entrenamientos ni borrados sin
  confirmación explícita del usuario.
- Commits de una sola línea y **sin** trailers de coautoría/atribución.
- Todo lo común a los dos detectores va en `src/detection_common/`; M3 y M4 en paquetes propios.
- Todo módulo nuevo lleva pruebas con pytest.
- Mantener `docs/m2_m3_reentrenar_modelos.txt` al día con cada entrenamiento.
- Cada entrenamiento en una carpeta `--output` nueva (hallazgo (e)).
- `AGENTS.md` se versiona; `CLAUDE.md` es una copia idéntica que queda fuera del repo
  (`.gitignore`). Al cambiar uno, copiar al otro.
