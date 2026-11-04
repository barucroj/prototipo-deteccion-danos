"""Web page that evaluates a labeled batch of A/B pairs (``src.evaluation``).

Uses the sidebar settings of the app, so the thresholds can be tried on the
whole batch. Results are written to an output folder and can be reopened
without running the models again.
"""

import os

import cv2
import streamlit as st

from src.app.evaluation_view import draw_ground_truth, matrix_rows, pair_damage_rows, pair_rows
from src.app.view import to_display
from src.evaluation.ground_truth import GroundTruthError, load_batch
from src.evaluation.metrics import DEFAULT_MIN_IOU
from src.evaluation.run import PAIRS_SUBDIR, evaluate_batch, evaluation_config, load_evaluation
from src.m4_comparison.report import CHANGE_MAP_FILE

DEFAULT_BATCH = os.path.join("data", "raw", "own-data", "batch1")
EXTRA_IOU = 0.5


def _fmt(value) -> str:
    return "—" if value is None else f"{value:.3f}"


def _batch_inputs(project_root: str):
    col_batch, col_output = st.columns(2)
    batch = col_batch.text_input("Carpeta del lote (images/ + annotations/)", DEFAULT_BATCH,
                                 help="Ruta relativa a la raíz del proyecto o absoluta.")
    default_output = os.path.join("data", "processed", "eval", os.path.basename(os.path.normpath(batch)))
    output = col_output.text_input("Carpeta de salida", default_output)
    col_force, col_iou = st.columns(2)
    force = col_force.checkbox("Comparar también los pares que no pasan la verificación de vista", value=False)
    min_iou = col_iou.slider("IoU mínimo para contar un daño como detectado", 0.1, 0.9, DEFAULT_MIN_IOU, 0.05)

    def resolve(path):
        return path if os.path.isabs(path) else os.path.join(project_root, path)

    return resolve(batch), resolve(output), force, round(min_iou, 2)


def _run(batch_dir, output_dir, force, ious, settings, get_pipeline):
    try:
        pairs = load_batch(batch_dir)
    except (GroundTruthError, OSError) as error:
        st.error(f"No se pudo leer el lote: {error}")
        return None

    pipeline = get_pipeline()
    progress = st.progress(0.0, text=f"0/{len(pairs)} pares")

    def on_pair(index, truth, status):
        progress.progress((index + 1) / len(pairs), text=f"{index + 1}/{len(pairs)} pares · {truth.par}: {status}")

    config = evaluation_config(batch_dir, None, settings, pipeline.checkpoints, force)
    record = evaluate_batch(pipeline, pairs, settings, ious, output_dir, config, force, on_pair=on_pair)
    progress.empty()
    return record


def _show_summary(record, iou):
    summary = record["resumen"][iou]
    new, pairs, det = summary["nuevo"], summary["por_par"], summary["deteccion"]
    st.markdown("**Daños nuevos**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision", _fmt(new["precision"]), help=f"TP {new['tp']} · FP {new['fp']}")
    c2.metric("Recall", _fmt(new["recall"]), help=f"{new['tp']} de {new['soporte']} nuevos reales")
    c3.metric("F1", _fmt(new["f1"]))
    c4.metric("Nuevos falsos", new["fp"],
              help=f"Se cobrarían al cliente: {new['fp_falsos']} detecciones sin daño real y "
                   f"{new['fp_reales']} daños preexistentes o no visibles marcados como nuevos. "
                   f"{_fmt(new['falsos_por_par'])} por par.")

    st.markdown("**Por par: ¿B tiene un daño nuevo?**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", _fmt(pairs["accuracy"]),
              help=f"TP {pairs['tp']} · FP {pairs['fp']} · FN {pairs['fn']} · TN {pairs['tn']}")
    c2.metric("Pares exactos", f"{pairs['exactos']}/{summary['pares']}",
              help="Todos sus nuevos encontrados y ningún nuevo falso.")
    c3.metric("Pares con / sin nuevo", f"{pairs['positivos']} / {pairs['negativos']}")
    c4.metric("Pares no comparados", summary["pares_bloqueados"])
    if pairs["negativos"] == 0:
        st.caption("No hay pares sin daño nuevo: la accuracy por par todavía no mide los falsos positivos.")

    with st.expander("Informativo: detección, tipos y matriz de estados"):
        st.caption(f"Daños reales {det['reales']} · detectados {det['detectados']} "
                   f"(recall de detección {_fmt(det['recall'])}) · detecciones falsas {det['falsos']} · "
                   f"tipo correcto {summary['tipos']['coinciden']}/{summary['tipos']['detectados']}")
        st.dataframe(matrix_rows(summary), use_container_width=True, hide_index=True)


def _show_pair(record, iou, output_dir):
    pairs = record["pares"][iou]
    st.markdown("**Pares**")
    st.dataframe(pair_rows(pairs), use_container_width=True, hide_index=True)

    names = [f"{p['par']} · {p['vista']}" for p in pairs]
    choice = st.selectbox("Ver un par", names)
    pair = pairs[names.index(choice)]
    st.dataframe(pair_damage_rows(pair), use_container_width=True, hide_index=True)

    pair_dir = os.path.join(output_dir, PAIRS_SUBDIR, pair["par"])
    change_map = os.path.join(pair_dir, CHANGE_MAP_FILE)
    col_a, col_b = st.columns(2)
    image_a = cv2.imread(os.path.join(pair_dir, "A_anotada.jpg"))
    if image_a is not None:
        col_a.image(to_display(draw_ground_truth(image_a, pair, "A")),
                    caption="A — detecciones de M3 y etiquetas (verde)", use_container_width=True)
    image_b = cv2.imread(change_map if os.path.isfile(change_map) else os.path.join(pair_dir, "B_anotada.jpg"))
    if image_b is not None:
        caption = "B — mapa de cambios" if os.path.isfile(change_map) else "B — detecciones (par no comparado)"
        col_b.image(to_display(draw_ground_truth(image_b, pair, "B")), caption=f"{caption} y etiquetas (verde)",
                    use_container_width=True)
    if image_a is None and image_b is None:
        st.caption("Sin imágenes para este par (M1 lo rechazó o la carpeta de salida cambió).")


def show_batch_evaluation(settings, get_pipeline, project_root: str) -> None:
    """The page. ``get_pipeline()`` returns the cached :class:`PairPipeline`."""
    st.caption("Corre M1→M4 sobre cada par de un lote etiquetado en CVAT (COCO 1.0) y lo compara con las "
               "etiquetas. Usa los umbrales de la barra lateral.")
    batch_dir, output_dir, force, min_iou = _batch_inputs(project_root)
    ious = [min_iou] if min_iou == EXTRA_IOU else [min_iou, EXTRA_IOU]

    col_run, col_load = st.columns(2)
    if col_run.button("Evaluar lote", type="primary"):
        record = _run(batch_dir, output_dir, force, ious, settings, get_pipeline)
        if record is not None:
            st.session_state["evaluation"] = (output_dir, record)
    if col_load.button("Abrir evaluación guardada", disabled=not os.path.isfile(
            os.path.join(output_dir, "evaluacion.json"))):
        st.session_state["evaluation"] = (output_dir, load_evaluation(output_dir))

    stored = st.session_state.get("evaluation")
    if stored is None:
        return
    shown_dir, record = stored
    config = record["configuracion"]
    st.divider()
    st.caption(f"Lote `{config['lote']}` · salida `{shown_dir}` · umbrales: partes {config['umbral_partes']}, "
               f"daños {config['umbral_danos']}, distancia partes {config['distancia_partes']}, "
               f"distancia daños {config['distancia_danos']} · forzado: {'sí' if config['forzar'] else 'no'}")

    available = sorted(record["resumen"])
    main_iou = record["iou_principal"]
    labels = [f"≥ {v}" + (" (principal)" if v == main_iou else "") for v in available]
    choice = st.radio("IoU", labels, index=available.index(main_iou), horizontal=True)
    iou = available[labels.index(choice)]
    _show_summary(record, iou)
    _show_pair(record, iou, shown_dir)
