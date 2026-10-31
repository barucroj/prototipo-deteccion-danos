"""Minimal web interface: upload a rental pair A/B and look at the result.

    ./venv_cuda/Scripts/python.exe -m streamlit run src/app/streamlit_app.py

Runs M1 -> M2 -> M3 -> M4 on the pair through :class:`src.pipeline.orchestrator.PairPipeline`.
The models are loaded once per server process (``st.cache_resource``).
"""

import json
import os
import sys
import tempfile

import cv2
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.app.view import comparison_rows, damage_rows, part_rows, to_display  # noqa: E402
from src.detection_common.visualize import draw_detections  # noqa: E402
from src.m1_preprocessing.pipeline import PreprocessConfig  # noqa: E402
from src.m1_preprocessing.validation import ImageValidationError  # noqa: E402
from src.m3_damage.assignment.render import draw_analysis  # noqa: E402
from src.m4_comparison.report import CHANGE_MAP_FILE, REPORT_FILE  # noqa: E402
from src.pipeline.orchestrator import (DEFAULT_DAMAGE_CHECKPOINT, DEFAULT_PARTS_CHECKPOINT,  # noqa: E402
                                       PairPipeline, PipelineSettings)

st.set_page_config(page_title="Detección de daños vehiculares", layout="wide")


@st.cache_resource(show_spinner="Cargando modelos (M2 partes, M3 daños)...")
def get_pipeline(parts_checkpoint: str, damage_checkpoint: str) -> PairPipeline:
    return PairPipeline.from_checkpoints(parts_checkpoint, damage_checkpoint)


def save_upload(upload, folder: str) -> str:
    """Write an uploaded file to disk: M1 validates and loads from a path.

    A and B go to separate folders, since both photos may share a name.
    """
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, os.path.basename(upload.name))
    with open(path, "wb") as handle:
        handle.write(upload.getbuffer())
    return path


def sidebar() -> PipelineSettings:
    st.sidebar.header("Configuración")
    st.sidebar.subheader("M1. Preprocesamiento")
    validate = st.sidebar.checkbox("Exigir resolución mínima 1920×1080", value=True)
    specular = st.sidebar.checkbox(
        "Reducir reflejos especulares", value=False,
        help="Los modelos vigentes (car_parts/v2, damage/v2) se entrenaron con imágenes crudas.")
    clahe = st.sidebar.checkbox("CLAHE", value=False)

    st.sidebar.subheader("M2 / M3. Detección")
    part_threshold = st.sidebar.slider("Umbral de confianza de partes", 0.05, 0.95, 0.70, 0.05)
    damage_threshold = st.sidebar.slider("Umbral de confianza de daños", 0.05, 0.95, 0.50, 0.05)
    min_overlap = st.sidebar.slider(
        "Solapamiento mínimo daño→parte", 0.05, 0.95, 0.50, 0.05,
        help="Fracción del daño que debe caer dentro de una parte para asignársela.")

    st.sidebar.subheader("M4. Comparación")
    distance_help = "Sin calibrar: aún no hay pares A/B con ground truth."
    part_max_distance = st.sidebar.slider("Distancia máxima entre partes", 0.05, 0.50, 0.25, 0.05,
                                          help=distance_help)
    damage_max_distance = st.sidebar.slider("Distancia máxima entre daños", 0.05, 0.50, 0.15, 0.05,
                                            help=distance_help)

    with st.sidebar.expander("Modelos"):
        st.caption(f"Partes: `{os.path.relpath(DEFAULT_PARTS_CHECKPOINT, PROJECT_ROOT)}`")
        st.caption(f"Daños: `{os.path.relpath(DEFAULT_DAMAGE_CHECKPOINT, PROJECT_ROOT)}`")

    return PipelineSettings(preprocess=PreprocessConfig(specular=specular, clahe=clahe), validate=validate,
                            part_threshold=part_threshold, damage_threshold=damage_threshold,
                            min_overlap=min_overlap, part_max_distance=part_max_distance,
                            damage_max_distance=damage_max_distance)


def show_image_result(result, title: str) -> None:
    report = result.analysis.report
    c1, c2, c3 = st.columns(3)
    c1.metric("Daños", len(report["danos"]))
    c2.metric("Partes detectadas", len(report["partes"]))
    c3.metric("Tiempo (s)", f"{result.seconds['m1'] + result.seconds['m2_m3']:.1f}",
              help=f"M1 {result.seconds['m1']} s · M2+M3 {result.seconds['m2_m3']} s")

    view = st.radio("Vista", ["Daños y su parte", "Todas las partes", "Todos los daños"],
                    horizontal=True, key=f"view_{title}")
    if view == "Daños y su parte":
        canvas = draw_analysis(result.image, result.analysis)
    else:
        detections = result.analysis.parts if view == "Todas las partes" else result.analysis.damages
        canvas = draw_detections(result.image.copy(), detections.boxes, list(range(len(detections))),
                                 detections.scores, dict(enumerate(detections.names)), masks=detections.masks)
    st.image(to_display(canvas), caption=f"{title} — {report['archivo']}", use_container_width=True)

    st.markdown("**Daños**")
    rows = damage_rows(report)
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.caption("Sin daños sobre el umbral.")
    with st.expander("Partes detectadas"):
        st.dataframe(part_rows(report), use_container_width=True, hide_index=True)
    with st.expander("Metadatos de M1"):
        st.json(result.metadata)


def show_comparison(pair) -> None:
    report = pair.comparison
    summary = report["resumen"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Nuevos", summary["nuevos"])
    c2.metric("Preexistentes", summary["preexistentes"])
    c3.metric("No visibles", summary["no_visibles"])

    st.image(to_display(pair.change_map), caption="Mapa de cambios sobre la imagen B",
             use_container_width=True)
    st.caption("Los no visibles se dibujan con contorno punteado en su posición estimada. "
               "Los umbrales de distancia aún no están calibrados con pares reales.")

    rows = comparison_rows(report)
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.caption("No se detectaron daños en ninguna de las dos imágenes.")
    st.download_button("Descargar mapa de cambios (PNG)", st.session_state["change_map_png"],
                       file_name=CHANGE_MAP_FILE, mime="image/png")


def main() -> None:
    st.title("Detección de daños vehiculares")
    st.caption("Compara la foto de entrega (A) con la de devolución (B) de la misma vista del vehículo.")
    settings = sidebar()

    col_a, col_b = st.columns(2)
    with col_a:
        upload_a = st.file_uploader("A — antes (entrega)", type=["jpg", "jpeg", "png"], key="upload_a")
        if upload_a:
            st.image(upload_a, use_container_width=True)
    with col_b:
        upload_b = st.file_uploader("B — después (devolución)", type=["jpg", "jpeg", "png"], key="upload_b")
        if upload_b:
            st.image(upload_b, use_container_width=True)

    if st.button("Analizar par", type="primary", disabled=not (upload_a and upload_b)):
        pipeline = get_pipeline(DEFAULT_PARTS_CHECKPOINT, DEFAULT_DAMAGE_CHECKPOINT)
        with tempfile.TemporaryDirectory() as folder:
            path_a = save_upload(upload_a, os.path.join(folder, "A"))
            path_b = save_upload(upload_b, os.path.join(folder, "B"))
            try:
                with st.spinner("Analizando A y B..."):
                    pair = pipeline.analyze_pair(path_a, path_b, settings,
                                                 name_a=upload_a.name, name_b=upload_b.name)
                st.session_state["pair"] = pair
                st.session_state["change_map_png"] = cv2.imencode(".png", pair.change_map)[1].tobytes()
            except ImageValidationError as error:
                st.session_state.pop("pair", None)
                st.error(str(error))

    pair = st.session_state.get("pair")
    if pair is None:
        return

    st.divider()
    tab_m4, tab_a, tab_b, tab_json = st.tabs(["Comparación (M4)", "Imagen A", "Imagen B", "JSON"])
    with tab_m4:
        show_comparison(pair)
    with tab_a:
        show_image_result(pair.a, "A")
    with tab_b:
        show_image_result(pair.b, "B")
    with tab_json:
        report = pair.to_json()
        st.download_button("Descargar reporte JSON", json.dumps(report, indent=2, ensure_ascii=False),
                           file_name=REPORT_FILE, mime="application/json")
        st.json(report, expanded=False)


main()
