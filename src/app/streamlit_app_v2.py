"""Web interface, two pages: settings (with what each one changes) and analysis of a pair A/B.

    ./venv_cuda/Scripts/python.exe -m streamlit run src/app/streamlit_app_v2.py

Same pipeline as :mod:`src.app.streamlit_app` (M1 -> M2 -> M3 -> M4 through
:class:`src.pipeline.orchestrator.PairPipeline`, models loaded once per server
process). The settings live in ``st.session_state`` so they survive switching
pages.
"""

import json
import os
import sys
import tempfile

import cv2
import numpy as np
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.app.view import comparison_rows, damage_label, damage_rows, part_rows, to_display  # noqa: E402
from src.detection_common.visualize import draw_detections, put_label  # noqa: E402
from src.m1_preprocessing.pipeline import PreprocessConfig  # noqa: E402
from src.m1_preprocessing.validation import ImageValidationError  # noqa: E402
from src.m3_damage.assignment.analyze import DEFAULT_DAMAGE_THRESHOLD, DEFAULT_PART_THRESHOLD  # noqa: E402
from src.m3_damage.assignment.merge import DEFAULT_MERGE_CONTAINMENT, DEFAULT_MERGE_IOU, MergeSettings  # noqa: E402
from src.m4_comparison.damages import (DEFAULT_DAMAGE_MAX_DISTANCE, DEFAULT_HYSTERESIS_THRESHOLD,  # noqa: E402
                                       NO_VISIBLE, NUEVO, PREEXISTENTE)
from src.m4_comparison.parts import DEFAULT_PART_MAX_DISTANCE  # noqa: E402
from src.m4_comparison.render import STATE_COLORS, _dashed_box  # noqa: E402
from src.m4_comparison.report import CHANGE_MAP_FILE, REPORT_FILE  # noqa: E402
from src.m4_comparison.view_check import DEFAULT_MIN_VIEW_OVERLAP, DEFAULT_MIN_VIEW_POINTS  # noqa: E402
from src.pipeline.orchestrator import (DEFAULT_DAMAGE_CHECKPOINT, DEFAULT_PARTS_CHECKPOINT,  # noqa: E402
                                       PairPipeline, PipelineSettings)

#: Part classes of car_parts/v3_mix -> label shown to the user.
PART_LABELS = {"bumper": "Defensa", "door": "Puerta", "fender": "Salpicadera", "glass": "Cristal",
               "grille": "Parrilla", "hood": "Cofre", "light": "Faro / calavera", "mirror": "Espejo",
               "trunk": "Cajuela", "wheel": "Rueda"}
STATE_ORDER = [NUEVO, PREEXISTENTE, NO_VISIBLE]
STATE_NAMES = {NUEVO: "Nuevos", PREEXISTENTE: "Preexistentes", NO_VISIBLE: "No visibles"}
STATE_SINGULAR = {NUEVO: "Nuevo", PREEXISTENTE: "Preexistente", NO_VISIBLE: "No visible"}
FILL_ALPHA = {NUEVO: 0.5, PREEXISTENTE: 0.3, NO_VISIBLE: 0.0}

#: Every setting with its default; the keys are the session_state keys of the widgets.
DEFAULTS = {
    "cfg_validate": True,
    "cfg_equalize": True,
    "cfg_specular": False,
    "cfg_clahe": False,
    "cfg_part_threshold": float(DEFAULT_PART_THRESHOLD),
    "cfg_damage_threshold": float(DEFAULT_DAMAGE_THRESHOLD),
    "cfg_min_overlap": 0.50,
    "cfg_merge": True,
    "cfg_merge_iou": float(DEFAULT_MERGE_IOU),
    "cfg_merge_containment": float(DEFAULT_MERGE_CONTAINMENT),
    "cfg_preexisting_enabled": True,
    "cfg_preexisting_threshold": float(DEFAULT_HYSTERESIS_THRESHOLD),
    "cfg_part_distance": float(DEFAULT_PART_MAX_DISTANCE),
    "cfg_damage_distance": float(DEFAULT_DAMAGE_MAX_DISTANCE),
    "cfg_view_points": int(DEFAULT_MIN_VIEW_POINTS),
    "cfg_view_overlap": float(DEFAULT_MIN_VIEW_OVERLAP),
    "cfg_show_states": [NUEVO],
}

CSS = """
<style>
:root { --ink: #1f2328; --muted: #6b6f76; --line: #e6e3dc; --card: #ffffff; --accent: #6c1d45; }
.stApp { background: #faf9f6; color: var(--ink); }
[data-testid="stHeader"] { background: rgba(250, 249, 246, 0.92); border-bottom: 1px solid var(--line); }
#MainMenu, footer, [data-testid="stDecoration"] { visibility: hidden; }
.block-container { padding-top: 4.5rem; max-width: 1280px; }
h1, h2, h3 { color: var(--ink); letter-spacing: -0.01em; font-weight: 600; }
h1 { font-size: 1.9rem; }
.subtitle { color: var(--muted); font-size: 1rem; margin-top: -0.6rem; margin-bottom: 1.4rem; }
[data-testid="stVerticalBlockBorderWrapper"] { background: var(--card); border-color: var(--line) !important;
  border-radius: 14px; }
[data-testid="stMetric"] { background: var(--card); border: 1px solid var(--line); border-radius: 12px;
  padding: 0.9rem 1rem; }
[data-testid="stMetricLabel"] p { color: var(--muted); font-size: 0.85rem; }
.verdict { border-radius: 14px; padding: 1rem 1.2rem; margin: 0.4rem 0 1rem 0; border: 1px solid var(--line);
  background: var(--card); font-size: 1.05rem; }
.verdict b { font-size: 1.15rem; }
.verdict.new { border-left: 6px solid #c62828; }
.verdict.clean { border-left: 6px solid #2e7d32; }
.verdict.warn { border-left: 6px solid #b26a00; }
.effect { color: var(--muted); font-size: 0.88rem; line-height: 1.45; margin: -0.3rem 0 1rem 0; }
.effect b { color: var(--ink); font-weight: 600; }
.chips span { display: inline-block; border: 1px solid var(--line); background: var(--card); border-radius: 999px;
  padding: 0.15rem 0.7rem; margin: 0 0.35rem 0.35rem 0; font-size: 0.82rem; color: var(--muted); }
.stTabs [data-baseweb="tab-list"] { gap: 0.4rem; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { padding: 0.5rem 0.9rem; }
</style>
"""


# ---------------------------------------------------------------- settings state

def init_settings() -> None:
    """Defaults on the first run; afterwards re-assign each key so Streamlit keeps it when the page changes."""
    for key, value in DEFAULTS.items():
        st.session_state[key] = st.session_state.get(key, value)


def reset_settings() -> None:
    for key, value in DEFAULTS.items():
        st.session_state[key] = value


def current_settings() -> PipelineSettings:
    s = st.session_state
    return PipelineSettings(
        preprocess=PreprocessConfig(specular=s.cfg_specular, clahe=s.cfg_clahe),
        validate=s.cfg_validate, equalize_size=s.cfg_equalize,
        part_threshold=s.cfg_part_threshold, damage_threshold=s.cfg_damage_threshold,
        min_overlap=s.cfg_min_overlap,
        merge=MergeSettings(s.cfg_merge, s.cfg_merge_iou, s.cfg_merge_containment),
        part_max_distance=s.cfg_part_distance, damage_max_distance=s.cfg_damage_distance,
        hysteresis_threshold=s.cfg_preexisting_threshold if s.cfg_preexisting_enabled else None,
        min_view_points=int(s.cfg_view_points), min_view_overlap=s.cfg_view_overlap)


def changed_settings() -> list:
    return [key for key, value in DEFAULTS.items() if st.session_state[key] != value]


@st.cache_resource(show_spinner="Cargando los modelos (partes y daños)...")
def get_pipeline() -> PairPipeline:
    return PairPipeline.from_checkpoints(DEFAULT_PARTS_CHECKPOINT, DEFAULT_DAMAGE_CHECKPOINT)


# ---------------------------------------------------------------- drawing

def shown_damages(comparison: dict, states) -> list:
    """Damages of the M4 report in the chosen states, in display order (new first); numbered 1..N."""
    order = [s for s in STATE_ORDER if s in states]
    return sorted((d for d in comparison["danos"] if d["estado"] in order), key=lambda d: order.index(d["estado"]))


def draw_damages_on_b(image_b: np.ndarray, comparison: dict, result_b, states) -> np.ndarray:
    """Image B with the damages in ``states``: mask, box (dashed if not visible) and a number for the table."""
    canvas = image_b.copy()
    analyses = [result_b.analysis] + ([result_b.support] if result_b.support is not None else [])
    masks = {d["id"]: m for a in analyses for d, m in zip(a.report["danos"], a.damages.masks)}
    scale = max(0.5, max(canvas.shape[:2]) / 1600)
    line = max(3, int(round(scale * 4)))
    shown = shown_damages(comparison, states)
    for d in shown:
        mask = masks.get(d["id_b"]) if d["id_b"] else None
        if mask is not None:
            alpha = FILL_ALPHA[d["estado"]]
            canvas[mask] = ((1 - alpha) * canvas[mask] + alpha * np.array(STATE_COLORS[d["estado"]])).astype(np.uint8)
    for number, d in enumerate(shown, 1):
        color = STATE_COLORS[d["estado"]]
        x1, y1, x2, y2 = (int(v) for v in d["bbox_en_b"])
        if d["estado"] == NO_VISIBLE:
            _dashed_box(canvas, (x1, y1, x2, y2), color, line)
        else:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, line if d["estado"] == NUEVO else max(2, line // 2))
        put_label(canvas, f" {number} ", (x1, y1 - 8 * scale), scale * 1.8, color)
    return canvas


def draw_only(result, what: str) -> np.ndarray:
    detections = result.analysis.parts if what == "parts" else result.analysis.damages
    names = [PART_LABELS.get(n, n) if what == "parts" else n for n in detections.names]
    names = [n.encode("ascii", "ignore").decode() for n in names]  # OpenCV fonts are ASCII only
    return draw_detections(result.image.copy(), detections.boxes, list(range(len(detections))),
                           detections.scores, dict(enumerate(names)), masks=detections.masks)


def png(image_bgr: np.ndarray) -> bytes:
    return cv2.imencode(".png", image_bgr)[1].tobytes()


def part_name(name):
    return PART_LABELS.get(name, name) if name else "Sin parte asignada"


# ---------------------------------------------------------------- page: settings

def effect(text: str) -> None:
    st.markdown(f'<div class="effect">{text}</div>', unsafe_allow_html=True)


def settings_page() -> None:
    st.title("Ajustes del análisis")
    st.markdown('<div class="subtitle">Los valores predeterminados son los que se usaron en la evaluación del '
                'sistema. Cada ajuste explica qué pasa si se sube o se baja.</div>', unsafe_allow_html=True)

    top = st.columns([3, 1])
    changed = changed_settings()
    top[0].caption(f"{len(changed)} ajuste(s) distinto(s) al predeterminado." if changed
                   else "Todos los ajustes están en su valor predeterminado.")
    top[1].button("Restaurar predeterminados", on_click=reset_settings, width="stretch", disabled=not changed)

    with st.container(border=True):
        st.subheader("Qué mostrar sobre la imagen B")
        st.segmented_control("Daños dibujados en el resultado", STATE_ORDER, selection_mode="multi",
                             format_func=STATE_NAMES.get, key="cfg_show_states")
        effect("Solo cambia lo que se dibuja y se lista en la pestaña «Resultado»; el análisis es el mismo. "
               "<b>Nuevos</b> (rojo): aparecen en B y no estaban en A. <b>Preexistentes</b> (naranja): ya estaban en "
               "A. <b>No visibles</b> (azul, contorno punteado): estaban en A y no se ven en B; se dibujan en su "
               "posición estimada y conviene revisarlos.")

    with st.container(border=True):
        st.subheader("Detección de daños")
        st.slider("Confianza mínima de un daño", 0.05, 0.95, step=0.05, key="cfg_damage_threshold")
        effect("Un daño se reporta solo si el detector está al menos así de seguro. <b>Subirlo</b> quita "
               "detecciones falsas (reflejos, suciedad) pero deja fuera daños poco evidentes. <b>Bajarlo</b> "
               "encuentra más daños reales, a cambio de más falsos. Con 0.30 la evaluación dio cerca de un nuevo "
               "falso por par; con 0.75 (predeterminado), uno cada cuatro o cinco pares.")
        st.checkbox("Reconocer daños preexistentes con menor confianza en la foto A",
                    key="cfg_preexisting_enabled")
        st.slider("Confianza mínima de un daño preexistente", 0.05, 0.95, step=0.05,
                  key="cfg_preexisting_threshold", disabled=not st.session_state.cfg_preexisting_enabled)
        effect("Antes de marcar un daño de B como <b>nuevo</b>, el sistema lo busca en A (mismo tipo y misma "
               "posición) aceptando detecciones con esta confianza más baja. Si lo encuentra, es "
               "<b>preexistente</b>. <b>Bajarlo</b> reconoce más daños que ya estaban, con menos nuevos falsos, pero "
               "una detección muy débil en A puede ocultar un daño nuevo real. <b>Subirlo</b> hasta la confianza "
               "mínima de un daño equivale a apagarlo. En la evaluación, entre 0.20 y 0.35 dio el mejor resultado.")
        if (st.session_state.cfg_preexisting_enabled
                and st.session_state.cfg_preexisting_threshold >= st.session_state.cfg_damage_threshold):
            st.info("Con este valor igual o mayor que la confianza mínima de un daño, no tiene efecto.")

    with st.container(border=True):
        st.subheader("Partes del vehículo")
        st.slider("Confianza mínima de una parte", 0.05, 0.95, step=0.05, key="cfg_part_threshold")
        effect("<b>Subirlo</b> deja solo partes seguras: algunos daños quedan \"sin parte asignada\". <b>Bajarlo</b> "
               "agrega partes dudosas que pueden recibir un daño por error. Entre 0.65 y 0.90 el resultado es "
               "prácticamente igual; 0.70 es el predeterminado.")
        st.slider("Fracción del daño que debe caer dentro de la parte", 0.05, 0.95, step=0.05, key="cfg_min_overlap")
        effect("Un daño se asigna a la parte que más lo cubre si la cubre al menos en esta fracción. "
               "<b>Subirlo</b> deja sin parte a los daños que están en la unión de dos piezas; <b>bajarlo</b> "
               "asigna parte incluso cuando el daño apenas la toca.")
        st.checkbox("Unir detecciones del mismo tipo que se enciman", key="cfg_merge")
        cols = st.columns(2)
        cols[0].slider("Solape mínimo para unir", 0.05, 0.95, step=0.05, key="cfg_merge_iou",
                       disabled=not st.session_state.cfg_merge)
        cols[1].slider("Contención mínima para unir", 0.05, 1.0, step=0.05, key="cfg_merge_containment",
                       disabled=not st.session_state.cfg_merge)
        effect("Evita contar dos veces el mismo rayón cuando el detector lo parte en dos. Valores <b>más bajos</b> "
               "unen más (pueden juntar dos daños distintos); <b>más altos</b> unen menos.")

    with st.container(border=True):
        st.subheader("Comparación de A y B")
        cols = st.columns(2)
        cols[0].slider("Distancia máxima entre partes", 0.05, 0.50, step=0.05, key="cfg_part_distance")
        cols[1].slider("Distancia máxima entre daños", 0.05, 0.50, step=0.05, key="cfg_damage_distance")
        effect("Qué tan lejos (como fracción de la imagen o de la parte) puede estar un daño de B de uno de A para "
               "considerarlos el mismo. <b>Subirlo</b> tolera fotos tomadas un poco movidas, pero puede emparejar "
               "dos daños distintos y ocultar uno nuevo. <b>Bajarlo</b> exige fotos casi idénticas; si no lo son, "
               "aparecen nuevos falsos. Estos valores aún no se calibraron con pares reales.")
        cols = st.columns(2)
        cols[0].slider("Puntos coincidentes mínimos", 5, 100, step=5, key="cfg_view_points")
        cols[1].slider("Solape mínimo del vehículo", 0.10, 0.95, step=0.05, key="cfg_view_overlap")
        effect("Antes de comparar se verifica que A y B sean la misma vista del vehículo. <b>Bajarlos</b> bloquea "
               "menos pares: en fotos reales de coches lisos u oscuros hubo pares correctos con 7 a 30 puntos. "
               "Pero también aceptaría fotos de otra vista u otro coche, y la comparación no tendría sentido. "
               "Si un par queda bloqueado, en la página de análisis se puede comparar de todos modos.")

    with st.container(border=True):
        st.subheader("Preparación de las fotos")
        st.checkbox("Exigir resolución mínima (1920 × 1080)", key="cfg_validate")
        effect("Rechaza fotos pequeñas antes de analizarlas. Apagarlo permite probar con fotos de menor "
               "resolución, donde los daños pequeños se ven peor.")
        st.checkbox("Igualar el tamaño de A y B", key="cfg_equalize")
        effect("Reduce la foto más grande al tamaño de la otra, sin recortarla ni agrandar la pequeña. Así las "
               "dos se analizan a la misma escala. Se recomienda dejarlo encendido.")
        cols = st.columns(2)
        cols[0].checkbox("Reducir reflejos", key="cfg_specular")
        cols[1].checkbox("Mejorar contraste (CLAHE)", key="cfg_clahe")
        effect("Los modelos se entrenaron con fotos sin estos filtros. En las pruebas, <b>los dos empeoraron</b> "
               "la detección de daños, por eso están apagados.")


# ---------------------------------------------------------------- page: analysis

def save_upload(upload, folder: str) -> str:
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, os.path.basename(upload.name))
    with open(path, "wb") as handle:
        handle.write(upload.getbuffer())
    return path


def settings_chips(settings: PipelineSettings) -> None:
    chips = [f"Daños ≥ {settings.damage_threshold:.2f}",
             (f"Preexistentes ≥ {settings.hysteresis_threshold:.2f}" if settings.hysteresis_threshold is not None
              else "Preexistentes: sin confianza reducida"),
             f"Partes ≥ {settings.part_threshold:.2f}",
             f"Vista: {settings.min_view_points} puntos, solape {settings.min_view_overlap:.2f}"]
    st.markdown('<div class="chips">' + "".join(f"<span>{c}</span>" for c in chips) + "</div>",
                unsafe_allow_html=True)


def run_analysis(upload_a, upload_b, settings) -> None:
    pipeline = get_pipeline()
    with tempfile.TemporaryDirectory() as folder:
        path_a = save_upload(upload_a, os.path.join(folder, "A"))
        path_b = save_upload(upload_b, os.path.join(folder, "B"))
        try:
            with st.spinner("Analizando las dos fotos..."):
                pair = pipeline.analyze_pair(path_a, path_b, settings, name_a=upload_a.name, name_b=upload_b.name)
        except ImageValidationError as error:
            st.session_state.pop("pair", None)
            st.error(str(error))
            return
    st.session_state["pair"] = pair
    st.session_state["pair_settings"] = settings


def force_comparison() -> None:
    pair = st.session_state["pair"]
    pair.comparison, pair.change_map = get_pipeline().compare(pair.a, pair.b, st.session_state["pair_settings"],
                                                              force=True)


def verdict(pair) -> None:
    report = pair.comparison
    view = report["verificacion_vista"]
    if pair.change_map is None:
        st.markdown(f'<div class="verdict warn"><b>No se pudo comparar el par.</b><br>A y B no parecen la misma '
                    f'vista del vehículo ({view["puntos"]} puntos coincidentes, mínimo {view["umbral_puntos"]}; '
                    f'solape {view["solape"]:.2f}, mínimo {view["umbral_solape"]:.2f}). Revisa que las fotos '
                    f'correspondan o compara de todos modos.</div>', unsafe_allow_html=True)
        st.button("Comparar de todos modos", on_click=force_comparison, type="primary")
        return
    summary = report["resumen"]
    new = summary["nuevos"]
    others = (f'{summary["preexistentes"]} preexistente(s) y {summary["no_visibles"]} no visible(s) en B '
              f'(estaban en A y no se ven). Los daños se dibujan en la pestaña «Resultado».')
    if new:
        st.markdown(f'<div class="verdict new"><b>{new} daño(s) nuevo(s) en la devolución.</b><br>'
                    f'Además: {others}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="verdict clean"><b>Sin daños nuevos.</b><br>{others}</div>',
                    unsafe_allow_html=True)
    if report["comparacion_forzada"]:
        st.warning("Comparación forzada: la verificación indicó que A y B podrían no ser la misma vista. "
                   "Revisa todos los daños.")


def shown_rows(comparison: dict, states) -> list:
    return [{"#": i, "Estado": STATE_SINGULAR[d["estado"]], "Tipo": damage_label(d["tipo"]),
             "Parte": part_name(d["parte"]), "Confianza": round(d["confianza"], 2),
             "Revisar": "Sí" if d["requiere_revision"] else "No"}
            for i, d in enumerate(shown_damages(comparison, states), 1)]


def result_tab(pair) -> None:
    states = st.session_state.cfg_show_states or []
    cols = st.columns(2, gap="medium")
    cols[0].image(to_display(pair.a.image), caption=f"A · antes · {pair.a.analysis.report['archivo']}",
                  width="stretch")
    if pair.change_map is None:
        cols[1].image(to_display(pair.b.image), caption=f"B · después · {pair.b.analysis.report['archivo']}",
                      width="stretch")
        return
    new_image = draw_damages_on_b(pair.b.image, pair.comparison, pair.b, states)
    shown = ", ".join(STATE_NAMES[s].lower() for s in STATE_ORDER if s in states) or "ningún estado"
    cols[1].image(to_display(new_image), caption=f"B · después · se muestran: {shown}", width="stretch")
    rows = shown_rows(pair.comparison, states)
    if rows:
        st.dataframe(rows, hide_index=True, width="stretch")
    elif states:
        st.caption("No hay daños en los estados elegidos.")
    else:
        st.caption("No hay estados elegidos: cámbialo en «Ajustes» → «Qué mostrar sobre la imagen B».")
    with st.expander("Mapa de cambios completo (nuevos, preexistentes y no visibles)"):
        st.image(to_display(pair.change_map), width="stretch")
        st.caption("Rojo: nuevo · naranja tenue: preexistente · contorno punteado: no visible (posición estimada).")
    d1, d2, _ = st.columns([1, 1, 2])
    d1.download_button("Descargar imagen B marcada (PNG)", png(new_image), file_name="imagen_b_marcada.png",
                       mime="image/png", width="stretch")
    d2.download_button("Descargar mapa de cambios (PNG)", png(pair.change_map), file_name=CHANGE_MAP_FILE,
                       mime="image/png", width="stretch")


def only_tab(pair, what: str) -> None:
    cols = st.columns(2, gap="medium")
    for col, result, name in ((cols[0], pair.a, "A · antes"), (cols[1], pair.b, "B · después")):
        report = result.analysis.report
        with col:
            st.image(to_display(draw_only(result, what)), caption=name, width="stretch")
            if what == "parts":
                rows = part_rows(report)
                for row in rows:
                    row["Parte"] = part_name(row["Parte"])
            else:
                rows = damage_rows(report)
                for row, dano in zip(rows, report["danos"]):
                    row["Parte"] = part_name(dano["parte"])
            if rows:
                st.dataframe(rows, hide_index=True, width="stretch")
            else:
                st.caption("Nada detectado con los ajustes actuales.")


def detail_tab(pair) -> None:
    rows = comparison_rows(pair.comparison)
    for row in rows:
        row["Parte"] = part_name(row["Parte"]) if row["Parte"] != "Sin parte asignada" else row["Parte"]
    if rows:
        st.dataframe(rows, hide_index=True, width="stretch")
    else:
        st.caption("No se detectaron daños en ninguna de las dos fotos.")
    view = pair.comparison["verificacion_vista"]
    st.caption(f"Verificación de vista: {view['puntos']} puntos coincidentes (mínimo {view['umbral_puntos']}), "
               f"solape {view['solape']:.2f} (mínimo {view['umbral_solape']:.2f}).")


def json_tab(pair) -> None:
    report = pair.to_json()
    st.download_button("Descargar reporte JSON", json.dumps(report, indent=2, ensure_ascii=False),
                       file_name=REPORT_FILE, mime="application/json")
    st.json(report, expanded=False)


def analysis_page() -> None:
    st.title("Análisis de un par de fotos")
    st.markdown('<div class="subtitle">Sube la foto de la entrega (A) y la de la devolución (B), tomadas desde la '
                'misma posición. El sistema marca los daños que aparecen en B y no estaban en A.</div>',
                unsafe_allow_html=True)

    settings = current_settings()
    with st.container(border=True):
        cols = st.columns(2, gap="medium")
        upload_a = cols[0].file_uploader("Foto A · antes (entrega)", type=["jpg", "jpeg", "png"], key="upload_a")
        upload_b = cols[1].file_uploader("Foto B · después (devolución)", type=["jpg", "jpeg", "png"],
                                         key="upload_b")
        bottom = st.columns([3, 1], vertical_alignment="center")
        with bottom[0]:
            settings_chips(settings)
        if bottom[1].button("Analizar", type="primary", width="stretch", disabled=not (upload_a and upload_b)):
            run_analysis(upload_a, upload_b, settings)

    pair = st.session_state.get("pair")
    if pair is None:
        return
    if st.session_state.get("pair_settings") != settings:
        st.info("Los ajustes cambiaron desde el último análisis. Pulsa «Analizar» para aplicarlos.")

    verdict(pair)
    tabs = st.tabs(["Resultado", "Solo daños", "Solo partes", "Detalle por daño", "JSON"])
    with tabs[0]:
        result_tab(pair)
    with tabs[1]:
        only_tab(pair, "damages")
    with tabs[2]:
        only_tab(pair, "parts")
    with tabs[3]:
        detail_tab(pair)
    with tabs[4]:
        json_tab(pair)


# ---------------------------------------------------------------- app

def main() -> None:
    st.set_page_config(page_title="Detección de daños vehiculares", page_icon=":material/directions_car:",
                       layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    init_settings()
    page = st.navigation([st.Page(analysis_page, title="Análisis", icon=":material/compare:", default=True),
                          st.Page(settings_page, title="Ajustes", icon=":material/tune:")], position="top")
    page.run()


if __name__ == "__main__":
    main()
