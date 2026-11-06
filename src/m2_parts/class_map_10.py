"""Ten-class part maps for ``car_parts/v3_10c`` (by original name, not by id).

Final classes: bumper, door, glass, light, fender, hood, trunk, mirror, wheel,
grille. Left/right, front/back and every kind of glass and light are merged:
the system only gets front, side and rear photos, and M3 only needs to know
which part a damage falls on. Small parts (handle, emblem, fuel cover) and
accessories (roof, roof rack, steps) are discarded (``None``).

Final ids are given by :func:`src.detection_common.coco_dataset.final_categories`
(final names sorted alphabetically -> 1..10, 0 = background), as for the
29-class map, so they do **not** follow the order listed above.

:data:`CLASS_MAP_ROBOFLOW` covers the 47 Roboflow Car-Parts classes. Closed on
2026-11-06 after the visual sample (``models/checkpoints/car_parts/_muestra_clases_10c``):
``windshield`` is discarded (in Roboflow it labels the wipers; the glass is
``front_glass``) and ``front_mirror`` is discarded.

**light = headlights and taillights only** (decided 2026-11-06, to mix with
the Humans in the Loop images, which label neither fog lights nor
indicators): every fog light (``left_fog_light``, ``right_fog_lights``,
``fog_lights``) and ``indicator_light`` go to ``None``, here and therefore in
:data:`MAP_V2_TO_10`. ``back_light`` is also ``None``: the visual sample shows
a bumper reflector or the rear window, not a taillight (``car_parts/v2`` had
discarded it for the same reason).

**wheel = the whole wheel, one instance each** (decided 2026-11-06): in
Roboflow ``wheel`` is the rim, nested inside ``tire`` (the whole wheel; 615 of
624 rims of train lie inside a tire), so each wheel was two nested instances
of wheel while HITL has one. The rim goes to ``None`` and ``tire`` is wheel.

:data:`MAP_V2_TO_10` takes the 29 final classes of ``car_parts/v2`` to these
ten; it is derived from the two maps, so it cannot disagree with them.

:data:`CLASS_MAP_CARPARTS_SEG` covers the 23 classes of Ultralytics
Carparts-Seg (``data/raw/carparts-seg.ndjson``), only for the external test
(``compare_10c --dataset carparts_seg``); Carparts-Seg is not used for training.
It labels neither fender nor grille, and of glass only windshield and rear
window.
"""

from __future__ import annotations

FINAL_CLASSES = ("bumper", "door", "glass", "light", "fender", "hood", "trunk", "mirror", "wheel", "grille")

CLASS_MAP_ROBOFLOW = {
    # bumper
    "bumper": "bumper", "back_bumper": "bumper",
    # door
    "left_front_door": "door", "right_front_door": "door",
    "left_back_door": "door", "right_back_door": "door",
    # glass
    "front_glass": "glass", "back_glass": "glass",
    "left_front_glass": "glass", "right_front_glass": "glass",
    "left_back_glass": "glass", "right_back_glass": "glass",
    "left_quater_glass": "glass", "right_quarter_glass": "glass",  # "quater" is the dataset's typo
    # light
    "left_front_light": "light", "right_front_lights": "light",
    "left_back_lights": "light", "right_back_light": "light",
    # not labeled in HITL: fog lights and indicators (2026-11-06)
    "left_fog_light": None, "right_fog_lights": None, "fog_lights": None,
    "indicator_light": None,
    # fender
    "front_fender": "fender", "back_fender": "fender",
    # one to one
    "hood": "hood", "trunk": "trunk", "side_mirror": "mirror",
    "tire": "wheel", "radiator": "grille",
    "wheel": None,  # the rim, nested inside its tire: one instance per wheel, as in HITL (2026-11-06)
    # decided with the visual sample (2026-11-06)
    "windshield": None,  # the wipers, not the glass
    "back_light": None,  # bumper reflector or rear window, not a taillight
    "front_mirror": None,
    "mudguard": None,
    # discarded
    "air_intake": None, "left_side_door": None, "left_windowark": None,
    "right_windowark": None, "right_glass": None, "shield": None,
    "roof": None, "roof_trunk": None, "emblem": None, "handle": None,
    "fuel_cover": None, "side_steps": None, "step": None,
}

#: Entries waiting for a decision on the visual sample (none since 2026-11-06).
PENDING_VISUAL_CHECK = ()

CLASS_MAP_CARPARTS_SEG = {
    "front_bumper": "bumper", "back_bumper": "bumper",
    "front_door": "door", "back_door": "door", "front_left_door": "door",
    "front_right_door": "door", "back_left_door": "door", "back_right_door": "door",
    "front_glass": "glass", "back_glass": "glass",
    "front_light": "light", "back_light": "light", "front_left_light": "light",
    "front_right_light": "light", "back_left_light": "light", "back_right_light": "light",
    "hood": "hood", "trunk": "trunk", "tailgate": "trunk",
    "left_mirror": "mirror", "right_mirror": "mirror", "wheel": "wheel",
    "object": None,
}


def compose_maps(first, second):
    """``{final name of first: final name of second}`` through their shared original names.

    Raises if one class of ``first`` would split into two classes of
    ``second``: then the composition is not a map.
    """
    out = {}
    for original, middle in first.items():
        if middle is None:
            continue
        target = second[original]
        if middle in out and out[middle] != target:
            raise ValueError(f"{middle!r} maps to both {out[middle]!r} and {target!r}")
        out[middle] = target
    return out


def _v2_to_10():
    from src.m2_parts.class_map import CLASS_MAP
    return compose_maps(CLASS_MAP, CLASS_MAP_ROBOFLOW)


#: The 29 final classes of ``car_parts/v2`` -> the ten classes (or ``None``).
MAP_V2_TO_10 = _v2_to_10()
