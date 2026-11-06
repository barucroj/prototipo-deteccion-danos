"""Ten-class map for the Humans in the Loop car-parts images (by original name).

Original titles as in the dataset's ``meta.json`` (21 classes). Final classes
and ids are the same as for Roboflow (:data:`src.m2_parts.class_map_10.FINAL_CLASSES`,
forced through ``final_classes``), so a model trained on either lines up.

Labeling criterion, from the visual sample
(``models/checkpoints/car_parts/_muestra_hitl_10c``, 2026-11-06):

- door **includes** its window and usually the mirror in the original
  labels, unlike Roboflow (found 2026-11-06 by measuring overlaps; the first
  visual check got it wrong). The conversion subtracts them
  (:func:`src.m2_parts.hitl_dataset.fix_door`, option A); the window frame
  above the window stays in the door (see :data:`PENDING_DECISIONS`);
- side windows are labeled (``Back-window`` includes the quarter glass);
- ``Fender`` is the front fender and ``Quarter-panel`` the rear panel around
  the rear wheel (Roboflow's ``back_fender``); ``Rocker-panel`` is the sill
  under the doors (discarded);
- wheel is the whole wheel (tire and rim), like the merged Roboflow
  ``wheel`` + ``tire``;
- light: only ``Headlight`` and ``Tail-light`` exist; fog lights and
  indicators are **not** labeled. Decided 2026-11-06: light means headlights
  and taillights in every map, so Roboflow's fog lights and indicators are
  discarded too (:mod:`src.m2_parts.class_map_10`).
"""

from __future__ import annotations

from src.m2_parts.class_map_10 import FINAL_CLASSES

CLASS_MAP_HITL = {
    # bumper
    "Front-bumper": "bumper", "Back-bumper": "bumper",
    # door
    "Front-door": "door", "Back-door": "door",
    # glass
    "Windshield": "glass", "Back-windshield": "glass",
    "Front-window": "glass", "Back-window": "glass",
    # light
    "Headlight": "light", "Tail-light": "light",
    # fender
    "Fender": "fender", "Quarter-panel": "fender",
    # wheel
    "Front-wheel": "wheel", "Back-wheel": "wheel",
    # one to one
    "Hood": "hood", "Trunk": "trunk", "Mirror": "mirror", "Grille": "grille",
    # discarded
    "License-plate": None, "Roof": None, "Rocker-panel": None,
}

#: Labeling differences with Roboflow still open. Settled on 2026-11-06: light
#: (headlights and taillights only), door (window and mirror subtracted, option A,
#: and the window frame cut at the belt line) and wheel (Roboflow's rim discarded,
#: one instance per wheel).
PENDING_DECISIONS = {}

__all__ = ["CLASS_MAP_HITL", "CLASS_MAP_MIX", "FINAL_CLASSES", "MIX_SEPARATOR", "PENDING_DECISIONS", "prefixed"]

#: Separator between the source and the original name in the mixed dataset
#: (``roboflow/back_bumper``, ``hitl/Back-bumper``): the two sources share no
#: category names, but a prefix keeps the map auditable by origin.
MIX_SEPARATOR = "/"


def prefixed(class_map, prefix: str):
    """``class_map`` with every original name as ``<prefix>/<name>``."""
    return {f"{prefix}{MIX_SEPARATOR}{name}": target for name, target in class_map.items()}


def _mix_map():
    from src.m2_parts.class_map_10 import CLASS_MAP_ROBOFLOW
    return {**prefixed(CLASS_MAP_ROBOFLOW, "roboflow"), **prefixed(CLASS_MAP_HITL, "hitl")}


#: Map of the mixed Roboflow + HITL dataset (:mod:`src.m2_parts.mix_dataset`).
CLASS_MAP_MIX = _mix_map()
