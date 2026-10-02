"""Mapping from the 47 Roboflow car-parts classes to the final M2 classes.

A name -> name table (not ids) so it can be audited by reading it. ``None``
means the class is discarded: its annotations are removed and it never
reaches the model. Final ids are assigned by
:func:`src.detection.common.coco_dataset.final_categories`: final names sorted
alphabetically -> 1..N, with 0 kept as background.

Design decision (2026-09-29): left/right pairs merge into one class. The part
name therefore no longer says which side of the car it is on; M3 assigns each
damage to the part *instance* with the largest overlap, and M4 must pair parts
of A and B by class *and* position in the image, not by name alone.

Training-split counts below are Roboflow-augmented: train holds 111 real
photos x 3 copies, so "3 train instances" is a single photo.

Closed 2026-09-29: the doubtful classes were decided by the user from the
visual sample in ``models/checkpoints/car_parts/_muestra_clases/`` (see the
"Decided from the visual sample" block). 29 final classes.
"""

from __future__ import annotations

CLASS_MAP = {
    # Discarded: fewer than 10 train instances (<= 2 real photos each).
    "air_intake": None,
    "left_side_door": None,
    "left_windowark": None,
    "right_windowark": None,
    "right_glass": None,
    "shield": None,

    # Left/right merged.
    "left_front_door": "front_door",
    "right_front_door": "front_door",
    "left_back_door": "back_door",
    "right_back_door": "back_door",
    "left_front_glass": "front_door_glass",
    "right_front_glass": "front_door_glass",
    "left_back_glass": "back_door_glass",
    "right_back_glass": "back_door_glass",
    "left_quater_glass": "quarter_glass",   # "quater" is the dataset's typo
    "right_quarter_glass": "quarter_glass",
    "left_front_light": "headlight",
    "right_front_lights": "headlight",
    "left_back_lights": "taillight",
    "right_back_light": "taillight",
    "left_fog_light": "fog_light",
    "right_fog_lights": "fog_light",

    # Decided from the visual sample (user, 2026-09-29).
    "back_light": None,              # inconsistent labels (rear window or bumper reflector)
    "fog_lights": "fog_light",       # same part as left/right fog lights
    "front_glass": "windshield",     # the windshield glass itself
    "windshield": None,              # despite its name, the wipers: discarded
    "bumper": "bumper",              # front bumper, kept apart from back_bumper
    "back_bumper": "back_bumper",
    "front_mirror": None,            # discarded
    "side_mirror": "side_mirror",

    # Unchanged (only typos/plurals fixed).
    "back_fender": "back_fender",
    "back_glass": "back_glass",
    "emblem": "emblem",
    "front_fender": "front_fender",
    "fuel_cover": "fuel_cover",
    "handle": "handle",
    "hood": "hood",
    "indicator_light": "indicator_light",
    "mudguard": "mudguard",
    "radiator": "radiator",
    "roof": "roof",
    "roof_trunk": "roof_trunk",
    "side_steps": "side_step",
    "step": "step",
    "tire": "tire",
    "trunk": "trunk",
    "wheel": "wheel",
}

#: Original classes whose final name is still provisional. Empty: the map is closed.
PENDING_DECISIONS = {}
