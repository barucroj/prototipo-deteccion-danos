"""Match the part instances of image A with those of image B (thesis module M4).

M2 merges left/right pairs into one class (``front_door``), so two parts of
the same class can be on one photo and the name alone does not say which is
which. Parts are paired per class by the position of their box centre,
normalized to each image's size so A and B may differ in resolution.

The capture protocol guarantees that A and B show the same view of the car;
``max_distance`` only absorbs small framing differences between the shots.

Works on the M3 JSON reports (``partes`` with ``id``, ``clase_parte`` and
``bbox``), so it is testable without models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from src.m4_comparison.matching import match_points

#: Maximum distance between normalized box centres of the same part in A and B.
#: Not calibrated: there are no A/B pairs with ground truth yet.
DEFAULT_PART_MAX_DISTANCE = 0.25


@dataclass(frozen=True)
class PartMatch:
    """One part instance seen in both images."""

    clase_parte: str
    parte_id_a: str
    parte_id_b: str
    distancia: float


@dataclass
class PartMatching:
    """Result of pairing the parts of A and B.

    ``solo_a`` / ``solo_b`` hold the ids of parts with no counterpart: hidden
    or missed by M2 in the other image, or further than ``max_distance``.
    """

    pares: List[PartMatch]
    solo_a: List[str]
    solo_b: List[str]

    def a_to_b(self) -> Dict[str, str]:
        """Part id in A -> matched part id in B."""
        return {m.parte_id_a: m.parte_id_b for m in self.pares}


def box_center(bbox, width: float, height: float) -> Tuple[float, float]:
    """Centre of an ``[x1, y1, x2, y2]`` box, normalized to the image size."""
    x1, y1, x2, y2 = (float(v) for v in bbox)
    return (x1 + x2) / 2.0 / width, (y1 + y2) / 2.0 / height


def match_parts(report_a: dict, report_b: dict,
                max_distance: float = DEFAULT_PART_MAX_DISTANCE) -> PartMatching:
    """Pair the parts of two M3 reports, class by class, with the Hungarian method.

    Args:
        report_a / report_b: M3 reports of the before and after images.
        max_distance: Maximum distance between normalized box centres.
    """
    parts_a, parts_b = report_a["partes"], report_b["partes"]
    classes = sorted({p["clase_parte"] for p in parts_a} | {p["clase_parte"] for p in parts_b})

    pares: List[PartMatch] = []
    for clase in classes:
        in_a = [p for p in parts_a if p["clase_parte"] == clase]
        in_b = [p for p in parts_b if p["clase_parte"] == clase]
        centers_a = [box_center(p["bbox"], report_a["width"], report_a["height"]) for p in in_a]
        centers_b = [box_center(p["bbox"], report_b["width"], report_b["height"]) for p in in_b]
        for i, j, distance in match_points(centers_a, centers_b, max_distance):
            pares.append(PartMatch(clase, in_a[i]["id"], in_b[j]["id"], round(distance, 4)))

    matched_a = {m.parte_id_a for m in pares}
    matched_b = {m.parte_id_b for m in pares}
    return PartMatching(
        pares=pares,
        solo_a=[p["id"] for p in parts_a if p["id"] not in matched_a],
        solo_b=[p["id"] for p in parts_b if p["id"] not in matched_b],
    )
