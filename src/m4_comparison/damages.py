"""Classify each damage as new, preexisting or not visible (thesis module M4).

Deterministic, no neural network. Damages of the same type are paired between
A and B with the Hungarian method, in two stages:

1. **por parte** — inside each part matched by :mod:`src.m4_comparison.parts`,
   by the distance between their ``pos_relativa`` (centroid normalized to the
   part box), which is robust to small framing changes between the shots;
2. **imagen** — whatever is left (damages with no part, on a part seen in only
   one image, or assigned to a different part in each image, e.g. on a seam)
   by the distance between their box centres normalized to the image.

Then:

* a damage of B paired with one of A is **preexistente**;
* a damage of B with no pair is **nuevo**;
* a damage of A with no pair is **no_visible**: flagged for review, never
  dropped (it may be hidden, missed by M3, or repaired).

The distance threshold is not calibrated: there are no A/B pairs with ground
truth yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from src.m4_comparison.matching import match_points
from src.m4_comparison.parts import (DEFAULT_PART_MAX_DISTANCE, PartMatching, box_center,
                                     match_parts)

#: Maximum distance between ``pos_relativa`` (stage 1) or normalized box
#: centres (stage 2) of the same damage in A and B.
DEFAULT_DAMAGE_MAX_DISTANCE = 0.15

NUEVO = "nuevo"
PREEXISTENTE = "preexistente"
NO_VISIBLE = "no_visible"

POR_PARTE = "parte"
POR_IMAGEN = "imagen"


@dataclass(frozen=True)
class DamageStatus:
    """Verdict on one damage.

    ``id_a`` / ``id_b`` are the M3 damage ids; one of them is ``None`` for
    ``nuevo`` and ``no_visible``. ``distancia`` and ``metodo`` (``"parte"`` or
    ``"imagen"``) are set only for ``preexistente``.
    """

    estado: str
    tipo: str
    id_a: Optional[str]
    id_b: Optional[str]
    parte: Optional[str]
    distancia: Optional[float] = None
    metodo: Optional[str] = None


@dataclass
class Comparison:
    """M4 result for one A/B pair."""

    partes: PartMatching
    danos: List[DamageStatus]

    def count(self, estado: str) -> int:
        return sum(1 for d in self.danos if d.estado == estado)


def _rel_pos(dano: dict):
    return dano["pos_relativa"]


def _match_same_type(damages_a: Sequence[dict], damages_b: Sequence[dict], position_a,
                     position_b, max_distance: float) -> List[Tuple[dict, dict, float]]:
    """Hungarian matching type by type; ``position_x(damage)`` gives its point."""
    pairs = []
    for tipo in sorted({d["tipo"] for d in damages_a} & {d["tipo"] for d in damages_b}):
        in_a = [d for d in damages_a if d["tipo"] == tipo]
        in_b = [d for d in damages_b if d["tipo"] == tipo]
        points_a = [position_a(d) for d in in_a]
        points_b = [position_b(d) for d in in_b]
        for i, j, distance in match_points(points_a, points_b, max_distance):
            pairs.append((in_a[i], in_b[j], distance))
    return pairs


def classify_damages(report_a: dict, report_b: dict, part_matching: PartMatching,
                     max_distance: float = DEFAULT_DAMAGE_MAX_DISTANCE) -> List[DamageStatus]:
    """Classify every damage of A and B given the matched parts.

    Returns:
        One :class:`DamageStatus` per damage of B (in B's order), followed by
        one per unmatched damage of A (in A's order). Every damage of both
        reports appears exactly once.
    """
    danos_a, danos_b = report_a["danos"], report_b["danos"]
    matched: Dict[str, Tuple[dict, float, str]] = {}  # id in B -> (damage in A, distance, method)
    used_a = set()

    for part in part_matching.pares:
        in_a = [d for d in danos_a if d["parte_id"] == part.parte_id_a]
        in_b = [d for d in danos_b if d["parte_id"] == part.parte_id_b]
        for dano_a, dano_b, distance in _match_same_type(
                in_a, in_b, _rel_pos, _rel_pos, max_distance):
            matched[dano_b["id"]] = (dano_a, distance, POR_PARTE)
            used_a.add(dano_a["id"])

    left_a = [d for d in danos_a if d["id"] not in used_a]
    left_b = [d for d in danos_b if d["id"] not in matched]

    # Each side is normalized to its own image size.
    for dano_a, dano_b, distance in _match_same_type(
            left_a, left_b,
            lambda d: box_center(d["bbox"], report_a["width"], report_a["height"]),
            lambda d: box_center(d["bbox"], report_b["width"], report_b["height"]),
            max_distance):
        matched[dano_b["id"]] = (dano_a, distance, POR_IMAGEN)
        used_a.add(dano_a["id"])

    estados: List[DamageStatus] = []
    for dano in danos_b:
        if dano["id"] in matched:
            dano_a, distance, method = matched[dano["id"]]
            estados.append(DamageStatus(PREEXISTENTE, dano["tipo"], dano_a["id"], dano["id"],
                                        dano["parte"], round(distance, 4), method))
        else:
            estados.append(DamageStatus(NUEVO, dano["tipo"], None, dano["id"], dano["parte"]))
    for dano in danos_a:
        if dano["id"] not in used_a:
            estados.append(DamageStatus(NO_VISIBLE, dano["tipo"], dano["id"], None, dano["parte"]))
    return estados


def compare_reports(report_a: dict, report_b: dict,
                    part_max_distance: float = DEFAULT_PART_MAX_DISTANCE,
                    damage_max_distance: float = DEFAULT_DAMAGE_MAX_DISTANCE) -> Comparison:
    """Run M4 on the M3 reports of the before (A) and after (B) images."""
    part_matching = match_parts(report_a, report_b, part_max_distance)
    danos = classify_damages(report_a, report_b, part_matching, damage_max_distance)
    return Comparison(partes=part_matching, danos=danos)
