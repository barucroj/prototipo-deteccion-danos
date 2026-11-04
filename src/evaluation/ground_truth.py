"""Ground truth of an A/B batch labeled in CVAT and exported as COCO 1.0.

A batch is a folder with ``images/`` and ``annotations/instances_images.json``.
Each image is named ``pNN_<vista>_A.jpg`` or ``pNN_<vista>_B.jpg``; the two
photos of a pair share ``pNN`` and the view.

The same physical damage carries the same ``dano_id`` attribute in A and B.
Its expected state follows from where it appears: only in B is ``nuevo``, in
both is ``preexistente`` and only in A is ``no_visible``. An annotation without
``dano_id`` is a damage of its own, so it is ``nuevo`` in B and
``no_visible`` in A.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.m4_comparison.damages import NO_VISIBLE, NUEVO, PREEXISTENTE

IMAGES_SUBDIR = "images"
ANNOTATIONS_FILE = os.path.join("annotations", "instances_images.json")
PAIR_FILE_PATTERN = re.compile(r"^(p\d+)_(.+)_([AB])\.(?:jpe?g|png)$", re.IGNORECASE)
DAMAGE_ID_ATTRIBUTE = "dano_id"


class GroundTruthError(ValueError):
    """The batch does not follow the naming or labeling rules."""


@dataclass(frozen=True)
class LabeledDamage:
    """One annotation: ``bbox`` is ``[x1, y1, x2, y2]`` in pixels of its image."""

    annotation_id: int
    tipo: str
    bbox: Tuple[float, float, float, float]
    dano_id: Optional[int]


@dataclass(frozen=True)
class PhysicalDamage:
    """A real damage of the pair, seen in A, in B or in both.

    ``index_a`` / ``index_b`` point into :attr:`PairGroundTruth.danos_a` /
    ``danos_b``; ``None`` where the damage is not visible.
    """

    id: str
    tipo: str
    estado: str
    index_a: Optional[int]
    index_b: Optional[int]


@dataclass
class PairGroundTruth:
    par: str
    vista: str
    path_a: str
    path_b: str
    danos_a: List[LabeledDamage] = field(default_factory=list)
    danos_b: List[LabeledDamage] = field(default_factory=list)

    def physical_damages(self) -> List[PhysicalDamage]:
        """Group the annotations of A and B into real damages with their expected state."""
        by_id_a = _index_by_damage_id(self.danos_a, f"{self.par} A")
        by_id_b = _index_by_damage_id(self.danos_b, f"{self.par} B")

        result: List[PhysicalDamage] = []
        for i, dano in enumerate(self.danos_b):
            index_a = by_id_a.get(dano.dano_id) if dano.dano_id is not None else None
            if index_a is not None and self.danos_a[index_a].tipo != dano.tipo:
                raise GroundTruthError(f"{self.par}: dano_id {dano.dano_id} tiene otro tipo en A y en B")
            estado = PREEXISTENTE if index_a is not None else NUEVO
            result.append(PhysicalDamage(_physical_id(self.par, dano, "B"), dano.tipo, estado, index_a, i))
        for i, dano in enumerate(self.danos_a):
            if dano.dano_id is None or dano.dano_id not in by_id_b:
                result.append(PhysicalDamage(_physical_id(self.par, dano, "A"), dano.tipo, NO_VISIBLE, i, None))
        return result


def _index_by_damage_id(danos: List[LabeledDamage], where: str) -> Dict[int, int]:
    index: Dict[int, int] = {}
    for i, dano in enumerate(danos):
        if dano.dano_id is None:
            continue
        if dano.dano_id in index:
            raise GroundTruthError(f"{where}: dano_id {dano.dano_id} repetido en la misma imagen")
        index[dano.dano_id] = i
    return index


def _physical_id(par: str, dano: LabeledDamage, side: str) -> str:
    if dano.dano_id is not None:
        return f"{par}-d{dano.dano_id}"
    return f"{par}-{side}{dano.annotation_id}"


def _damage_id(annotation: dict) -> Optional[int]:
    value = annotation.get("attributes", {}).get(DAMAGE_ID_ATTRIBUTE)
    if value in (None, ""):
        return None
    return int(float(value))


def load_batch(batch_dir: str, annotations_path: Optional[str] = None) -> List[PairGroundTruth]:
    """Read a batch and return its pairs sorted by name.

    Raises:
        GroundTruthError: a file name does not follow ``pNN_<vista>_{A,B}``, a
            pair lacks one photo, an image of the JSON is not on disk or a
            ``dano_id`` is repeated in one image.
    """
    annotations_path = annotations_path or os.path.join(batch_dir, ANNOTATIONS_FILE)
    with open(annotations_path, encoding="utf-8") as handle:
        coco = json.load(handle)

    images_dir = os.path.join(batch_dir, IMAGES_SUBDIR)
    categories = {c["id"]: c["name"] for c in coco["categories"]}
    by_image: Dict[int, List[LabeledDamage]] = defaultdict(list)
    for ann in coco["annotations"]:
        x, y, w, h = ann["bbox"]
        by_image[ann["image_id"]].append(LabeledDamage(ann["id"], categories[ann["category_id"]],
                                                       (x, y, x + w, y + h), _damage_id(ann)))

    sides: Dict[Tuple[str, str], Dict[str, dict]] = defaultdict(dict)
    for image in coco["images"]:
        match = PAIR_FILE_PATTERN.match(image["file_name"])
        if not match:
            raise GroundTruthError(f"nombre fuera del formato pNN_<vista>_A|B: {image['file_name']}")
        par, vista, side = match.group(1).lower(), match.group(2), match.group(3).upper()
        if side in sides[(par, vista)]:
            raise GroundTruthError(f"{par} tiene dos imágenes {side}")
        path = os.path.join(images_dir, image["file_name"])
        if not os.path.isfile(path):
            raise GroundTruthError(f"no existe {path}")
        sides[(par, vista)][side] = {"path": path, "danos": by_image.get(image["id"], [])}

    pars = [p for p, _ in sides]
    repeated = sorted({p for p in pars if pars.count(p) > 1})
    if repeated:
        raise GroundTruthError(f"el mismo número de par con vistas distintas: {', '.join(repeated)}")

    pairs = []
    for (par, vista), images in sorted(sides.items()):
        missing = {"A", "B"} - set(images)
        if missing:
            raise GroundTruthError(f"a {par} le falta la imagen {''.join(sorted(missing))}")
        pairs.append(PairGroundTruth(par, vista, images["A"]["path"], images["B"]["path"],
                                     list(images["A"]["danos"]), list(images["B"]["danos"])))
    return pairs
