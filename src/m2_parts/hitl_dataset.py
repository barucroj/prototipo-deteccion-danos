"""Humans in the Loop "Car parts and car damages" dataset: Supervisely -> COCO.

Source: https://humansintheloop.org/resources/datasets/car-parts-and-car-damages-dataset/
(CC0 1.0). Only the 998 **part** images are used. Note that the two folders of
the download are named the other way round: the parts are in
``Car damages dataset/File1`` (21 part classes) and the damages in
``Car parts dataset/File1``.

Supervisely format: ``meta.json`` (classes) and, per image, ``ann/<image>.json``
with ``size`` and ``objects``; each object is a ``polygon`` with
``points.exterior`` and optional ``points.interior`` rings (holes), in pixels.

COCO polygons cannot hold holes and :class:`CocoDetectionDataset` does not read
RLE, so a polygon with holes (140 of 15,767, almost all doors: the hole is the
window) is written as **several hole-free polygons** (:func:`polygon_pieces`):
the exact mask (exterior minus holes) is cut by a 1-px column through each hole
and the outer contours of the pieces become the COCO polygon list. The cost is
that column of pixels per hole; IoU against the exact mask is checked in the
tests and reported by the conversion. (A single "keyhole" ring was tried first:
median IoU 0.98 but down to 0.74 when the cut crossed other edges.)

Categories keep the original titles (``Front-door``...), ids 1..21 in
alphabetical order; :mod:`src.m2_parts.class_map_hitl` maps them to the ten
final classes.

**Door criterion aligned with Roboflow** (decided 2026-11-06, option A): in
HITL a door's polygon includes its window (and usually the mirror), which
are also labeled apart; in Roboflow the door excludes them. :func:`fix_door`
subtracts from each door the union of the windows (``Front-window``,
``Back-window``) and mirrors that lie mostly inside it (more than
:data:`DOOR_INSIDE` of their area), drops fragments under
:data:`DOOR_MIN_FRAGMENT` of the original door area and discards the door if
it keeps less than :data:`DOOR_MIN_KEPT` of it. The result is written as
hole-free pieces like any other polygon with holes. Same rule on every split.

**Window frame removed** (decided 2026-11-06): after subtracting the windows
the frame above them stayed in the door (median 5.3 % of its area) and took
the box up to the roof; Roboflow's doors end at the window's lower edge.
:func:`cut_frame` erases, column by column, the door pixels above the lower
edge of the subtracted windows; outside the windows' columns it uses a line
fitted to that edge (the belt line), so the pillars of the frame go too.

Splits are assigned by **base photo** (:func:`assign_splits`): every image of
one group of near-duplicates goes to the same split, so a split cannot hold a
copy of another split's photo (the leak found in Carparts-Seg).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

DEFAULT_ROOT = os.path.join("data", "raw", "Car parts and car damages dataset", "Car damages dataset")
DEFAULT_OUTPUT = os.path.join("data", "raw", "hitl_parts_coco")
SPLITS = ("train", "val", "test")

#: Door correction (option A, 2026-11-06).
DOOR_CLASSES = ("Front-door", "Back-door")
DOOR_SUBTRACT = ("Front-window", "Back-window", "Mirror")
DOOR_INSIDE = 0.5        # a window/mirror is subtracted if > 50 % of it lies inside the door
DOOR_MIN_FRAGMENT = 0.05  # door fragments under 5 % of the original door area are dropped
DOOR_MIN_KEPT = 0.4       # a door that loses more than 60 % of its area is discarded
DOOR_WINDOWS = ("Front-window", "Back-window")  # their lower edge is the door's upper edge
DOOR_REMOVE_FRAME = True  # cut the window frame above that edge (2026-11-06)


# ------------------------------------------------------------------------ geometry
def polygon_area(points) -> float:
    p = np.asarray(points, dtype=float)
    if len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2)


def polygon_mask(exterior, interiors: Sequence, width: int, height: int) -> np.ndarray:
    """Exact mask: ``exterior`` filled, ``interiors`` erased (uint8 0/1)."""
    import cv2

    mask = np.zeros((height, width), np.uint8)
    cv2.fillPoly(mask, [np.round(np.asarray(exterior, dtype=float)).astype(np.int32)], 1)
    for ring in interiors:
        cv2.fillPoly(mask, [np.round(np.asarray(ring, dtype=float)).astype(np.int32)], 0)
    return mask


def polygon_pieces(exterior, interiors: Sequence, width: int, height: int) -> List[List[float]]:
    """Flat COCO polygons, without holes, covering ``exterior`` minus ``interiors``.

    Without holes it is the exterior itself. With holes, the exact mask is cut
    by a 1-px column through each hole's centroid and the pieces' outer
    contours are returned; if a hole survives (the column missed it), it is
    cut again through its own centroid.
    """
    import cv2

    if not interiors:
        return [[float(v) for v in np.asarray(exterior, dtype=float).reshape(-1)]]
    mask = polygon_mask(exterior, interiors, width, height)
    for ring in interiors:
        cx = int(round(float(np.mean(np.asarray(ring, dtype=float)[:, 0]))))
        mask[:, min(max(cx, 0), width - 1)] = 0
    for _ in range(10):
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        holes = [c for c, hh in zip(contours, hierarchy[0]) if hh[3] >= 0] if hierarchy is not None else []
        if not holes:
            break
        for c in holes:
            mask[:, int(round(float(c[:, 0, 0].mean())))] = 0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [[float(v) for v in c.reshape(-1)] for c in contours if len(c) >= 3]


def mask_pieces(mask: np.ndarray) -> List[List[float]]:
    """Flat COCO polygons, without holes, covering a binary mask (holes cut by 1-px columns)."""
    import cv2

    mask = (mask > 0).astype(np.uint8).copy()
    for _ in range(20):
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        holes = [c for c, hh in zip(contours, hierarchy[0]) if hh[3] >= 0] if hierarchy is not None else []
        if not holes:
            break
        for c in holes:
            mask[:, int(round(float(c[:, 0, 0].mean())))] = 0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [[float(v) for v in c.reshape(-1)] for c in contours if len(c) >= 3]


def cut_frame(door: np.ndarray, windows: np.ndarray) -> np.ndarray:
    """``door`` without the pixels above the lower edge of ``windows`` (bool masks).

    The edge is the belt line: a line fitted to the window's lowest pixel per
    column, using only the columns where that pixel is at or below its median
    (the bottom of the window; its slanted sides would tilt the line and leave
    a pillar of the frame standing), clamped to the range of those columns
    and applied to every column. Where the window reaches below the line, the
    window's own lowest pixel is used.
    """
    door = door.astype(bool)
    windows = windows.astype(bool)
    cols = np.nonzero(windows.any(0))[0]
    if not len(cols):
        return door
    h = windows.shape[0]
    bottom = h - 1 - np.argmax(windows[::-1, cols], axis=0)
    xs = np.arange(door.shape[1])
    low = bottom >= np.median(bottom)
    if len(np.unique(cols[low])) > 1:
        slope, icept = np.polyfit(cols[low].astype(float), bottom[low].astype(float), 1)
        edge = np.clip(slope * xs + icept, bottom[low].min(), bottom.max())
    else:
        edge = np.full(xs.shape, float(bottom.max()))
    edge[cols] = np.maximum(edge[cols], bottom)
    above = np.arange(h)[:, None] <= np.round(edge)[None, :]
    return door & ~above


def fix_door(door: np.ndarray, others: Sequence[np.ndarray], inside: float = DOOR_INSIDE,
             min_fragment: float = DOOR_MIN_FRAGMENT, min_kept: float = DOOR_MIN_KEPT,
             is_window: Optional[Sequence[bool]] = None, remove_frame: bool = False,
             info: Optional[dict] = None):
    """Door mask minus the ``others`` (windows, mirrors) lying mostly inside it.

    With ``remove_frame``, the door pixels above the lower edge of the
    subtracted windows (``is_window`` marks which ``others`` are windows) are
    removed as well (:func:`cut_frame`). Returns ``(mask or None if discarded,
    number of others subtracted)``; the mask is the input unchanged when
    nothing is subtracted. ``info``, if given, receives ``frame_px`` (pixels
    removed by the frame cut).
    """
    import cv2

    door = door.astype(bool)
    area = int(door.sum())
    cut = np.zeros_like(door)
    win = np.zeros_like(door)
    n = 0
    for k, m in enumerate(others):
        m = m.astype(bool)
        if m.sum() and (m & door).sum() > inside * m.sum():
            cut |= m
            n += 1
            if is_window is not None and is_window[k]:
                win |= m
    if info is not None:
        info["frame_px"] = 0
    if not n or not (cut & door).any():
        return door, 0
    new = door & ~cut
    if remove_frame and win.any():
        framed = new
        new = cut_frame(new, win)
        if info is not None:
            info["frame_px"] = int(framed.sum() - new.sum())
    count, labels, comp, _ = cv2.connectedComponentsWithStats(new.astype(np.uint8), connectivity=8)
    for k in range(1, count):
        if comp[k, cv2.CC_STAT_AREA] < min_fragment * area:
            new[labels == k] = False
    if new.sum() < min_kept * area:
        return None, n
    return new, n


# --------------------------------------------------------------------- conversion
def read_meta(root: str) -> List[str]:
    with open(os.path.join(root, "meta.json"), encoding="utf-8") as f:
        return sorted(c["title"] for c in json.load(f)["classes"])


def list_images(root: str) -> List[str]:
    """Image file names that have an annotation, sorted."""
    names = [os.path.basename(p)[:-len(".json")] for p in glob.glob(os.path.join(root, "File1", "ann", "*.json"))]
    return sorted(names)


def convert(root: str, image_names: Optional[Iterable[str]] = None,
            door_fix: bool = True) -> Tuple[dict, dict]:
    """COCO dict (and validation stats) for ``image_names`` (default: all).

    Image ids are the position in the sorted list of all images + 1, so they
    are stable across splits. Objects with fewer than 3 points, zero area or
    an empty mask are dropped and counted in the stats. ``door_fix`` applies
    :func:`fix_door` (stats under ``"door_fix"``).
    """
    import cv2

    classes = read_meta(root)
    cat_id = {name: i for i, name in enumerate(classes, start=1)}
    all_names = list_images(root)
    image_id = {n: i for i, n in enumerate(all_names, start=1)}
    wanted = all_names if image_names is None else sorted(image_names)

    images, annotations = [], []
    stats = {"objects": 0, "with_holes": 0, "dropped": {"menos_de_3_puntos": 0, "area_0": 0,
                                                       "mascara_vacia": 0},
             "out_of_image": 0, "images_without_objects": 0, "holes_iou_min": None}
    doors = {"doors": 0, "modified": 0, "discarded": 0, "subtracted_objects": 0, "area_kept": [],
             "frame_cut": []}
    for name in wanted:
        with open(os.path.join(root, "File1", "ann", name + ".json"), encoding="utf-8") as f:
            ann = json.load(f)
        w, h = ann["size"]["width"], ann["size"]["height"]
        images.append({"id": image_id[name], "file_name": name, "width": w, "height": h})
        kept = 0
        door_masks = {}
        if door_fix:
            for k, obj in enumerate(ann["objects"]):
                if obj["classTitle"] in DOOR_CLASSES + DOOR_SUBTRACT and len(obj["points"]["exterior"]) >= 3:
                    rings = [r for r in (obj["points"].get("interior") or []) if len(r) >= 3]
                    door_masks[k] = polygon_mask(obj["points"]["exterior"], rings, w, h).astype(bool)
        for k, obj in enumerate(ann["objects"]):
            stats["objects"] += 1
            if obj.get("geometryType") != "polygon":
                raise ValueError(f"{name}: unsupported geometry {obj.get('geometryType')!r}")
            ext = obj["points"]["exterior"]
            holes = [r for r in (obj["points"].get("interior") or []) if len(r) >= 3]
            if len(ext) < 3:
                stats["dropped"]["menos_de_3_puntos"] += 1
                continue
            area = polygon_area(ext) - sum(polygon_area(r) for r in holes)
            if area <= 0:
                stats["dropped"]["area_0"] += 1
                continue
            e = np.asarray(ext, dtype=float)
            if (e < -1).any() or (e[:, 0] > w + 1).any() or (e[:, 1] > h + 1).any():
                stats["out_of_image"] += 1
            fixed = None
            if door_fix and obj["classTitle"] in DOOR_CLASSES and k in door_masks:
                doors["doors"] += 1
                subs = [j for j in door_masks if ann["objects"][j]["classTitle"] in DOOR_SUBTRACT]
                info = {}
                fixed, n_sub = fix_door(door_masks[k], [door_masks[j] for j in subs],
                                        is_window=[ann["objects"][j]["classTitle"] in DOOR_WINDOWS for j in subs],
                                        remove_frame=DOOR_REMOVE_FRAME, info=info)
                if fixed is None:
                    doors["discarded"] += 1
                    doors["subtracted_objects"] += n_sub
                    continue
                if n_sub:
                    doors["modified"] += 1
                    doors["subtracted_objects"] += n_sub
                    doors["area_kept"].append(float(fixed.sum() / max(1, door_masks[k].sum())))
                    doors["frame_cut"].append(info["frame_px"] / max(1, door_masks[k].sum()))
                else:
                    fixed = None
            if fixed is not None:
                pieces = mask_pieces(fixed)
                area = float(fixed.sum())
                ys, xs = np.nonzero(fixed)
                e = np.stack([xs, ys], 1).astype(float)
                e = np.concatenate([e, e + 1])
                holes = []
            else:
                pieces = polygon_pieces(ext, holes, w, h)
            mask = np.zeros((h, w), np.uint8)
            cv2.fillPoly(mask, [np.round(np.asarray(pc).reshape(-1, 2)).astype(np.int32) for pc in pieces], 1)
            if not mask.any():
                stats["dropped"]["mascara_vacia"] += 1
                continue
            if holes:
                stats["with_holes"] += 1
                exact = polygon_mask(ext, holes, w, h)
                iou = float((mask & exact).sum() / max(1, (mask | exact).sum()))
                stats["holes_iou_min"] = iou if stats["holes_iou_min"] is None else min(stats["holes_iou_min"], iou)
            x0, y0 = np.clip(e.min(0), 0, [w, h])
            x1, y1 = np.clip(e.max(0), 0, [w, h])
            annotations.append({"id": len(annotations) + 1, "image_id": image_id[name],
                                "category_id": cat_id[obj["classTitle"]], "segmentation": pieces,
                                "area": float(area), "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                                "iscrowd": 0})
            kept += 1
        stats["images_without_objects"] += kept == 0
    if door_fix:
        kept_area = np.asarray(doors.pop("area_kept"))
        doors["area_kept_percentiles"] = ({f"p{q}": float(np.percentile(kept_area, q)) for q in (0, 5, 25, 50, 75, 95, 100)}
                                          if len(kept_area) else {})
        doors["area_kept_histogram"] = {f"{lo:.1f}-{lo + 0.1:.1f}": int(((kept_area >= lo) & (kept_area < lo + 0.1 + (lo >= 0.9))).sum())
                                        for lo in np.arange(0.4, 1.0, 0.1).round(1)}
        frame = np.asarray(doors.pop("frame_cut"))
        doors["frame_cut_doors"] = int((frame > 0).sum())
        doors["frame_cut_percentiles"] = ({f"p{q}": float(np.percentile(frame, q)) for q in (0, 5, 25, 50, 75, 95, 100)}
                                          if len(frame) else {})
        doors["rule"] = {"subtract": list(DOOR_SUBTRACT), "inside_more_than": DOOR_INSIDE,
                         "min_fragment": DOOR_MIN_FRAGMENT, "discard_if_kept_below": DOOR_MIN_KEPT,
                         "remove_frame_above": list(DOOR_WINDOWS) if DOOR_REMOVE_FRAME else None}
        stats["door_fix"] = doors
    coco = {"images": images, "annotations": annotations,
            "categories": [{"id": i, "name": n, "supercategory": "car_part"} for n, i in cat_id.items()]}
    return coco, stats


# ------------------------------------------------------------------------- splits
def assign_splits(groups: Mapping[str, str], seed: int = 0,
                  fractions: Sequence[float] = (0.8, 0.1, 0.1),
                  forced: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """``{image: split}`` assigning whole groups (base photos) to one split each.

    Args:
        groups: ``{image: group id}``; every image of a group gets the same split.
        fractions: Target share of **groups** for train, val, test.
        forced: ``{group id: split}`` for groups whose split is fixed
            beforehand (e.g. ``"excluded"`` for copies of Roboflow val/test).
    """
    forced = dict(forced or {})
    ids = sorted(set(groups.values()) - set(forced))
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_train = int(round(fractions[0] * n))
    n_val = int(round(fractions[1] * n))
    split_of = dict(forced)
    for k, g in enumerate(ids):
        split_of[g] = "train" if k < n_train else "val" if k < n_train + n_val else "test"
    return {img: split_of[g] for img, g in groups.items()}


def write_splits(root: str, output: str, split_of: Mapping[str, str]) -> Dict[str, dict]:
    """Write ``instances_{split}.json`` for each split in ``split_of``; return the stats."""
    os.makedirs(output, exist_ok=True)
    out = {}
    for split in sorted(set(split_of.values())):
        coco, stats = convert(root, [n for n, s in split_of.items() if s == split])
        with open(os.path.join(output, f"instances_{split}.json"), "w", encoding="utf-8") as f:
            json.dump(coco, f)
        out[split] = {"images": len(coco["images"]), "annotations": len(coco["annotations"]), **stats}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Convert the HITL car-parts images (Supervisely) to COCO.")
    p.add_argument("--root", default=DEFAULT_ROOT, help="Folder with meta.json and File1/ (the PARTS).")
    p.add_argument("--splits", required=True,
                   help="JSON {image file name: split}; split 'excluded' is written but never used.")
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    args = p.parse_args(argv)
    with open(args.splits, encoding="utf-8") as f:
        split_of = json.load(f)
    summary = write_splits(args.root, args.output, split_of)
    with open(os.path.join(args.output, "resumen_conversion.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
