"""Ground truth of a CVAT COCO batch, on a synthetic batch in a temp folder."""

import json

import pytest

from src.evaluation.ground_truth import GroundTruthError, load_batch


def annotation(ann_id, image_id, category_id, bbox, dano_id=None):
    attributes = {"occluded": False}
    if dano_id is not None:
        attributes["dano_id"] = dano_id
    return {"id": ann_id, "image_id": image_id, "category_id": category_id, "bbox": bbox,
            "segmentation": [], "area": 1, "iscrowd": 0, "attributes": attributes}


def write_batch(root, names, annotations):
    (root / "images").mkdir()
    (root / "annotations").mkdir()
    for name in names:
        (root / "images" / name).write_bytes(b"")
    coco = {"categories": [{"id": 1, "name": "dent"}, {"id": 2, "name": "scratch"}],
            "images": [{"id": i + 1, "file_name": n, "width": 100, "height": 100} for i, n in enumerate(names)],
            "annotations": annotations}
    (root / "annotations" / "instances_images.json").write_text(json.dumps(coco), encoding="utf-8")
    return str(root)


def test_pairs_are_built_from_the_names_and_boxes_become_corners(tmp_path):
    batch = write_batch(tmp_path, ["p02_duster-back_B.jpg", "p01_duster-front_A.jpg",
                                   "p01_duster-front_B.jpg", "p02_duster-back_A.jpg"],
                        [annotation(1, 3, 1, [10, 20, 30, 40])])
    pairs = load_batch(batch)
    assert [(p.par, p.vista) for p in pairs] == [("p01", "duster-front"), ("p02", "duster-back")]
    assert pairs[0].path_a.endswith("p01_duster-front_A.jpg")
    assert pairs[0].danos_a == [] and pairs[0].danos_b[0].bbox == (10, 20, 40, 60)
    assert pairs[0].danos_b[0].tipo == "dent" and pairs[0].danos_b[0].dano_id is None


def test_states_follow_where_each_damage_id_appears(tmp_path):
    batch = write_batch(tmp_path, ["p01_x_A.jpg", "p01_x_B.jpg"],
                        [annotation(1, 1, 2, [0, 0, 5, 5], dano_id=1),
                         annotation(2, 1, 1, [50, 50, 5, 5], dano_id=2),
                         annotation(3, 2, 2, [1, 0, 5, 5], dano_id=1),
                         annotation(4, 2, 1, [80, 80, 5, 5], dano_id=3)])
    damages = {d.id: d for d in load_batch(batch)[0].physical_damages()}
    assert damages["p01-d1"].estado == "preexistente"
    assert (damages["p01-d1"].index_a, damages["p01-d1"].index_b) == (0, 0)
    assert damages["p01-d3"].estado == "nuevo" and damages["p01-d3"].index_a is None
    assert damages["p01-d2"].estado == "no_visible" and damages["p01-d2"].index_b is None


def test_damages_without_id_are_never_paired(tmp_path):
    batch = write_batch(tmp_path, ["p01_x_A.jpg", "p01_x_B.jpg"],
                        [annotation(1, 1, 1, [0, 0, 5, 5]), annotation(2, 2, 1, [0, 0, 5, 5])])
    states = sorted(d.estado for d in load_batch(batch)[0].physical_damages())
    assert states == ["no_visible", "nuevo"]


@pytest.mark.parametrize("names, message", [
    (["p01_x_A.jpg", "foto_B.jpg"], "formato"),
    (["p01_x_A.jpg"], "le falta la imagen B"),
    (["p01_x_A.jpg", "p01_y_B.jpg"], "vistas distintas"),
])
def test_bad_batches_are_rejected(tmp_path, names, message):
    batch = write_batch(tmp_path, names, [])
    with pytest.raises(GroundTruthError, match=message):
        load_batch(batch)


def test_missing_image_file_is_rejected(tmp_path):
    batch = write_batch(tmp_path, ["p01_x_A.jpg", "p01_x_B.jpg"], [])
    (tmp_path / "images" / "p01_x_B.jpg").unlink()
    with pytest.raises(GroundTruthError, match="no existe"):
        load_batch(batch)


def test_repeated_damage_id_in_one_image_is_rejected(tmp_path):
    batch = write_batch(tmp_path, ["p01_x_A.jpg", "p01_x_B.jpg"],
                        [annotation(1, 2, 1, [0, 0, 5, 5], dano_id=1), annotation(2, 2, 1, [50, 50, 5, 5], dano_id=1)])
    with pytest.raises(GroundTruthError, match="repetido"):
        load_batch(batch)[0].physical_damages()
