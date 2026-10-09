"""Damage-to-part assignment on synthetic masks.

Every case is a small hand-drawn layout where the right answer is obvious by
construction, so a failure points at the rule, not at a model.
"""

import numpy as np
import pytest

from src.damage_assignment.assign import (
    DEFAULT_MIN_OVERLAP,
    assign_damage,
    assign_damages,
    overlap_fractions,
    relative_position,
)

H, W = 100, 200


def rect(x1, y1, x2, y2):
    """Boolean (H, W) mask filled on ``[y1:y2, x1:x2]``."""
    mask = np.zeros((H, W), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


# Two doors side by side: same class in M2, told apart only by instance.
LEFT_DOOR = rect(0, 0, 100, 100)
RIGHT_DOOR = rect(100, 0, 200, 100)
PARTS = [LEFT_DOOR, RIGHT_DOOR]
BOXES = [[0, 0, 100, 100], [100, 0, 200, 100]]


def test_overlap_is_fraction_of_damage_area_not_of_part():
    damage = rect(80, 40, 120, 60)  # 20 px wide on each door
    np.testing.assert_allclose(overlap_fractions(damage, PARTS), [0.5, 0.5])


def test_overlap_of_empty_damage_is_zero():
    empty = np.zeros((H, W), dtype=bool)
    np.testing.assert_array_equal(overlap_fractions(empty, PARTS), [0.0, 0.0])


def test_damage_fully_inside_a_part_is_assigned_to_it():
    result = assign_damage(rect(120, 20, 140, 40), PARTS, BOXES)
    assert result.part_index == 1
    assert result.overlap == pytest.approx(1.0)


def test_damage_goes_to_the_part_with_the_largest_overlap():
    damage = rect(70, 40, 110, 60)  # 30 px on the left door, 10 on the right
    result = assign_damage(damage, PARTS, BOXES)
    assert result.part_index == 0
    assert result.overlap == pytest.approx(0.75)


def test_tie_goes_to_the_first_part():
    result = assign_damage(rect(80, 40, 120, 60), PARTS, BOXES)
    assert result.part_index == 0
    assert result.overlap == pytest.approx(0.5)


def test_overlap_exactly_at_threshold_is_accepted():
    damage = rect(50, 80, 60, 100)
    half_covering = [rect(0, 90, 200, 100)]
    result = assign_damage(damage, half_covering, [[0, 90, 200, 100]])
    assert result.overlap == pytest.approx(DEFAULT_MIN_OVERLAP)
    assert result.part_index == 0


def test_damage_below_threshold_is_kept_unassigned():
    damage = rect(50, 40, 60, 100)
    small_part = [rect(0, 90, 200, 100)]  # covers 1/6 of the damage
    result = assign_damage(damage, small_part, [[0, 90, 200, 100]])
    assert result.part_index is None
    assert result.rel_pos is None
    assert result.overlap == pytest.approx(1 / 6)


def test_no_parts_leaves_damage_unassigned():
    result = assign_damage(rect(10, 10, 20, 20), [], [])
    assert result.part_index is None
    assert result.overlap == 0.0


def test_min_overlap_is_configurable():
    damage = rect(70, 40, 110, 60)  # best overlap 0.75
    assert assign_damage(damage, PARTS, BOXES, min_overlap=0.8).part_index is None


def test_mismatched_parts_and_boxes_raise():
    with pytest.raises(ValueError):
        assign_damage(rect(10, 10, 20, 20), PARTS, BOXES[:1])


def test_relative_position_is_centroid_normalized_to_the_part_box():
    damage = rect(120, 20, 140, 40)  # centroid (129.5, 29.5)
    rel_x, rel_y = relative_position(damage, BOXES[1])
    assert rel_x == pytest.approx(0.295)
    assert rel_y == pytest.approx(0.295)


def test_relative_position_does_not_depend_on_where_the_part_is_in_the_image():
    # Same spot on each door -> same rel_pos, which is what M4 compares.
    on_left = relative_position(rect(20, 20, 40, 40), BOXES[0])
    on_right = relative_position(rect(120, 20, 140, 40), BOXES[1])
    assert on_left == pytest.approx(on_right)


def test_relative_position_is_clipped_to_the_part_box():
    spilling = rect(150, 0, 200, 100)  # centroid right of the left door's box
    rel_x, rel_y = relative_position(spilling, BOXES[0])
    assert rel_x == 1.0
    assert 0.0 <= rel_y <= 1.0


def test_relative_position_of_degenerate_inputs_is_none():
    assert relative_position(np.zeros((H, W), dtype=bool), BOXES[0]) is None
    assert relative_position(rect(10, 10, 20, 20), [50, 50, 50, 80]) is None


def test_assign_damages_keeps_every_damage_in_order():
    damages = [rect(120, 20, 140, 40), rect(50, 40, 60, 100), rect(10, 10, 30, 30)]
    only_right = [RIGHT_DOOR]
    results = assign_damages(damages, only_right, [BOXES[1]])
    assert [r.part_index for r in results] == [0, None, None]
