"""Distance-gated Hungarian matching shared by the part and damage stages of M4.

Both stages pair points of image A with points of image B so that the total
distance is minimal and no pair is farther apart than a threshold. Greedy
nearest-neighbour matching is order dependent and can steal a point from a
closer pair; :func:`scipy.optimize.linear_sum_assignment` is not.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def match_points(points_a: Sequence[Sequence[float]], points_b: Sequence[Sequence[float]],
                 max_distance: float) -> List[Tuple[int, int, float]]:
    """Pair points of A and B minimizing the total Euclidean distance.

    Args:
        points_a / points_b: ``(x, y)`` per point, in the same normalized space.
        max_distance: Pairs farther apart than this are not matched.

    Returns:
        ``(index_a, index_b, distance)`` per matched pair, sorted by
        ``index_a``. Unlisted indices have no match.

    Pairs beyond ``max_distance`` get a cost larger than any valid total, so
    the solver first maximizes the number of valid pairs and only then
    minimizes their distance; the invalid ones are dropped afterwards.
    """
    if len(points_a) == 0 or len(points_b) == 0:
        return []

    a = np.asarray(points_a, dtype=np.float64).reshape(-1, 2)
    b = np.asarray(points_b, dtype=np.float64).reshape(-1, 2)
    distances = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)

    invalid = distances > max_distance
    forbidden = 1.0 + distances[~invalid].sum() if (~invalid).any() else 1.0
    cost = np.where(invalid, forbidden, distances)

    rows, cols = linear_sum_assignment(cost)
    return [(int(r), int(c), float(distances[r, c]))
            for r, c in zip(rows, cols) if not invalid[r, c]]
