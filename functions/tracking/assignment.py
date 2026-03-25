from __future__ import annotations

from typing import Iterable

import numpy as np


def _hungarian(cost: np.ndarray) -> list[tuple[int, int]]:
    """
    Minimal Hungarian algorithm for square cost matrices.
    Returns list of (row, col) assignments.
    """
    cost = cost.copy()
    n = cost.shape[0]
    # Step 1: subtract row minima
    cost -= cost.min(axis=1, keepdims=True)
    # Step 2: subtract column minima
    cost -= cost.min(axis=0, keepdims=True)

    # Masks: 0=none, 1=starred, 2=primed
    mask = np.zeros((n, n), dtype=np.int8)
    row_covered = np.zeros(n, dtype=bool)
    col_covered = np.zeros(n, dtype=bool)

    # Step 3: star zeros
    for r in range(n):
        for c in range(n):
            if cost[r, c] == 0 and not row_covered[r] and not col_covered[c]:
                mask[r, c] = 1
                row_covered[r] = True
                col_covered[c] = True
    row_covered[:] = False
    col_covered[:] = False

    def cover_columns_with_stars():
        col_covered[:] = False
        for c in range(n):
            if np.any(mask[:, c] == 1):
                col_covered[c] = True

    def find_uncovered_zero():
        for r in range(n):
            if row_covered[r]:
                continue
            for c in range(n):
                if col_covered[c]:
                    continue
                if cost[r, c] == 0:
                    return r, c
        return None

    def find_star_in_row(r):
        cols = np.where(mask[r] == 1)[0]
        return cols[0] if cols.size else None

    def find_star_in_col(c):
        rows = np.where(mask[:, c] == 1)[0]
        return rows[0] if rows.size else None

    def find_prime_in_row(r):
        cols = np.where(mask[r] == 2)[0]
        return cols[0] if cols.size else None

    def augment_path(start_r, start_c):
        path = [(start_r, start_c)]
        while True:
            r = find_star_in_col(path[-1][1])
            if r is None:
                break
            path.append((r, path[-1][1]))
            c = find_prime_in_row(path[-1][0])
            path.append((path[-1][0], c))
        for r, c in path:
            if mask[r, c] == 1:
                mask[r, c] = 0
            else:
                mask[r, c] = 1

    def clear_primes():
        mask[mask == 2] = 0

    cover_columns_with_stars()
    while np.sum(col_covered) < n:
        zero = find_uncovered_zero()
        if zero is None:
            # Step 6: adjust matrix
            min_uncovered = np.min(cost[~row_covered][:, ~col_covered])
            cost[row_covered] += min_uncovered
            cost[:, ~col_covered] -= min_uncovered
            continue
        r, c = zero
        mask[r, c] = 2  # prime it
        star_col = find_star_in_row(r)
        if star_col is not None:
            row_covered[r] = True
            col_covered[star_col] = False
        else:
            augment_path(r, c)
            row_covered[:] = False
            col_covered[:] = False
            clear_primes()
            cover_columns_with_stars()

    assignments = []
    for r in range(n):
        c = np.where(mask[r] == 1)[0]
        if c.size:
            assignments.append((r, int(c[0])))
    return assignments


def hungarian_assign(
    left_points: Iterable[tuple[float, float]],
    right_points: Iterable[tuple[float, float]],
    max_dist: float,
) -> list[tuple[int, int, float]]:
    """
    One-to-one assignment between left_points and right_points using Hungarian algorithm.
    Returns (left_idx, right_idx, dist) for assignments within max_dist.
    """
    left = np.asarray(list(left_points), dtype=np.float32)
    right = np.asarray(list(right_points), dtype=np.float32)
    if left.size == 0 or right.size == 0:
        return []

    n = left.shape[0]
    m = right.shape[0]
    size = max(n, m)
    big = max_dist * 1000.0 if max_dist > 0 else 1e6

    cost = np.full((size, size), big, dtype=np.float32)
    for i in range(n):
        for j in range(m):
            dx = left[i, 0] - right[j, 0]
            dy = left[i, 1] - right[j, 1]
            dist = float((dx * dx + dy * dy) ** 0.5)
            if dist <= max_dist:
                cost[i, j] = dist

    assignments = _hungarian(cost)
    matches = []
    for i, j in assignments:
        if i < n and j < m:
            dist = float(cost[i, j])
            if dist <= max_dist:
                matches.append((i, j, dist))
    return matches
