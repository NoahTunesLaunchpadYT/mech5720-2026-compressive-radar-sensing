"""
Demultiplexing algorithms for the coded-mask radar (numpy only, no hardware or UI).

Model, per range bin, with at most one point target:
    y_m = a * B_m(p) + noise        m = masks used, p = one of the 9x9 grid pixels
B is the calibrated beam matrix (masks x pixels): B_m(p) = |capture - ambient| of mask m
with the reflector at pixel p. A pixel whose column has any NaN (not calibrated for one
of the masks used) is never a candidate.

Strategy A: matched filter. Score = cosine similarity between y and each column; the
    maximum-likelihood choice for additive Gaussian noise with unknown amplitude.
Strategy B: cross-mask normalised shapes. Divide y and every column by their mean over
    masks and pick the column with the smallest squared distance; insensitive to gain
    and better suited to errors that scale with the signal.
"""

import numpy as np

STRATEGIES = ("A", "B")


def ca_cfar(x, guard, train, scale):
    """Cell-averaging CFAR threshold for profile x: scale x the mean of up to `train`
    cells each side, skipping `guard` cells next to the cell under test. Training
    windows are clipped at the ends of the profile."""
    n = len(x)
    c = np.concatenate(([0.0], np.cumsum(x)))
    thr = np.full(n, np.nan)
    for i in range(n):
        l0, l1 = max(0, i - guard - train), max(0, i - guard)
        r0, r1 = min(n, i + guard + 1), min(n, i + guard + train + 1)
        count = (l1 - l0) + (r1 - r0)
        if count:
            thr[i] = scale * ((c[l1] - c[l0]) + (c[r1] - c[r0])) / count
    return thr


def detect(x, thr, search, min_sep=0, min_peak=0.0):
    """Indices of detections, in range order: the peak of each contiguous run of cells
    inside the boolean `search` mask where x exceeds thr and min_peak. Peaks closer
    than min_sep cells keep only the strongest (one target spreads over several bins)."""
    hit = search & (x > thr) & (x >= min_peak)
    peaks, i = [], 0
    while i < len(x):
        if hit[i]:
            j = i
            while j < len(x) and hit[j]:
                j += 1
            peaks.append(i + int(np.argmax(x[i:j])))
            i = j
        else:
            i += 1
    kept = []
    for k in sorted(peaks, key=lambda k: -x[k]):
        if all(abs(k - q) >= min_sep for q in kept):
            kept.append(k)
    return sorted(kept)


def valid_pixels(B):
    """Candidate pixels: calibrated for every mask row (none when there are no rows)."""
    return np.isfinite(B).all(axis=0) & (len(B) > 0)


def localise(B, y, strategy, positions=None):
    """Most likely pixel for measurement y (one value per mask row of B).

    positions: (row, col) of each column in calibration-grid units. When given, the
    runner-up is the best candidate at least one grid pixel from the winner, so on an
    interpolated grid the margin still compares distinct locations, not neighbours.

    Returns dict: score (per pixel, NaN = not a candidate), best, second,
    higher_is_better, margin (A: best - second cosine; B: second / best cost)
    and amplitude (A only: least-squares a for the best column)."""
    ok = valid_pixels(B)
    score = np.full(B.shape[1], np.nan)
    if strategy == "A":
        cols = B[:, ok]
        score[ok] = cols.T @ y / (np.linalg.norm(cols, axis=0) * np.linalg.norm(y))
        order = np.argsort(-np.nan_to_num(score, nan=-np.inf))
    else:
        cols = B[:, ok] / B[:, ok].mean(axis=0)
        score[ok] = ((cols - (y / y.mean())[:, None]) ** 2).sum(axis=0)
        order = np.argsort(np.nan_to_num(score, nan=np.inf))
    order = order[:ok.sum()]                     # candidates only
    best = int(order[0])
    if positions is not None:
        far = np.hypot(*(positions[order] - positions[best]).T) >= 1
        order = order[far]
    else:
        order = order[1:]
    second = int(order[0]) if len(order) else None
    if strategy == "A":
        margin = score[best] - score[second] if second is not None else np.nan
        b = B[:, best]
        amplitude = float(b @ y / (b @ b))
    else:
        with np.errstate(divide="ignore"):   # an exact match has zero cost
            margin = score[second] / score[best] if second is not None else np.nan
        amplitude = np.nan
    return {"score": score, "best": best, "second": second,
            "higher_is_better": strategy == "A", "margin": float(margin),
            "amplitude": amplitude}


# Calibration grid: 9x9 points, 120 mm apart, on a board 1400 mm from the radar.
# Pixel (row, col): row 0 at the top, col 0 on the left as seen from the radar;
# pixel 44 is straight ahead.
GRID_SPACING_MM = 120.0
GRID_DISTANCE_MM = 1400.0
GRID_SIZE = 9
GRID_CENTRE = 4


def pixel_offset_mm(row, col):
    """(across, up) position of a grid point on the board, from its centre."""
    return (col - GRID_CENTRE) * GRID_SPACING_MM, (GRID_CENTRE - row) * GRID_SPACING_MM


def pixel_angles(row, col):
    """(bearing, elevation) in radians of the direction from the radar to a grid point.
    Bearing is positive to the right, elevation positive up."""
    x, y = pixel_offset_mm(row, col)
    return np.arctan(x / GRID_DISTANCE_MM), np.arctan(y / np.hypot(GRID_DISTANCE_MM, x))


def to_xyz(range_mm, bearing, elevation):
    """(across, forward, up) in mm of a point at slant range range_mm along a direction."""
    return (range_mm * np.cos(elevation) * np.sin(bearing),
            range_mm * np.cos(elevation) * np.cos(bearing),
            range_mm * np.sin(elevation))


def upsample_beams(B, factor):
    """Bilinear interpolation of the beam columns (masks x 81, the 9x9 grid) onto a grid
    `factor` times finer: masks x n*n with n = 8*factor + 1, in row-major order, where
    fine point (i, j) is grid position (i / factor, j / factor). A fine point is NaN
    unless every calibrated point it is interpolated from is valid."""
    n = (GRID_SIZE - 1) * factor + 1
    u = np.arange(n) / factor
    i0 = np.minimum(np.floor(u).astype(int), GRID_SIZE - 2)
    t = u - i0
    W = np.zeros((n, GRID_SIZE))                 # interpolation weights along one axis
    W[np.arange(n), i0] = 1 - t
    W[np.arange(n), i0 + 1] += t
    grid = B.reshape(len(B), GRID_SIZE, GRID_SIZE)
    fine = np.einsum("ir,mrc,jc->mij", W, np.nan_to_num(grid), W)
    uses = (W > 0).astype(float)
    bad = np.einsum("ir,mrc,jc->mij", uses, np.isnan(grid).astype(float), uses) > 0
    fine[bad] = np.nan
    return fine.reshape(len(B), n * n)
