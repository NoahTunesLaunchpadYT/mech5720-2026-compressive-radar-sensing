"""Demultiplex a sliding 4x4 coded-mask radar capture into target positions.

Example:
    python demux2.py Trial_Captures/10_02

The input folder must contain mask00.npz ... mask23.npz.  The script writes
demux2.npz and demux2_heatmap.png beside those readings.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).parent
MASK_FILE = ROOT / "mask_patterns" / "mask_pattern.csv"
N_MASKS = 24
GRID_SHAPE = (4, 4)


def load_y(folder):
    """Return y and r from the 24 raw radar captures."""
    y = []
    r = None
    for m in range(N_MASKS):
        with np.load(folder / f"mask{m:02d}.npz") as capture:
            r_m = capture["distances_m"]
            if r is None:
                r = r_m
            elif not np.array_equal(r, r_m):
                raise ValueError("all captures must use the same distance axis")
            # y_m(r) = mean_{frame,sweep} |s_m(frame, sweep, r)|.
            y.append(np.abs(capture["frames"]).mean(axis=(0, 1)))
    return np.asarray(y), r


def make_A():
    """Return the 24 by 16 binary coding matrix A for the sliding 4x4 aperture."""
    pattern = np.loadtxt(MASK_FILE, delimiter=",", dtype=float)
    if pattern.shape != (4, 27):
        raise ValueError("mask_pattern.csv must be a 4 by 27 binary pattern")
    # A_mn = mask value at pixel n when the aperture is at position m.
    return np.array([pattern[:, m:m + 4].ravel() for m in range(N_MASKS)])


def demultiplex(A, y, y_0):
    """Estimate the range-resolved image x_hat from coded measurements y."""
    if A.shape != (N_MASKS, np.prod(GRID_SHAPE)) or y.shape[0] != N_MASKS:
        raise ValueError("A and y must describe all 24 mask positions")
    if y_0.shape != y.shape:
        raise ValueError("ambient measurement y_0 must have the same shape as y")
    # y - y_0 = A x + e  =>  x_hat = argmin_x ||A x - (y - y_0)||_2^2.
    x_hat, _, _, _ = np.linalg.lstsq(A, y - y_0, rcond=None)
    return x_hat.reshape(*GRID_SHAPE, y.shape[1])


def estimate_depth(x_hat, r):
    """Return each grid cell's non-negative peak distance r_hat and intensity."""
    x_plus = np.maximum(x_hat, 0.0)
    k_hat = x_plus.argmax(axis=-1)
    intensity = x_plus.max(axis=-1)
    # r_hat(i, j) = r[argmax_k max(x_hat(i, j, k), 0)].
    r_hat = r[k_hat].astype(float)
    noise = np.median(np.abs(x_hat - np.median(x_hat)))
    threshold = max(5 * noise, x_plus.max() * 1e-12, np.finfo(float).eps)
    r_hat[intensity < threshold] = np.nan
    return r_hat, intensity, threshold


def save_heatmap(r_hat, intensity, output):
    """Save the requested grid-position versus target-distance heatmap."""
    cmap = plt.colormaps["turbo"].copy()
    cmap.set_bad("#303030")
    fig, ax = plt.subplots(figsize=(5.4, 4.8), constrained_layout=True)
    image = ax.imshow(r_hat, cmap=cmap, vmin=np.nanmin(r_hat), vmax=np.nanmax(r_hat))
    ax.set(title="Demultiplexed target distance", xlabel="Grid column", ylabel="Grid row",
           xticks=range(GRID_SHAPE[1]), yticks=range(GRID_SHAPE[0]))
    for i, j in np.ndindex(GRID_SHAPE):
        if np.isfinite(r_hat[i, j]):
            ax.text(j, i, f"{r_hat[i, j]:.2f} m\n{intensity[i, j]:.0f}",
                    ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(image, ax=ax, label="Estimated target distance (m)")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def self_test():
    A = make_A()
    x = np.zeros((16, 3))
    x[6, 1] = 10
    x_hat = demultiplex(A, A @ x, np.zeros((24, 3)))
    r_hat, _, _ = estimate_depth(x_hat, np.array([0.5, 1.0, 1.5]))
    assert np.isclose(r_hat[1, 2], 1.0) and np.isnan(r_hat).sum() == 15


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", nargs="?", type=Path, default=ROOT / "Trial_Captures" / "10_02")
    parser.add_argument("--ambient", type=Path, default=ROOT / "Trial_Captures" / "ambient")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("demux2 self-test passed")
        return

    A = make_A()
    y, r = load_y(args.scene)
    y_0, r_0 = load_y(args.ambient)
    if not np.array_equal(r, r_0):
        raise ValueError("scene and ambient distance axes differ")
    x_hat = demultiplex(A, y, y_0)
    r_hat, intensity, threshold = estimate_depth(x_hat, r)
    if not np.isfinite(r_hat).any():
        raise ValueError("no reconstructed target exceeds the noise threshold")
    np.savez(args.scene / "demux2.npz", A=A, y=y, y_0=y_0, r=r, x_hat=x_hat,
             r_hat=r_hat, intensity=intensity, threshold=threshold)
    output = args.scene / "demux2_heatmap.png"
    save_heatmap(r_hat, intensity, output)
    print(f"saved {output} (threshold {threshold:.3g})")


if __name__ == "__main__":
    main()
