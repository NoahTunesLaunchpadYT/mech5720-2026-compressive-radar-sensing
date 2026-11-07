"""Display the intensity-versus-range readings for the right-sweep capture."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FILE = Path(__file__).parent / "Trial_Captures" / "02_right_sweep" / "scene_map.npz"


def main():
    with np.load(FILE) as data:
        scene = data["scene"]
        distances_m = data["distances_m"]

    assert scene.shape[1] == distances_m.size, "scene and distance axis do not match"
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    image = ax.imshow(
        scene,
        aspect="auto",
        origin="lower",
        extent=(distances_m[0], distances_m[-1], 0, scene.shape[0] - 1),
        cmap="turbo",
    )
    ax.set(title="Right-sweep coded-mask readings", xlabel="Distance (m)", ylabel="Mask position")
    fig.colorbar(image, ax=ax, label="Mean radar intensity")
    plt.show()


if __name__ == "__main__":
    main()
