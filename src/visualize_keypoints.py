import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from preprocess_keypoints import fill_missing_frames, normalize_sequence


sequence = np.load(
    "data/keypoints/fall-01-cam0-rgb.npy"
)

sequence = fill_missing_frames(sequence)
sequence = normalize_sequence(sequence)


connections = [
    (5, 6),

    (5, 7),
    (7, 9),

    (6, 8),
    (8, 10),

    (5, 11),
    (6, 12),

    (11, 12),

    (11, 13),
    (13, 15),

    (12, 14),
    (14, 16),
]


fig, ax = plt.subplots()


def update(frame_index):
    ax.clear()

    pose = sequence[frame_index]

    x = pose[:, 0]
    y = pose[:, 1]

    ax.scatter(x, y)

    for start, end in connections:
        ax.plot(
            [pose[start, 0], pose[end, 0]],
            [pose[start, 1], pose[end, 1]]
        )

    ax.set_xlim(-3, 3)
    ax.set_ylim(3, -3)

    ax.set_aspect("equal")

    ax.set_title(f"Frame {frame_index}")


animation = FuncAnimation(
    fig,
    update,
    frames=len(sequence),
    interval=50
)

plt.show()