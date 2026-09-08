import numpy as np


def fill_missing_frames(sequence):
    sequence = sequence.copy()

    valid_frames = ~np.isnan(sequence).all(axis=(1, 2))

    if not valid_frames.any():
        return np.zeros_like(sequence)

    first_valid = np.where(valid_frames)[0][0]

    sequence[:first_valid] = sequence[first_valid]

    for i in range(first_valid + 1, len(sequence)):
        if np.isnan(sequence[i]).all():
            sequence[i] = sequence[i - 1]

    return sequence



def normalize_sequence(sequence):
    sequence = sequence.copy()

    for i in range(len(sequence)):
        pose = sequence[i]

        left_shoulder = pose[5]
        right_shoulder = pose[6]
        left_hip = pose[11]
        right_hip = pose[12]

        hip_center = (left_hip + right_hip) / 2
        shoulder_center = (left_shoulder + right_shoulder) / 2

        body_size = np.linalg.norm(
            shoulder_center - hip_center
        )

        if body_size < 1e-6:
            continue

        pose = pose - hip_center
        pose = pose / body_size

        sequence[i] = pose

    return sequence