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


def extract_features(sequence):
    features = []

    previous_hip_center = None
    for pose in sequence:
        left_shoulder = pose[5]
        right_shoulder = pose[6]
        left_hip = pose[11]
        right_hip = pose[12]

        shoulder_center = (left_shoulder + right_shoulder) / 2
        hip_center = (left_hip + right_hip) / 2
        body_size = np.linalg.norm(shoulder_center - hip_center)



        normalized_pose = (pose -hip_center) / body_size

        if previous_hip_center is None:
            hip_velocity = np.zeros(2)
        else:
            hip_velocity = hip_center - previous_hip_center

        previous_hip_center = hip_center.copy()

        body_vector = shoulder_center - hip_center
        body_angle = np.arctan2(body_vector[1], body_vector[0])

        min_xy = np.min(pose, axis=0)
        max_xy = np.max(pose, axis=0)
        width = max_xy[0] - min_xy[0]
        height = max_xy[1] - min_xy[1]
        aspect_ratio = width / height
        frame_features = np.concatenate([
            normalized_pose.flatten(),
            hip_center,
            hip_velocity,
            np.array([body_angle,aspect_ratio])
        ])
        features.append(frame_features)
    return np.array(features,dtype=np.float32)

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