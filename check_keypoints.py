import numpy as np

sequence = np.load(
    "src/data/keypoints/fall-01-cam0-rgb.npy"
)

print("Shape:", sequence.shape)

nan_frames = np.isnan(sequence).all(axis=(1, 2))

print("Frames insgesamt:", len(sequence))
print("Frames ohne Pose:", nan_frames.sum())
print("Frames mit Pose:", (~nan_frames).sum())