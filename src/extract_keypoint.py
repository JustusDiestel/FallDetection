from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO


model = YOLO("yolo11n-pose.pt")


def extract_sequence(folder):
    folder = Path(folder)

    image_paths = sorted(folder.rglob("*.png"))

    print("Ordner:", folder.resolve())
    print("Existiert:", folder.exists())
    print("PNG-Dateien:", len(image_paths))

    sequence = []
    detected = 0

    for image_path in image_paths:
        frame = cv2.imread(str(image_path))

        results = model(frame, verbose=False)
        result = results[0]

        if result.keypoints is None or result.keypoints.xy.shape[0] == 0:
            keypoints_xy = np.full((17, 2), np.nan)
        else:
            keypoints_xy = result.keypoints.xy[0].cpu().numpy()
            detected += 1

        sequence.append(keypoints_xy)

    print("Posen erkannt:", detected)
    print()

    return np.array(sequence)

fall_sequence = extract_sequence("data/raw/fall-01-cam0-rgb")
adl_sequence = extract_sequence("data/raw/adl-01-cam0-rgb")

Path("data/keypoints").mkdir(parents=True, exist_ok=True)
np.save("data/keypoints/fall_sequence.npy", fall_sequence)
np.save("data/keypoints/adl_sequence.npy", adl_sequence)

print("Fall:", fall_sequence.shape)
print("ADL:", adl_sequence.shape)
