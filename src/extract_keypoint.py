from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


model = YOLO("yolo11n-pose.pt")

RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/keypoints")


def extract_sequence(folder):
    image_paths = sorted(folder.rglob("*.png"))

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

    return np.array(sequence), detected


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    folders = sorted(
        folder
        for folder in RAW_DIR.iterdir()
        if folder.is_dir()
        and (
            folder.name.startswith("fall-")
            or folder.name.startswith("adl-")
        )
    )

    print(f"Gefundene Sequenzen: {len(folders)}")
    print()

    for folder in folders:
        print(f"Verarbeite {folder.name}")

        sequence, detected = extract_sequence(folder)

        if len(sequence) == 0:
            print("Keine PNG-Dateien gefunden")
            print()
            continue

        output_file = OUTPUT_DIR / f"{folder.name}.npy"

        np.save(output_file, sequence)

        print(f"Frames: {len(sequence)}")
        print(f"Pose erkannt: {detected}")
        print(f"Shape: {sequence.shape}")
        print(f"Gespeichert: {output_file}")
        print()


if __name__ == "__main__":
    main()