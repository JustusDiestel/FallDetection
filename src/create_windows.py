from pathlib import Path
import csv
import numpy as np

from preprocess_keypoints import fill_missing_frames, normalize_sequence, extract_features

KEYPOINT_DIR = Path("data/keypoints")
LABEL_FILE = Path("data/urfall/urfall-cam0-falls.csv")
OUTPUT_DIR = Path("data/processed")

WINDOW_SIZE = 30
STEP_SIZE = 10


def load_labels():
    labels = {}

    with open(LABEL_FILE, newline="") as file:
        reader = csv.reader(file)

        for row in reader:
            sequence_name = row[0]
            frame_number = int(row[1])
            label = int(row[2])

            labels[(sequence_name, frame_number)] = label

    return labels


def create_fall_windows(sequence, sequence_name, labels):
    X = []
    y = []

    for start in range(0, len(sequence) - WINDOW_SIZE + 1, STEP_SIZE):
        end = start + WINDOW_SIZE

        window = sequence[start:end]

        frame_labels = []

        for frame_index in range(start, end):
            frame_number = frame_index + 1

            label = labels[
                (sequence_name, frame_number)
            ]

            frame_labels.append(label)

        if 0 in frame_labels or 1 in frame_labels:
            window_label = 1
        else:
            window_label = 0

        X.append(window)
        y.append(window_label)

    return X, y


def create_adl_windows(sequence):
    X = []
    y = []

    for start in range(0, len(sequence) - WINDOW_SIZE + 1, STEP_SIZE):
        end = start + WINDOW_SIZE

        window = sequence[start:end]

        X.append(window)
        y.append(0)

    return X, y


def main():
    labels = load_labels()

    all_X = []
    all_y = []
    all_groups = []

    for file_path in sorted(KEYPOINT_DIR.glob("fall-*.npy")):
        sequence = np.load(file_path)

        sequence = fill_missing_frames(sequence)
        sequence = extract_features(sequence)

        sequence_name = file_path.name.split("-cam")[0]

        X, y = create_fall_windows(
            sequence,
            sequence_name,
            labels
        )

        all_X.extend(X)
        all_y.extend(y)

        all_groups.extend(
            [sequence_name] * len(X)
        )

    for file_path in sorted(KEYPOINT_DIR.glob("adl-*.npy")):
        sequence = np.load(file_path)

        sequence = fill_missing_frames(sequence)
        sequence = extract_features(sequence)

        sequence_name = file_path.name.split("-cam")[0]

        X, y = create_adl_windows(sequence)

        all_X.extend(X)
        all_y.extend(y)

        all_groups.extend(
            [sequence_name] * len(X)
        )

    X = np.array(all_X)
    y = np.array(all_y)
    groups = np.array(all_groups)

    print("X:", X.shape)
    print("y:", y.shape)
    print("groups:", groups.shape)

    print("NORMAL:", np.sum(y == 0))
    print("FALL:", np.sum(y == 1))

    print("NaNs in X:", np.isnan(X).sum())

    print("Sequenzen:", np.unique(groups))

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(OUTPUT_DIR / "X.npy",X)
    np.save(OUTPUT_DIR / "y.npy",y)
    np.save(OUTPUT_DIR / "groups.npy",groups)

    print()
    print("Gespeichert:")
    print(OUTPUT_DIR / "X.npy")
    print(OUTPUT_DIR / "y.npy")
    print(OUTPUT_DIR / "groups.npy")


if __name__ == "__main__":
    main()