import csv
from pathlib import Path


LABEL_FILE = Path("data/urfall/urfall-cam0-falls.csv")


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


labels = load_labels()

print(labels[("fall-01", 1)])
print(labels[("fall-01", 10)])