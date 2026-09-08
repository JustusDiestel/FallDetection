import csv
from pathlib import Path


LABEL_FILE = Path("data/urfall/urfall-cam0-falls.csv")


with open(LABEL_FILE, newline="") as file:
    reader = csv.reader(file)

    for i, row in enumerate(reader):
        print(row)

        if i == 9:
            break