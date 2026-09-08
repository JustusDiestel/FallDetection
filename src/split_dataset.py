from pathlib import Path
import numpy as np


DATA_DIR = Path("data/processed")

X = np.load(DATA_DIR / "X.npy")
y = np.load(DATA_DIR / "y.npy")
groups = np.load(DATA_DIR / "groups.npy")


unique_groups = np.unique(groups)

print("Alle Sequenzen:")
print(unique_groups)


fall_groups = sorted([
    group for group in unique_groups
    if group.startswith("fall-")
])

adl_groups = sorted([
    group for group in unique_groups
    if group.startswith("adl-")
])


test_fall_groups = fall_groups[-1:]
test_adl_groups = adl_groups[-1:]

test_groups = test_fall_groups + test_adl_groups

train_groups = [
    group for group in unique_groups
    if group not in test_groups
]


train_mask = np.isin(groups, train_groups)
test_mask = np.isin(groups, test_groups)


X_train = X[train_mask]
y_train = y[train_mask]

X_test = X[test_mask]
y_test = y[test_mask]


print()
print("Train-Sequenzen:")
print(train_groups)

print()
print("Test-Sequenzen:")
print(test_groups)


print()
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("Train NORMAL:", np.sum(y_train == 0))
print("Train FALL:", np.sum(y_train == 1))


print()
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

print("Test NORMAL:", np.sum(y_test == 0))
print("Test FALL:", np.sum(y_test == 1))

OUTPUT_DIR = DATA_DIR / "split"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

np.save(OUTPUT_DIR / "X_train.npy", X_train)
np.save(OUTPUT_DIR / "y_train.npy", y_train)

np.save(OUTPUT_DIR / "X_test.npy", X_test)
np.save(OUTPUT_DIR / "y_test.npy", y_test)

print()
print("Split gespeichert in:", OUTPUT_DIR)