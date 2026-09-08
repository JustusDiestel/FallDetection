from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader

DATA_DIR = Path("data/processed/split")


class FallDataset(Dataset):

    def __init__(self, x_file, y_file):
        self.X = np.load(x_file)
        self.y = np.load(y_file)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = torch.tensor(
            self.X[index],
            dtype=torch.float32
        )

        y = torch.tensor(
            self.y[index],
            dtype=torch.long
        )

        return x, y


train_dataset = FallDataset(
    DATA_DIR / "X_train.npy",
    DATA_DIR / "y_train.npy"
)

test_dataset = FallDataset(
    DATA_DIR / "X_test.npy",
    DATA_DIR / "y_test.npy"
)


print("Train Samples:", len(train_dataset))
print("Test Samples:", len(test_dataset))

x, y = train_dataset[0]

print("X Shape:", x.shape)
print("Label:", y)


train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False
)


for X_batch, y_batch in train_loader:
    print("X Batch:", X_batch.shape)
    print("y Batch:", y_batch.shape)
    print("Labels:", y_batch)
    break