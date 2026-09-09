from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fall_dataset import FallDataset
from fall_model import FallClassifier


DATA_DIR = Path("data/processed/split")

BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 0.001


train_dataset = FallDataset(
    DATA_DIR / "X_train.npy",
    DATA_DIR / "y_train.npy"
)

test_dataset = FallDataset(
    DATA_DIR / "X_test.npy",
    DATA_DIR / "y_test.npy"
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


model = FallClassifier()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


for epoch in range(EPOCHS):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for X_batch, y_batch in train_loader:
        print(X_batch.shape, y_batch.shape)
        optimizer.zero_grad()

        output = model(X_batch)

        loss = criterion(
            output,
            y_batch
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        predictions = torch.argmax(
            output,
            dim=1
        )

        correct += (
            predictions == y_batch
        ).sum().item()

        total += len(y_batch)

    accuracy = correct / total

    print(
        f"Epoch {epoch + 1:02d} | "
        f"Loss: {total_loss / len(train_loader):.4f} | "
        f"Train Accuracy: {accuracy:.3f}"
    )

torch.save(
    model.state_dict(),
    "fall_classifier.pt"
)