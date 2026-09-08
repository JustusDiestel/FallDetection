import torch
import torch.nn as nn


class FallClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=34,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv1d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        self.relu = nn.ReLU()

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Linear(
            64,
            2
        )

    def forward(self, x):
        batch_size = x.shape[0]

        x = x.reshape(
            batch_size,
            30,
            34
        )

        x = x.transpose(1, 2)

        x = self.relu(
            self.conv1(x)
        )

        x = self.relu(
            self.conv2(x)
        )

        x = self.pool(x)

        x = x.squeeze(-1)

        x = self.classifier(x)

        return x


if __name__ == "__main__":

    model = FallClassifier()

    test_input = torch.randn(
        8,
        30,
        17,
        2
    )

    output = model(test_input)

    print("Input:", test_input.shape)
    print("Output:", output.shape)
    print(output)