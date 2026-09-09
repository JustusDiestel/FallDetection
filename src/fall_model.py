import torch.nn as nn
import torch.nn.functional as F
import torch

class FallClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=40,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1
        )

        self.pool = nn.AdaptiveMaxPool1d(1)

        self.classifier = nn.Linear(
            128,
            2
        )

    def forward(self, x):
        x = x.transpose(1, 2)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = self.pool(x)
        x = x.squeeze(-1)

        x = self.classifier(x)

        return x

if __name__ == "__main__":

    model = FallClassifier()

    test_input = torch.randn(8,30,40)

    output = model(test_input)

    print("Input:", test_input.shape)
    print("Output:", output.shape)
    print(output)