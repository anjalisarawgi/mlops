import torch

from torch import nn
import hydra


# @hydra.main(config_path="conf", config_name="model_conf.yaml", version_base="1.1")
class MyAwesomeModel(nn.Module):

    """My awesome model."""

    def __init__(self) -> None:
        super().__init__()
        hparams = config.model
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.dropout = nn.Dropout()
        self.fc1 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # print("x0 shape", x.shape)
        x = torch.relu(self.conv1(x))
        # print("x1 shape", x.shape)
        x = torch.max_pool2d(x, 2, 2)
        # print("x2 shape", x.shape)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.flatten(x, 1)

        x = self.dropout(x)
        x = self.fc1(x)

        return x


if __name__ == "__main__":
    model = MyAwesomeModel()
    print(f"Model architecture: {model}")
    print(
        f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
