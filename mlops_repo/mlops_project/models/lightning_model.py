import pytorch_lightning as pl
import torch
from torch import nn
import wandb
# @hydra.main(config_path="conf", config_name="model_conf.yaml", version_base="1.1")


class LightningModel(pl.LightningModule):

    """My awesome model."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.dropout = nn.Dropout(0.05)
        self.fc1 = nn.Linear(128, 10)

        self.loss_fn = nn.CrossEntropyLoss()

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

    def training_step(self, batch, batch_idx):
        x, y = batch  # x = images, y = labels
        x, y = x.to(self.device), y.to(self.device)
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        # self.log("train_loss", loss, prog_bar=True)
        # self.logger.experiment.log({'logits': wandb.Histogram(y_pred)})
        self.logger.experiment.log(
            {'logits': wandb.Histogram(y_pred.cpu().detach().numpy())})
        acc = (y == y_pred.argmax(dim=-1)).float().mean()
        self.log('train_loss', loss)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


if __name__ == "__main__":
    model = LightningModel()
    print(f"Model architecture: {model}")
    print(
        f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
