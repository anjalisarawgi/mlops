import click
import numpy as np
import matplotlib.pyplot as plt

import torch

from models.model import MyAwesomeModel


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print("Device", DEVICE)


@click.group()
def cli():
    """Command line interface."""

    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
@click.option("--batch_size", default=32, help="batch size to use for training")
@click.option("--epochs", default=10, help="number of epochs to train for")
def train(lr, batch_size, epochs) -> None:
    """Train a model on MNIST."""

    print("Training day and night")

    print(f"{lr=}, {batch_size=}, {epochs=}")

    model = MyAwesomeModel().to(DEVICE)

    train_img = torch.load("data/processed/train_images.pt")
    train_target = torch.load("data/processed/train_target.pt")
    train_set = torch.utils.data.TensorDataset(train_img, train_target)
    # print("train_set", train_set.shape)  # torch.Size([25000, 1, 28, 28])

    train_dataloader = torch.utils.data.DataLoader(
        train_set, batch_size=32)
    print("train_dataloader", len(train_dataloader))
    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    statistics = {"train_loss": [], "train_accuracy": []}

    for epoch in range(epochs):
        model.train()

        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
            statistics["train_loss"].append(loss.item())
            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"].append(accuracy)
            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")
    print("Training complete")

    torch.save(model.state_dict(), "models/model.pth")
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")

    axs[1].plot(statistics["train_accuracy"])

    axs[1].set_title("Train accuracy")

    fig.savefig("reports/figures/training_statistics.png")


cli.add_command(train)
if __name__ == "__main__":
    train()
