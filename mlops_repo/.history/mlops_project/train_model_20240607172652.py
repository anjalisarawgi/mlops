import click
import numpy as np
import matplotlib.pyplot as plt

import torch

from models.model import MyAwesomeModel
import random
from sklearn.decomposition import PCA
from matplotlib import cm
from scipy.interpolate import griddata
import os
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
import logging
import wandb
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


log = logging.getLogger(__name__)

# a function to load the data


# # @hydra.main(config_path="config", config_name="config.yaml")
# @click.command()
# @click.option("--lr", default=1e-3, help="learning rate to use for training")
# @click.option("--batch_size", default=32, help="batch size to use for training")
# @click.option("--epochs", default=10, help="number of epochs to train for")
@hydra.main(config_path="config", config_name="training_conf.yaml", version_base="1.1")
def train(config) -> None:
    """Train a model on MNIST."""

    log.info("Training day and night")

    # print(f"{lr=}, {batch_size=}, {epochs=}")
    hparams = config.training
    # Load model config
    model_cfg = OmegaConf.load(
        "../../../mlops_project/config/model_conf.yaml")
    hparams_models = model_cfg.model
    wandb.init(project="mlops_project")
    model = MyAwesomeModel(hparams_models).to(DEVICE)

    # # hyparms = config.experiment
    # train_img = torch.load("../data/processed/train_images.pt")
    # # train_img = hyparms["train_images"]
    # print("train_image passed")
    # train_target = torch.load("../data/processed/train_target.pt")
    # # train_image = to_absolute_path(
    # Convert relative paths to absolute paths
    train_img_path = to_absolute_path("data/processed/train_images.pt")
    train_target_path = to_absolute_path("data/processed/train_target.pt")

    # Debugging: Print the resolved paths
    # print(f"Train image path: {train_img_path}")
    # print(f"Train target path: {train_target_path}")

    # train_set = torch.utils.data.TensorDataset(
    #     train_img_path, train_target_path)

    # # Check if files exist before loading
    # if not os.path.exists(train_img_path):
    #     print(f"Error: File not found - {train_img_path}")
    #     return

    # if not os.path.exists(train_target_path):
    #     print(f"Error: File not found - {train_target_path}")
    #     return
    # # # Load the data
    print("Loading data")
    train_img = torch.load(train_img_path)
    train_target = torch.load(train_target_path)
    # try:
    #     train_img = torch.load(train_img_path)
    #     train_target = torch.load(train_target_path)
    # except FileNotFoundError as e:
    #     # print(f"Error: {e}")
    #     print("Please check the paths and ensure the data files exist.")
    #     return

    train_set = torch.utils.data.TensorDataset(train_img, train_target)
    # print("train_set", len(train_set))
    # print("train_set", train_set.shape)  # torch.Size([25000, 1, 28, 28])

    train_dataloader = torch.utils.data.DataLoader(
        train_set, hparams["batch_size"])
    # print("train_dataloader", len(train_dataloader))
    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=hparams["lr"])

    statistics = {"train_loss": [], "train_accuracy": []}
    weight_snapshots = []
    loss_landscapes = []
    for epoch in range(hparams["epochs"]):
        model.train()

        for i, (img, target) in enumerate(train_dataloader):
            # wandb.Image(img[0].squeeze().numpy())
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
                log.info(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

                # Log image example to wandb
                if i == 0:
                    images_t = img.cpu()  # Convert to CPU for logging
                    wandb.log({"examples": [wandb.Image(im)
                              for im in images_t]})

    log.info("Training complete")
    # Ensure the models directory exists
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), "models/model.pth")
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")

    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")

    # fig.savefig("reports/figures/training_statistics.png")
    fig.savefig("training_statistics.png")

    # Log predictions with images in a table
    predictions_t = torch.cat([model(img.to(DEVICE)).argmax(
        dim=1).cpu() for img, _ in train_dataloader])
    my_table = wandb.Table()
    my_table.add_column("image", train_img.cpu())
    my_table.add_column("label", train_target.cpu())
    my_table.add_column("class_prediction", predictions_t)
    wandb.log({"mnist_predictions": my_table})


# cli.add_command(train)
if __name__ == "__main__":
    train()
