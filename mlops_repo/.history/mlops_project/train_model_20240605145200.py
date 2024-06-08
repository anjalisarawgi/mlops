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

import click
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import os
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
import logging
import wandb
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors
import umap
from models.model import MyAwesomeModel
from mpl_toolkits.mplot3d import Axes3D
import torch.nn as nn
import torch.optim as optim

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


def evaluate_trustworthiness(original_data, reduced_data, n_neighbors=1):
    return trustworthiness(original_data, reduced_data, n_neighbors=n_neighbors)


def evaluate_continuity(original_data, reduced_data, n_neighbors=5):
    nbrs_original = NearestNeighbors(
        n_neighbors=n_neighbors).fit(original_data)
    nbrs_reduced = NearestNeighbors(n_neighbors=n_neighbors).fit(reduced_data)

    indices_original = nbrs_original.kneighbors(return_distance=False)
    indices_reduced = nbrs_reduced.kneighbors(return_distance=False)

    continuity_sum = 0.0
    n_samples = original_data.shape[0]

    for i in range(n_samples):
        intersect = len(set(indices_original[i]) & set(indices_reduced[i]))
        continuity_sum += intersect / n_neighbors

    continuity_score = continuity_sum / n_samples
    return continuity_score


def evaluate_mse(original_data, reconstructed_data):
    return mean_squared_error(original_data, reconstructed_data)


def plot_loss_landscape(weights_reduced, method_name, net, data_loader, criterion, pca=None, umap_reducer=None, autoencoder=None):
    grid_size = 30
    x = np.linspace(weights_reduced[:, 0].min(),
                    weights_reduced[:, 0].max(), grid_size)
    y = np.linspace(weights_reduced[:, 1].min(),
                    weights_reduced[:, 1].max(), grid_size)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    if method_name == "Autoencoder":
        grid_points_orig = autoencoder.decoder(torch.tensor(
            grid_points, dtype=torch.float32)).detach().numpy()
    elif method_name == "UMAP":
        grid_points_orig = umap_reducer.inverse_transform(grid_points)
    else:
        grid_points_orig = pca.inverse_transform(grid_points)

    grid_losses = []
    for i in range(grid_points_orig.shape[0]):
        projected_weight = grid_points_orig[i].reshape(
            net.conv1.lin.weight.data.shape)
        net.conv1.lin.weight.data = torch.tensor(
            projected_weight, dtype=torch.float32)

        batch_losses = []
        for data in data_loader:
            outputs = net(data)
            outputs = outputs.view(-1)
            loss = criterion(outputs, data.y.float()).item()
            batch_losses.append(loss)

        grid_losses.append(np.mean(batch_losses))

    grid_losses = np.array(grid_losses).reshape(xx.shape)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx, yy, grid_losses, cmap='viridis', edgecolor='none')
    ax.contour(xx, yy, grid_losses, zdir='z',
               offset=np.min(grid_losses), cmap='viridis')
    ax.set_xlabel(f'{method_name} Dimension 1')
    ax.set_ylabel(f'{method_name} Dimension 2')
    ax.set_zlabel('Loss')
    ax.set_title(f'3D {method_name} Visualization of Loss Landscape')
    plt.savefig(f'loss_landscape_{method_name.lower()}.png')
    plt.show()
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
    # wandb.init(project="mlops_project")
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
    print(f"Train image path: {train_img_path}")
    print(f"Train target path: {train_target_path}")

    # train_set = torch.utils.data.TensorDataset(
    #     train_img_path, train_target_path)

    # # Check if files exist before loading
    if not os.path.exists(train_img_path):
        print(f"Error: File not found - {train_img_path}")
        return

    if not os.path.exists(train_target_path):
        print(f"Error: File not found - {train_target_path}")
        return
    # # Load the data
    print("Loading data")
    try:
        train_img = torch.load(train_img_path)
        train_target = torch.load(train_target_path)
    except FileNotFoundError as e:
        # print(f"Error: {e}")
        print("Please check the paths and ensure the data files exist.")
        return

    train_set = torch.utils.data.TensorDataset(train_img, train_target)
    print("train_set", len(train_set))
    # print("train_set", train_set.shape)  # torch.Size([25000, 1, 28, 28])

    train_dataloader = torch.utils.data.DataLoader(
        train_set, hparams["batch_size"])
    print("train_dataloader", len(train_dataloader))
    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=hparams["lr"])

    statistics = {"train_loss": [], "train_accuracy": []}
    weight_snapshots = []
    loss_landscapes = []
    normalized_weights = []
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
                    # wandb.log({"examples": [wandb.Image(im)
                    #           for im in images_t]})
        # capture model weights
        weights = model.state_dict()["conv1.weight"].cpu().numpy().flatten()
        normalized_weights.append(normalize(weights.reshape(1, -1)))

    normalized_weights = np.array(normalized_weights).reshape(
        len(normalized_weights), -1)

    # Apply PCA
    pca = PCA(n_components=2)
    weights_pca = pca.fit_transform(normalized_weights)
    reconstructed_pca = pca.inverse_transform(weights_pca)

    trust_pca = evaluate_trustworthiness(normalized_weights, weights_pca)
    continuity_pca = evaluate_continuity(normalized_weights, weights_pca)
    mse_pca = evaluate_mse(normalized_weights, reconstructed_pca)
    print(f'Trustworthiness of PCA projection: {trust_pca:.2f}')
    print(f'Continuity of PCA projection: {continuity_pca:.2f}')
    print(f'MSE of PCA projection: {mse_pca:.4f}')

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    weights_tsne = tsne.fit_transform(normalized_weights)
    trust_tsne = evaluate_trustworthiness(normalized_weights, weights_tsne)
    continuity_tsne = evaluate_continuity(normalized_weights, weights_tsne)
    print(f'Trustworthiness of t-SNE projection: {trust_tsne:.2f}')
    print(f'Continuity of t-SNE projection: {continuity_tsne:.2f}')

    # Apply UMAP
    umap_reducer = umap.UMAP(n_components=2, random_state=42)
    weights_umap = umap_reducer.fit_transform(normalized_weights)
    trust_umap = evaluate_trustworthiness(normalized_weights, weights_umap)
    continuity_umap = evaluate_continuity(normalized_weights, weights_umap)
    print(f'Trustworthiness of UMAP projection: {trust_umap:.2f}')
    print(f'Continuity of UMAP projection: {continuity_umap:.2f}')

    plot_loss_landscape(weights_pca, "PCA", model, train_dataloader, loss_fn)
    plot_loss_landscape(weights_tsne, "t-SNE", model,
                        train_dataloader, loss_fn)
    plot_loss_landscape(weights_umap, "UMAP", model, train_dataloader, loss_fn)

    log.info("Training complete")
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), "models/model.pth")

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig("training_statistics.png")

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
    # wandb.log({"mnist_predictions": my_table})


# cli.add_command(train)
if __name__ == "__main__":
    train()
