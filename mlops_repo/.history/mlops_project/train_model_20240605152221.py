# import click
# import numpy as np
# import matplotlib.pyplot as plt

# import torch

# from models.model import MyAwesomeModel
# import random
# from sklearn.decomposition import PCA
# from matplotlib import cm
# from scipy.interpolate import griddata
# import os
# import hydra
# from hydra.utils import to_absolute_path
# from omegaconf import DictConfig, OmegaConf
# import logging
# import wandb


# from mpl_toolkits.mplot3d import Axes3D
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import normalize
# from sklearn.manifold import trustworthiness
# # import networkx as nx
# import random
# from sklearn.metrics import mean_squared_error
# from sklearn.neighbors import NearestNeighbors
# DEVICE = torch.device(
#     "cuda"
#     if torch.cuda.is_available()
#     else "mps"
#     if torch.backends.mps.is_available()
#     else "cpu"
# )
# print("Device", DEVICE)


# @click.group()
# def cli():
#     """Command line interface."""

#     pass


# log = logging.getLogger(__name__)


# def evaluate_trustworthiness(original_data, reduced_data, n_neighbors=5):
#     return trustworthiness(original_data, reduced_data, n_neighbors=n_neighbors)


# def evaluate_continuity(original_data, reduced_data, n_neighbors=5):
#     nbrs_original = NearestNeighbors(
#         n_neighbors=n_neighbors).fit(original_data)
#     nbrs_reduced = NearestNeighbors(n_neighbors=n_neighbors).fit(reduced_data)

#     indices_original = nbrs_original.kneighbors(return_distance=False)
#     indices_reduced = nbrs_reduced.kneighbors(return_distance=False)

#     continuity_sum = 0.0
#     n_samples = original_data.shape[0]

#     for i in range(n_samples):
#         intersect = len(set(indices_original[i]) & set(indices_reduced[i]))
#         continuity_sum += intersect / n_neighbors

#     continuity_score = continuity_sum / n_samples
#     return continuity_score


# def evaluate_mse(original_data, reconstructed_data):
#     return mean_squared_error(original_data, reconstructed_data)


# def plot_loss_landscape(weights_reduced, method_name, net, data_loader, criterion, autoencoder=None, umap_reducer=None, pca=None):
#     grid_size = 30
#     x = np.linspace(weights_reduced[:, 0].min(),
#                     weights_reduced[:, 0].max(), grid_size)
#     y = np.linspace(weights_reduced[:, 1].min(),
#                     weights_reduced[:, 1].max(), grid_size)
#     xx, yy = np.meshgrid(x, y)
#     grid_points = np.c_[xx.ravel(), yy.ravel()]

#     # For autoencoder and UMAP, use inverse transform
#     if method_name == "Autoencoder":
#         grid_points_orig = autoencoder.decoder(torch.tensor(
#             grid_points, dtype=torch.float32)).detach().numpy()
#     elif method_name == "UMAP":
#         grid_points_orig = umap_reducer.inverse_transform(grid_points)
#     else:
#         # PCA and t-SNE do not have inverse transform, so use PCA for both
#         grid_points_orig = pca.inverse_transform(grid_points)

#     grid_losses = []
#     for i in range(grid_points_orig.shape[0]):
#         # Set the network weights to the projected weights
#         projected_weight = grid_points_orig[i].reshape(
#             net.conv1.lin.weight.data.shape)
#         net.conv1.lin.weight.data = torch.tensor(
#             projected_weight, dtype=torch.float32)

#         # Forward pass and loss computation
#         batch_losses = []
#         for data in data_loader:
#             outputs = net(data)
#             outputs = outputs.view(-1)
#             loss = criterion(outputs, data.y.float()).item()
#             batch_losses.append(loss)

#         grid_losses.append(np.mean(batch_losses))

#     grid_losses = np.array(grid_losses).reshape(xx.shape)
#     fig = plt.figure(figsize=(12, 8))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot_surface(xx, yy, grid_losses, cmap='viridis', edgecolor='none')
#     ax.contour(xx, yy, grid_losses, zdir='z',
#                offset=np.min(grid_losses), cmap='viridis')
#     ax.set_xlabel(f'{method_name} Dimension 1')
#     ax.set_ylabel(f'{method_name} Dimension 2')
#     ax.set_zlabel('Loss')
#     ax.set_title(f'3D {method_name} Visualization of Loss Landscape')
#     plt.savefig(f'loss_landscape_{method_name.lower()}.png')
#     plt.show()


# # # @hydra.main(config_path="config", config_name="config.yaml")
# # @click.command()
# # @click.option("--lr", default=1e-3, help="learning rate to use for training")
# # @click.option("--batch_size", default=32, help="batch size to use for training")
# # @click.option("--epochs", default=10, help="number of epochs to train for")
# @hydra.main(config_path="config", config_name="training_conf.yaml", version_base="1.1")
# def train(config) -> None:
#     """Train a model on MNIST."""

#     log.info("Training day and night")

#     # print(f"{lr=}, {batch_size=}, {epochs=}")
#     hparams = config.training
#     # Load model config
#     model_cfg = OmegaConf.load(
#         "../../../mlops_project/config/model_conf.yaml")
#     hparams_models = model_cfg.model
#     # wandb.init(project="mlops_project")
#     model = MyAwesomeModel(hparams_models).to(DEVICE)

#     # # hyparms = config.experiment
#     # train_img = torch.load("../data/processed/train_images.pt")
#     # # train_img = hyparms["train_images"]
#     # print("train_image passed")
#     # train_target = torch.load("../data/processed/train_target.pt")
#     # # train_image = to_absolute_path(
#     # Convert relative paths to absolute paths
#     train_img_path = to_absolute_path("data/processed/train_images.pt")
#     train_target_path = to_absolute_path("data/processed/train_target.pt")

#     # Debugging: Print the resolved paths
#     print(f"Train image path: {train_img_path}")
#     print(f"Train target path: {train_target_path}")

#     # train_set = torch.utils.data.TensorDataset(
#     #     train_img_path, train_target_path)

#     # # Check if files exist before loading
#     if not os.path.exists(train_img_path):
#         print(f"Error: File not found - {train_img_path}")
#         return

#     if not os.path.exists(train_target_path):
#         print(f"Error: File not found - {train_target_path}")
#         return
#     # # Load the data
#     print("Loading data")
#     try:
#         train_img = torch.load(train_img_path)
#         train_target = torch.load(train_target_path)
#     except FileNotFoundError as e:
#         # print(f"Error: {e}")
#         print("Please check the paths and ensure the data files exist.")
#         return

#     train_set = torch.utils.data.TensorDataset(train_img, train_target)
#     print("train_set", len(train_set))
#     # print("train_set", train_set.shape)  # torch.Size([25000, 1, 28, 28])

#     train_dataloader = torch.utils.data.DataLoader(
#         train_set, hparams["batch_size"])
#     print("train_dataloader", len(train_dataloader))
#     loss_fn = torch.nn.CrossEntropyLoss()

#     optimizer = torch.optim.Adam(model.parameters(), lr=hparams["lr"])

#     statistics = {"train_loss": [], "train_accuracy": []}
#     # weight_snapshots = []
#     # loss_landscapes = []
#     normalized_weights = []
#     for epoch in range(hparams["epochs"]):
#         model.train()

#         for i, (img, target) in enumerate(train_dataloader):
#             # wandb.Image(img[0].squeeze().numpy())
#             img, target = img.to(DEVICE), target.to(DEVICE)
#             optimizer.zero_grad()
#             y_pred = model(img)
#             loss = loss_fn(y_pred, target)
#             loss.backward()
#             optimizer.step()
#             statistics["train_loss"].append(loss.item())
#             accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
#             statistics["train_accuracy"].append(accuracy)
#             if i % 100 == 0:
#                 log.info(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

#                 # Log image example to wandb
#                 if i == 0:
#                     images_t = img.cpu()  # Convert to CPU for logging
#                     # wandb.log({"examples": [wandb.Image(im)
#                     #           for im in images_t]})
#             # Capture weights
#         weights = model.state_dict()["conv1.weight"].cpu().numpy().flatten()
#         normalized_weights.append(normalize(weights.reshape(1, -1)))

#     normalized_weights = np.array(normalized_weights).reshape(
#         len(normalized_weights), -1)
#     print("normalized_weights", normalized_weights.shape)

#     log.info("Training complete")
#     # Ensure the models directory exists
#     os.makedirs('models', exist_ok=True)
#     torch.save(model.state_dict(), "models/model.pth")
#     fig, axs = plt.subplots(1, 2, figsize=(15, 5))
#     axs[0].plot(statistics["train_loss"])
#     axs[0].set_title("Train loss")

#     axs[1].plot(statistics["train_accuracy"])
#     axs[1].set_title("Train accuracy")

#     # fig.savefig("reports/figures/training_statistics.png")
#     fig.savefig("training_statistics.png")

#     # Log predictions with images in a table
#     predictions_t = torch.cat([model(img.to(DEVICE)).argmax(
#         dim=1).cpu() for img, _ in train_dataloader])
#     # my_table = wandb.Table()
#     # my_table.add_column("image", train_img.cpu())
#     # my_table.add_column("label", train_target.cpu())
#     # my_table.add_column("class_prediction", predictions_t)
#     # wandb.log({"mnist_predictions": my_table})


# # cli.add_command(train)
# if __name__ == "__main__":
#     train()

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
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print("Device", DEVICE)


@click.group()
def cli():
    """Command line interface."""
    pass


log = logging.getLogger(__name__)


def evaluate_trustworthiness(original_data, reduced_data, n_neighbors=5):
    n_samples = original_data.shape[0]
    n_neighbors = max(1, min(n_neighbors, n_samples // 2 - 1))
    return trustworthiness(original_data, reduced_data, n_neighbors=n_neighbors)


def evaluate_continuity(original_data, reduced_data, n_neighbors=5):
    n_samples = original_data.shape[0]
    n_neighbors = max(1, min(n_neighbors, n_samples // 2 - 1))

    nbrs_original = NearestNeighbors(
        n_neighbors=n_neighbors).fit(original_data)
    nbrs_reduced = NearestNeighbors(n_neighbors=n_neighbors).fit(reduced_data)

    indices_original = nbrs_original.kneighbors(return_distance=False)
    indices_reduced = nbrs_reduced.kneighbors(return_distance=False)

    continuity_sum = 0.0

    for i in range(n_samples):
        intersect = len(set(indices_original[i]) & set(indices_reduced[i]))
        continuity_sum += intersect / n_neighbors

    continuity_score = continuity_sum / n_samples
    return continuity_score


def evaluate_mse(original_data, reconstructed_data):
    return mean_squared_error(original_data, reconstructed_data)


def plot_loss_landscape(weights_reduced, method_name, net, data_loader, criterion, autoencoder=None, umap_reducer=None, pca=None):
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
            net.conv1.weight.data.shape)
        net.conv1.weight.data = torch.tensor(
            projected_weight, dtype=torch.float32)

        batch_losses = []
        for data in data_loader:
            img, target = data
            img, target = img.to(DEVICE), target.to(DEVICE)
            outputs = net(img)
            loss = criterion(outputs, target).item()
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


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent


def validate(model, dataloader, criterion):
    model.eval()
    losses = []
    accuracies = []
    with torch.no_grad():
        for img, target in dataloader:
            img, target = img.to(DEVICE), target.to(DEVICE)
            outputs = model(img)
            loss = criterion(outputs, target)
            losses.append(loss.item())
            accuracy = (outputs.argmax(dim=1) == target).float().mean().item()
            accuracies.append(accuracy)
    return np.mean(losses), np.mean(accuracies)


@hydra.main(config_path="config", config_name="training_conf.yaml", version_base="1.1")
def train(config) -> None:
    """Train a model on MNIST."""
    log.info("Training day and night")
    hparams = config.training
    print("hparams")
    model_cfg = OmegaConf.load(
        "../../../mlops_project/config/model_conf.yaml")
    print("model_cfg", model_cfg)
    hparams_models = model_cfg.model

    model = MyAwesomeModel(hparams_models).to(DEVICE)
    train_img_path = to_absolute_path("data/processed/train_images.pt")
    train_target_path = to_absolute_path("data/processed/train_target.pt")
    test_img_path = to_absolute_path("data/processed/test_images.pt")
    test_target_path = to_absolute_path("data/processed/test_target.pt")

    train_img_path = to_absolute_path("data/processed/train_images.pt")

    if not os.path.exists(train_img_path) or not os.path.exists(train_target_path) or not os.path.exists(test_img_path) or not os.path.exists(test_target_path):
        print("Error: Data files not found.")
        return

    train_img = torch.load(train_img_path)
    train_target = torch.load(train_target_path)
    test_img = torch.load(test_img_path)
    test_target = torch.load(test_target_path)
    train_set = torch.utils.data.TensorDataset(train_img, train_target)
    test_set = torch.utils.data.TensorDataset(test_img, test_target)

    train_dataloader = torch.utils.data.DataLoader(
        train_set, hparams["batch_size"], shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(
        test_set, hparams["batch_size"], shuffle=False)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=hparams["lr"])

    statistics = {"train_loss": [], "train_accuracy": [],
                  "test_loss": [], "test_accuracy": []}
    normalized_weights = []

    for epoch in range(hparams["epochs"]):
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
                log.info(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

        # Validation step
        test_loss, test_accuracy = validate(model, test_dataloader, loss_fn)
        statistics["test_loss"].append(test_loss)
        statistics["test_accuracy"].append(test_accuracy)

        # Capture weights
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

    # # Apply t-SNE
    # tsne = TSNE(n_components=2, random_state=42)
    # weights_tsne = tsne.fit_transform(normalized_weights)
    # trust_tsne = evaluate_trustworthiness(normalized_weights, weights_tsne)
    # continuity_tsne = evaluate_continuity(normalized_weights, weights_tsne)
    # print(f'Trustworthiness of t-SNE projection: {trust_tsne:.2f}')
    # print(f'Continuity of t-SNE projection: {continuity_tsne:.2f}')

    # # Apply UMAP
    # n_neighbors = max(1, len(normalized_weights) - 1)
    # umap_reducer = umap.UMAP(
    #     n_components=2, n_neighbors=n_neighbors, random_state=42)
    # weights_umap = umap_reducer.fit_transform(normalized_weights)
    # trust_umap = evaluate_trustworthiness(normalized_weights, weights_umap)
    # continuity_umap = evaluate_continuity(normalized_weights, weights_umap)
    # print(f'Trustworthiness of UMAP projection: {trust_umap:.2f}')
    # print(f'Continuity of UMAP projection: {continuity_umap:.2f}')
    # # Apply UMAP
    # n_neighbors = max(1, min(15, len(normalized_weights) - 1))
    # umap_reducer = umap.UMAP(
    #     n_components=2, n_neighbors=n_neighbors, random_state=42)
    # weights_umap = umap_reducer.fit_transform(normalized_weights)
    # trust_umap = evaluate_trustworthiness(normalized_weights, weights_umap)
    # continuity_umap = evaluate_continuity(normalized_weights, weights_umap)
    # print(f'Trustworthiness of UMAP projection: {trust_umap:.2f}')
    # print(f'Continuity of UMAP projection: {continuity_umap:.2f}')

    # Initialize and train the autoencoder
    input_dim = normalized_weights.shape[1]
    latent_dim = 2
    autoencoder = AutoEncoder(input_dim, latent_dim).to(DEVICE)
    optimizer_ae = optim.Adam(autoencoder.parameters(), lr=hparams["lr"])
    weights_tensor = torch.tensor(normalized_weights, dtype=torch.float32)

    num_epochs = 100
    for epoch in range(num_epochs):
        autoencoder.train()
        optimizer_ae.zero_grad()
        reconstructed, latent = autoencoder(weights_tensor)
        loss_ae = evaluate_mse(weights_tensor, reconstructed)
        loss_ae.backward()
        optimizer_ae.step()

    autoencoder.eval()
    with torch.no_grad():
        reconstructed_ae, weights_ae = autoencoder(weights_tensor)

    weights_ae = weights_ae.numpy()
    reconstructed_ae = reconstructed_ae.numpy()

    # trust_ae = evaluate_trustworthiness(normalized_weights, weights_ae)
    # continuity_ae = evaluate_continuity(normalized_weights, weights_ae)
    # mse_ae = evaluate_mse(normalized_weights, reconstructed_ae)
    # print(f'Trustworthiness of Autoencoder projection: {trust_ae:.2f}')
    # print(f'Continuity of Autoencoder projection: {continuity_ae:.2f}')
    # print(f'MSE of Autoencoder projection: {mse_ae:.4f}')

    plot_loss_landscape(weights_pca, "PCA", model,
                        train_dataloader, loss_fn, pca=pca)
    # plot_loss_landscape(weights_tsne, "t-SNE", model,
    # #                     train_dataloader, loss_fn)
    # plot_loss_landscape(weights_umap, "UMAP", model,
    #                     train_dataloader, loss_fn, umap_reducer=umap_reducer)
    # plot_loss_landscape(weights_ae, "Autoencoder", model,
    #                     train_dataloader, loss_fn, autoencoder=autoencoder)

    log.info("Training complete")
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), "models/model.pth")

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs[0, 0].plot(statistics["train_loss"], label="Train Loss")
    axs[0, 0].set_title("Train loss")
    axs[0, 1].plot(statistics["train_accuracy"], label="Train Accuracy")
    axs[0, 1].set_title("Train accuracy")
    axs[1, 0].plot(statistics["test_loss"], label="Test Loss")
    axs[1, 0].set_title("Test loss")
    axs[1, 1].plot(statistics["test_accuracy"], label="Test Accuracy")
    axs[1, 1].set_title("Test accuracy")
    fig.savefig("training_statistics.png")

    log.info("Training complete")
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), "models/model.pth")
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs[0, 0].plot(statistics["train_loss"], label="Train Loss")
    axs[0, 0].set_title("Train loss")
    axs[0, 1].plot(statistics["train_accuracy"], label="Train Accuracy")
    axs[0, 1].set_title("Train accuracy")
    axs[1, 0].plot(statistics["test_loss"], label="Test Loss")
    axs[1, 0].set_title("Test loss")
    axs[1, 1].plot(statistics["test_accuracy"], label="Test Accuracy")
    axs[1, 1].set_title("Test accuracy")
    fig.savefig("training_statistics.png")

    # Log predictions with images in a table
    predictions_t = torch.cat([model(img.to(DEVICE)).argmax(
        dim=1).cpu() for img, _ in train_dataloader])
    # my_table = wandb.Table()
    # my_table.add_column("image", train_img.cpu())
    # my_table.add_column("label", train_target.cpu())
    # my_table.add_column("class_prediction", predictions_t)
    # wandb.log({"mnist_predictions": my_table})


if __name__ == "__main__":
    train()
