import click
import numpy as np
import matplotlib.pyplot as plt

import torch

from models.model import MyAwesomeModel
import random
from sklearn.decomposition import PCA
from matplotlib import cm
from scipy.interpolate import griddata


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print("Device", DEVICE)
def compute_loss_landscape(model, loss_fn, dataloader, device):
    loss_landscape = []
    model.zero_grad()
    for img, target in dataloader:
        img, target = img.to(device), target.to(device)
        output = model(img)
        loss = loss_fn(output, target)
        loss_landscape.append(loss.item())
    return loss_landscape

def perturb_and_compute_loss(model, loss_fn, dataloader, original_params, perturb_range=0.5, grid_size=25):
    print("Perturbing and computing loss")
    params = list(model.parameters())
    loss_surface = np.zeros((grid_size, grid_size))

    for i in range(grid_size):
        for j in range(grid_size):
            # Perturb parameters
            for param, original_param in zip(params, original_params):
                param.data = original_param + (i - grid_size//2) * perturb_range / grid_size * torch.randn_like(original_param) + \
                                            (j - grid_size//2) * perturb_range / grid_size * torch.randn_like(original_param)
            
            # Compute loss
            total_loss = 0
            for img, target in dataloader:
                img, target = img.to(DEVICE), target.to(DEVICE)
                output = model(img)
                loss = loss_fn(output, target)
                total_loss += loss.item()
            loss_surface[i, j] = total_loss / len(dataloader)
            
            # Reset parameters to original
            for param, original_param in zip(params, original_params):
                param.data = original_param
            print(f"Finished iteration {i*grid_size + j + 1}/{grid_size**2}")
    return loss_surface
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
    weight_snapshots = []
    loss_landscapes = []
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
        
            # Save the model weights at each epoch
            weight_snapshot = []
            for param in model.parameters():
                weight_snapshot.append(param.data.cpu().numpy().flatten())
            weight_snapshots.append(np.concatenate(weight_snapshot))

        # Compute loss landscape
        loss_landscape = compute_loss_landscape(model, loss_fn, train_dataloader, DEVICE)
        loss_landscapes.append(loss_landscape)

    print("Training complete")

    torch.save(model.state_dict(), "models/model.pth")
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")

    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")

    fig.savefig("reports/figures/training_statistics.png")

    # Save original parameters
    print("Saving original parameters")
    original_params = [p.clone().detach() for p in model.parameters()]

    # Compute and plot loss landscape
    loss_surface = perturb_and_compute_loss(model, loss_fn, train_dataloader, original_params, perturb_range=0.5, grid_size=25)
    print("Loss surface computed")
    # Plot the loss surface
    X_axis = np.linspace(-0.5, 0.5, 25)
    Y_axis = np.linspace(-0.5, 0.5, 25)
    X_grid, Y_grid = np.meshgrid(X_axis, Y_axis)
    print("X_grid", X_grid.shape)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X_grid, Y_grid, loss_surface, cmap='viridis')

    ax.set_xlabel('Perturbation Direction 1')
    ax.set_ylabel('Perturbation Direction 2')
    ax.set_zlabel('Loss')
    plt.title('Loss Landscape')
    plt.show()

    # Apply PCA on the weight snapshots
    # Visualize loss landscape
    # plt.figure(figsize=(8, 6))
    # for epoch, landscape in enumerate(loss_landscapes):
    #     plt.plot(landscape, label=f"Epoch {epoch+1}")
    # plt.title("Loss Landscape")
    # plt.xlabel("Batch")
    # plt.ylabel("Loss")
    # plt.legend()
    # plt.savefig("reports/figures/loss_landscape.png")
    
    # # Apply PCA on the weight snapshots
    # pca = PCA(n_components=3)  # Change n_components to 3 for 3D visualization
    # pca_result = pca.fit_transform(weight_snapshots)

    # # Visualize PCA in 3D as a surface plot
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')

    # # Define grid points
    # alpha = pca_result[:, 0]
    # beta = pca_result[:, 1]
    # gamma = pca_result[:, 2]

    # # Create meshgrid with appropriate number of points
    # num_points = int(np.ceil(np.sqrt(len(alpha))))
    # alpha_grid, beta_grid = np.meshgrid(np.linspace(min(alpha), max(alpha), num_points),
    #                                     np.linspace(min(beta), max(beta), num_points))

    # # Interpolate gamma to fit the grid
    # from scipy.interpolate import griddata
    # gamma_grid = griddata((alpha, beta), gamma, (alpha_grid, beta_grid), method='cubic')

    # # Plot surface
    # surf = ax.plot_surface(alpha_grid, beta_grid, gamma_grid, cmap=cm.viridis, alpha=0.8)
    # cbar = fig.colorbar(surf, ax=ax)
    # cbar.set_label('Principal Component 3')

    # ax.set_title("PCA Visualization of Weight Space (3D)")
    # ax.set_xlabel("Principal Component 1")
    # ax.set_ylabel("Principal Component 2")
    # ax.set_zlabel("Principal Component 3")

    # plt.show()

    # # Plot training statistics
    # fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # axs[0].plot(statistics["train_loss"])
    # axs[0].set_title("Train loss")
    # axs[0].set_xlabel("Iteration")
    # axs[0].set_ylabel("Loss")

    # axs[1].plot(statistics["train_accuracy"])
    # axs[1].set_title("Train accuracy")
    # axs[1].set_xlabel("Iteration")
    # axs[1].set_ylabel("Accuracy")

    # fig.savefig("reports/figures/training_statistics.png")

    # # Apply PCA on the weight snapshots
    # pca = PCA(n_components=3)  # Change n_components to 3 for 3D visualization
    # pca_result = pca.fit_transform(weight_snapshots)

    # # Visualize PCA in 3D as a surface plot
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')

    # # Define grid points
    # alpha = pca_result[:, 0]
    # beta = pca_result[:, 1]
    # gamma = pca_result[:, 2]

    # # Create meshgrid with appropriate number of points
    # num_points = 100  # Increased number of points for higher resolution
    # alpha_grid, beta_grid = np.meshgrid(np.linspace(min(alpha), max(alpha), num_points),
    #                                     np.linspace(min(beta), max(beta), num_points))

    # # Interpolate gamma to fit the grid
    # gamma_grid = griddata((alpha, beta), gamma, (alpha_grid, beta_grid), method='cubic')

    # # Plot surface
    # surf = ax.plot_surface(alpha_grid, beta_grid, gamma_grid, cmap=cm.plasma, alpha=0.8)
    # cbar = fig.colorbar(surf, ax=ax)
    # cbar.set_label('Principal Component 3')

    # ax.set_title("PCA Visualization of Weight Space (3D)")
    # ax.set_xlabel("Principal Component 1")
    # ax.set_ylabel("Principal Component 2")
    # ax.set_zlabel("Principal Component 3")

    # plt.show()

    # Apply PCA on the weight snapshots

     # Apply PCA on the loss snapshots





cli.add_command(train)
if __name__ == "__main__":
    train()
