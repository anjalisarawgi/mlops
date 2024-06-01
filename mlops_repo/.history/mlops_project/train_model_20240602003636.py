import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import wandb
import torch
from models.model import MyAwesomeModel
from hydra.utils import to_absolute_path
import os
import matplotlib.pyplot as plt

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available(
    ) else "mps" if torch.backends.mps.is_available() else "cpu"
)
print("Device", DEVICE)

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="training_conf.yaml", version_base="1.1")
def train(config: DictConfig) -> None:
    """Train a model on MNIST."""

    log.info("Training day and night")

    hparams = config.training
    model_cfg = OmegaConf.load(to_absolute_path("config/model_conf.yaml"))
    hparams_models = model_cfg.model

    # Initialize Wandb with the config
    wandb.init(project="mlops_project",
               config=OmegaConf.to_container(config, resolve=True))
    model = MyAwesomeModel(hparams_models).to(DEVICE)

    train_img_path = to_absolute_path("data/processed/train_images.pt")
    train_target_path = to_absolute_path("data/processed/train_target.pt")

    print(f"Train image path: {train_img_path}")
    print(f"Train target path: {train_target_path}")

    if not os.path.exists(train_img_path):
        print(f"Error: File not found - {train_img_path}")
        return

    if not os.path.exists(train_target_path):
        print(f"Error: File not found - {train_target_path}")
        return

    print("Loading data")
    try:
        train_img = torch.load(train_img_path)
        train_target = torch.load(train_target_path)
    except FileNotFoundError as e:
        print("Please check the paths and ensure the data files exist.")
        return

    train_set = torch.utils.data.TensorDataset(train_img, train_target)
    print("train_set", len(train_set))

    train_dataloader = torch.utils.data.DataLoader(
        train_set, batch_size=hparams.batch_size)
    print("train_dataloader", len(train_dataloader))
    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.lr)

    statistics = {"train_loss": [], "train_accuracy": []}
    for epoch in range(hparams.epochs):
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
                wandb.log({"loss": loss.item(), "accuracy": accuracy})

    log.info("Training complete")
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), "models/model.pth")
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")

    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")

    fig.savefig("training_statistics.png")


if __name__ == "__main__":
    train()
