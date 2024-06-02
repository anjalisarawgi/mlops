import pytorch_lightning as pl
import torch
from torch import nn
from models.lightning_model import LightningModel
# data loader
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import wandb
train_img = torch.load("data/processed/train_images.pt")
train_target = torch.load("data/processed/train_target.pt")
train_set = torch.utils.data.TensorDataset(train_img, train_target)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

# call validation data as 30 percent of the training data
val_size = int(0.3 * len(train_set))
validation_set = torch.utils.data.Subset(train_set, range(val_size))

test_img = torch.load("data/processed/test_images.pt")
test_target = torch.load("data/processed/test_target.pt")
test_set = torch.utils.data.TensorDataset(test_img, test_target)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
# callbacks

early_stopping_callback = EarlyStopping(
    monitor="val_loss", patience=3, verbose=True, mode="min"
)
checkpoint_callback = ModelCheckpoint(
    dirpath="./models", monitor="val_loss", mode="min"
)

wandb_logger = pl.loggers.WandbLogger(project="mlops_s4", log_model=True)
# trainer
trainer = pl.Trainer(
    accelerator="gpu",
    max_epochs=10, default_root_dir="lightning_logs",
    limit_train_batches=0.2,
    callbacks=[early_stopping_callback, checkpoint_callback],
    logger=wandb_logger
)
model = LightningModel()
trainer.fit(model, train_loader, test_loader)
trainer.test(model, test_loader)
