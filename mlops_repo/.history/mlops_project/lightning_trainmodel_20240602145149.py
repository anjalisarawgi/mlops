import pytorch_lightning as pl
import torch
from torch import nn

# data loader
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

train_img = torch.load("data/processed/train_images.pt")
train_target = torch.load("data/processed/train_target.pt")
train_set = torch.utils.data.TensorDataset(train_img, train_target)
print("train_set", len(train_set))
