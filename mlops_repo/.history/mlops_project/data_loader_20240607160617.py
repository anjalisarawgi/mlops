import torch
from hydra.utils import to_absolute_path


def data_loader(dir_path):
    train_img = torch.load(
        to_absolute_path("data/processed/train_images.pt"))
    train_target = torch.load(
        to_absolute_path("data/processed/train_target.pt"))

    test_img = torch.load(
        to_absolute_path("data/processed/test_images.pt"))
    test_target = torch.load(
        to_absolute_path("data/processed/test_target.pt"))

    # train_img = torch.load(train_img_path)
    # train_target = torch.load(train_target_path)

    train_set = torch.utils.data.TensorDataset(train_img, train_target)
    test_set = torch.utils.data.TensorDataset(test_img, test_target)

    return train_set, test_set
