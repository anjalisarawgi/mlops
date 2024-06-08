from mlops_project.data_loader import data_loader
import torch
from tests import _PATH_DATA


def test_data():
    train, test = data_loader(_PATH_DATA)
    assert len(
        train) == 60000, "Dataset did not have the correct number of training samples"
    assert len(
        test) == 10000, "Dataset did not have the correct number of test samples"
    assert all(img.shape == (28, 28)
               for img in train), "Training images did not have the correct shape"
    assert all(img.shape == (28, 28)
               for img in dataset.test_images), "Test images did not have the correct shape"
    assert set(dataset.train_labels) == set(
        range(10)), "Not all labels are represented in the training set"
    assert set(dataset.test_labels) == set(
        range(10)), "Not all labels are represented in the test set"

    # assert len(train) == 25000
    # assert len(test) == 5000
    # for dataset in [train, test]:
    #     for x, y in dataset:
    #         assert x.shape == (1, 28, 28)
    #         assert y in range(10)
    # train_targets = torch.unique(train.tensors[1])
    # assert (train_targets == torch.arange(0, 10)).all()
    # test_targets = torch.unique(test.tensors[1])
    # assert (test_targets == torch.arange(0, 10)).all()
