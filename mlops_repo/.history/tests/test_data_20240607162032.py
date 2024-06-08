from mlops_project.data_loader import data_loader
import torch
from tests import _PATH_DATA


def test_data():
    train, test = data_loader(_PATH_DATA)
    assert len(train) == 25000
    assert len(test) == 5000

    for dataset in [train, test]:
        for images, lables in dataset:
            assert images.shape == (1, 28, 28)
            assert lables in range(10)

    # check all labels are represented
    train_targets = torch.unique(train.tensors[1])
    test_targets = torch.unique(test.tensors[1])

    assert (train_targets == torch.arange(0, 10)).all()
    assert (test_targets == torch.arange(0, 10)).all()
