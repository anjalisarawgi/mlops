from mlops_project.data_loader import data_loader
import torch


def test_data():
    train, test = data_loader()
    assert len(train) == 30000
    assert len(test) == 5000
    # for dataset in [train, test]:
    #     for x, y in dataset:
    #         assert x.shape == (1, 28, 28)
    #         assert y in range(10)
    # train_targets = torch.unique(train.tensors[1])
    # assert (train_targets == torch.arange(0, 10)).all()
    # test_targets = torch.unique(test.tensors[1])
    # assert (test_targets == torch.arange(0, 10)).all()
