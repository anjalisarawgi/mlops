from mlops_project.models.model import MyAwesomeModel
import torch


def test_model():
    model = LightningModel()
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    assert y.shape == (1, 10)
