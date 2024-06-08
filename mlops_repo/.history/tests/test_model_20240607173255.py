# from mlops_project.models.model import MyAwesomeModel
# import torch
# import pytest


# @pytest.fixture
# def hparams_model():
#     # Mocking the Hydra configuration
#     config = {
#         "conv1_out_channels": 32,
#         "conv2_out_channels": 64,
#         "conv3_out_channels": 128,
#         "dropout": 0.5,
#         "fc1_out_features": 10
#     }
#     return config


# def test_model():
#     model = MyAwesomeModel()
#     x = torch.randn(1, 1, 28, 28)
#     y = model(x)
#     assert y.shape == (1, 10)
