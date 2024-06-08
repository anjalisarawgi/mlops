# tests/test_model.py

import torch
# Adjust import based on your actual module
from your_model_module import YourModel


def test_model_output_shape():
    model = YourModel()
    input_tensor = torch.randn(1, 1, 28, 28)  # Example input
    output = model(input_tensor)
    assert output.shape == (
        1, 10), "Model output did not have the correct shape"
