# tests/test_data.py

import pytest
import os
from tests import _PATH_DATA
# Adjust import based on your actual module
from models.model import MyAwesomeModel


@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_data_loading():
    dataset = MNIST(train=True)
    assert len(
        dataset) == 50000, "Dataset did not have the correct number of training samples"
    sample, label = dataset[0]
    assert sample.shape == (1, 28, 28), "Sample did not have the correct shape"
    assert label in range(10), "Label is not within the expected range"

    dataset = MNIST(train=False)
    assert len(
        dataset) == 10000, "Dataset did not have the correct number of test samples"
