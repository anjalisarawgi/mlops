# test_suite/test_data.py

import pytest
import os
from tests import _PATH_DATA
from mlops_project.data.make_dataset import make_data


@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_data_loading():
    dataset = make_data(_PATH_DATA)
    assert len(
        dataset) == 50000, "Dataset did not have the correct number of training samples"
    sample, label = dataset[0]
    assert sample.shape == (1, 28, 28), "Sample did not have the correct shape"
    assert label in range(10), "Label is not within the expected range"

    dataset = make_data(_PATH_DATA, train=False)
    assert len(
        dataset) == 10000, "Dataset did not have the correct number of test samples"
