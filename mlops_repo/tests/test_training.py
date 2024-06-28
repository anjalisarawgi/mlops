from mlops_project.pytest.train_model_pytest import train
import pytest


def test_training():
    accuracy = train(epochs=1)
    assert accuracy > 0  # try 1 for making it fail!
    print("Doone")
