from mlops_project.pytest.train_model_pytest import train
import pytest


def test_training():
    print("works?")
    # train()
    initial_accuracy = 0.0
    final_accuracy = train(epochs=1)
    assert final_accuracy > initial_accuracy, "Training did not improve model accuracy"


def test_invalid_input():
    with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
        train()
