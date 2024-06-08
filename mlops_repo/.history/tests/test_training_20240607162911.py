import pytest
from mlops_project.train_model import train


def test_training():
    initial_accuracy = 0.0
    final_accuracy = train(epochs=1)
    assert final_accuracy > initial_accuracy, "Training did not improve model accuracy"


# def test_invalid_input():
#     with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
#         train(invalid_input=True)
