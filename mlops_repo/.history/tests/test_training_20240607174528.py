import pytest
from mlops_project.pytest.train_model_pytest import train
from hydra.experimental import initialize, compose


def test_training():
    train()
    # initial_accuracy = 0.0
    # final_accuracy = train(hydra_cfg, epochs=1)
    # assert final_accuracy > initial_accuracy, "Training did not improve model accuracy"


# def test_invalid_input():
#     with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
#         train(invalid_input=True)
