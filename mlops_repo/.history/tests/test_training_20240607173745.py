import pytest
from mlops_project.train_model import train
from hydra.experimental import initialize, compose


@pytest.fixture(scope="module")
def hydra_cfg():
    config = {
        "lr": 0.001,
        "batch_size": 32,
        "epochs": 3,
    }
    return config


def test_training(hydra_cfg):
    train(hydra_cfg)
    # initial_accuracy = 0.0
    # final_accuracy = train(hydra_cfg, epochs=1)
    # assert final_accuracy > initial_accuracy, "Training did not improve model accuracy"


# def test_invalid_input():
#     with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
#         train(invalid_input=True)
