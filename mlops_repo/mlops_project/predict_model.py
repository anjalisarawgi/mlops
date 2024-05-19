import torch
from .models import MyAwesomeModel


def predict(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader
) -> None:
    """Run prediction for a given model and dataloader.

    Args:
        model: model to use for prediction
        dataloader: dataloader with batches

    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """
    model.load_state_dict(torch.load("models/model.pth"))
    model.eval()
    # values = torch.cat([model(batch) for batch in dataloader], 0)
    # print("values", values.shape)
    return torch.cat([model(batch) for batch in dataloader], 0)


# dataloader
test_img = torch.load("data/processed/test_images.pt")
dataloader = torch.utils.data.DataLoader(test_img, batch_size=32)

if __name__ == "__main__":
    predict(model=MyAwesomeModel(),
            dataloader=dataloader)
