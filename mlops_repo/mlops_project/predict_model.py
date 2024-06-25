# import torch
# # import MyAwesomeModel
# from mlops_project.models.lightning_model import LightningModel

# def predict(
#     model: torch.nn.Module,
#     dataloader: torch.utils.data.DataLoader
# ) -> None:
#     """Run prediction for a given model and dataloader.

#     Args:
#         model: model to use for prediction
#         dataloader: dataloader with batches

#     Returns
#         Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

#     """
#     model.load_state_dict(torch.load("models/model.pth"))
#     model.eval()
#     # values = torch.cat([model(batch) for batch in dataloader], 0)
#     # print("values", values.shape)
#     return torch.cat([model(batch) for batch in dataloader], 0)


# # dataloader
# test_img = torch.load("data/processed/test_images.pt")
# dataloader = torch.utils.data.DataLoader(test_img, batch_size=32)

# if __name__ == "__main__":
#     predict(model=LightningModel(),
#             dataloader=dataloader)
import torch
import torch.nn.functional as F
from mlops_project.models.lightning_model import LightningModel

def predict(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module
) -> torch.Tensor:
    """Run prediction for a given model and dataloader and compute loss.

    Args:
        model: model to use for prediction
        dataloader: dataloader with batches
        criterion: loss function to use

    Returns:
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model
    """
    model.load_state_dict(torch.load("models/model.pth"))
    model.eval()
    
    all_predictions = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():  # Disable gradient calculation for inference
        for data, labels in dataloader:
            outputs = model(data)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            all_predictions.append(outputs)
            all_labels.append(labels)
    
    predictions = torch.cat(all_predictions, dim=0)
    labels = torch.cat(all_labels, dim=0)
    avg_loss = total_loss / len(dataloader)
    
    return predictions, avg_loss

# dataloader
test_img = torch.load("data/processed/test_images.pt")
test_labels = torch.load("data/processed/test_target.pt")

test_dataset = torch.utils.data.TensorDataset(test_img, test_labels)
dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

if __name__ == "__main__":
    model = LightningModel()
    criterion = torch.nn.CrossEntropyLoss()  # Replace with the appropriate loss function
    predictions, avg_loss = predict(model=model, dataloader=dataloader, criterion=criterion)
    
    # Print the average loss
    print("Average Loss:", avg_loss)
    
    # Optionally, print the first few predictions and their corresponding labels
    print("First few predictions:", predictions)
    print("Corresponding labels:", test_labels)
    
    # Save predictions to a file if needed
    torch.save(predictions, "data/processed/predictions.pt")
    print("Predictions saved to data/processed/predictions.pt")
