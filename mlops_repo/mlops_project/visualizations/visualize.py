import click
import matplotlib.pyplot as plt
import torch
from mlops_project.models.model import MyAwesomeModel
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


@click.command()
@click.option("--model_checkpoint", default="models/model.pth", help="Path to model checkpoint")
@click.option("--processed_dir", default="data/processed", help="Path to processed data directory")
@click.option("--figure_dir", default="reports/figures", help="Path to save figures")
@click.option("--figure_name", default="embeddings.png", help="Name of the figure")
def visualize(model_checkpoint: str,
              processed_dir: str,
              figure_dir: str,
              figure_name: str
              ):
    # loads the pretrained model using a checkpoint file
    # print("myawesome model", MyAwesomeModel())
    print("model_checkpoint", model_checkpoint)
    model = MyAwesomeModel()
    model.load_state_dict(torch.load(model_checkpoint))
    model.eval()
    model.fc = torch.nn.Identity()  # for feature extraction
    # torch.load (disk -> memory)
    test_images = torch.load(f"{processed_dir}/test_images.pt")
    test_labels = torch.load(f"{processed_dir}/test_target.pt")
    # to create a dataset from tensors which is already loaded in memory as tensors
    test_dataset = torch.utils.data.TensorDataset(test_images, test_labels)

    # embeddings, targets = [], []
    # dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
    # with torch.inference_mode():  # inference_mode() disabled gradient calculation
    #     for batch in dataloader:
    #         images, target = batch
    #         predictions = model(images)
    #         embeddings.append(predictions)
    #         targets.append(target)
    #     # print("embeddings", embeddings[1])
    #     # concatenates list of embeddings tensors into a single tensor and converts it into a numpy array
    #     embeddings = embeddings.numpy()
    #     embeddings = torch.cat(embeddings)
    #     # torch.cat(embeddings)  # .numpy()
    #     targets = torch.cat(target.numpy())

    embeddings, targets = [], []
    with torch.inference_mode():
        for batch in torch.utils.data.DataLoader(test_dataset, batch_size=32):
            images, target = batch
            predictions = model(images)
            embeddings.append(predictions)
            targets.append(target)
        embeddings = torch.cat(embeddings).numpy()
        targets = torch.cat(targets).numpy()

        if embeddings.shape[1] > 500:
            pca = PCA(n_components=100)
            # fits and transforms i.e. finds the principal components and does dimensionality reduction of the data
            embeddings = pca.fit_transform(embeddings)
        tsne = TSNE(n_components=2)
        embeddings = tsne.fit_transform(embeddings)

        plt.figure(figsize=(10, 10))
        for i in range(10):  # 10 is for 10 classes
            mask = targets == i
            print("mask", mask)
            print("embeddings.shape", embeddings.shape)
            print("embeddings[mask, 0]", embeddings[mask, 0])
            plt.scatter(embeddings[mask, 0], embeddings[mask, 1], label=str(i))
        plt.legend()
        plt.savefig(f"{figure_dir}/{figure_name}")


if __name__ == "__main__":
    visualize()
