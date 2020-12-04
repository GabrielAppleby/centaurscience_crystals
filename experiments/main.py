import pathlib
from typing import Tuple

import torch
import numpy as np
from torch_geometric.data import DataLoader, Dataset

from data.utils import MetaDataset, load_data
from modeling.models import BasicGCN

CURRENT_DIR = pathlib.Path(__file__).parent
SAVED_MODEL_PATH = pathlib.Path(CURRENT_DIR, 'basic_gcn.pt')
SAVED_EMBEDDINGS_PATH = pathlib.Path(CURRENT_DIR, 'gcn_embeddings.npz')

RANDOM_SEED: int = 42


def set_random_seeds(random_seed: int) -> None:
    """
    Set the random seed for any libraries used.
    :param random_seed: The random seed to set.
    :return: None.
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)


def train(model: torch.nn.Module,
          training_data: Dataset,
          device: torch.device,
          num_epochs: int = 20) -> torch.nn.Module:
    """
    Train the model for num_epochs epochs on the given device using the given data.

    Right now a lot of stuff is hardcoded for this specific model / dataset.
    Most importantly only the first column of the y target matrix is used.

    :param model: The model to train.
    :param training_data: The training data to use.
    :param device: The device to train on.
    :param num_epochs: The number of epochs to train for.
    :return: The trained model.
    """
    train_loader = DataLoader(training_data, batch_size=32, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_func = torch.nn.MSELoss()

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for data in train_loader:
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)
            batch = data.batch.to(device)

            # This dataset is for multitask learning, but
            # lets just stick to formation energy for now.
            targets = data.y[:, 0]
            targets = torch.reshape(targets, (-1, 1)).to(device)

            preds, _ = model(x, edge_index, batch)
            loss = loss_func(preds, targets)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
        print('{epoch} loss: {loss}'.format(epoch=epoch, loss=(epoch_loss / len(train_loader))))
    return model


def create_embeddings(model: torch.nn.Module,
                      dataset: Dataset,
                      device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates an embedding from data by forward pass through a trained model that returns an embedding
    as its second output.
    :param model: The trained model that ouputs embeddings.
    :param dataset: The dataset to embed.
    :param device: The device to do the forward pass on.
    :return: The embeddings and their corresponding cids.
    """
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    all_embeddings = []
    all_cids = []
    for data in data_loader:
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        batch = data.batch.to(device)

        _, embeddings = model(x, edge_index, batch)
        all_embeddings.append(embeddings.detach().cpu().numpy())
        all_cids.append([int(x) for x in data.id])
    return np.concatenate(all_embeddings), np.concatenate(all_cids)


def main():
    set_random_seeds(RANDOM_SEED)
    dataset: MetaDataset = load_data()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BasicGCN(dataset.num_node_features,
                     num_outputs=1,
                     num_hidden_neurons=64).to(device)

    model = train(model, dataset.train, device)
    torch.save(model, SAVED_MODEL_PATH)

    # model: BasicGCN = torch.load(SAVED_MODEL_PATH)
    # embeddings, cids = create_embeddings(model, dataset.train, device)
    # np.savez(SAVED_EMBEDDINGS_PATH, embedding=embeddings, cids=cids)







if __name__ == '__main__':
    main()
