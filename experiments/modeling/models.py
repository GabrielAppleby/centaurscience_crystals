import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool


class BasicGCN(torch.nn.Module):
    """
    A basic GCN.

    Consists of:

    Some GCNConv layers (see https://tkipf.github.io/graph-convolutional-networks/ for
    a good explanation.

    A global mean pooling layer.

    And then some linear layers.

    """
    def __init__(self,
                 num_node_features,
                 num_outputs,
                 num_hidden_neurons):
        """
        Construct the Basic GCN.
        :param num_node_features: The number of node features.
        :param num_outputs: The number of outputs.
        :param num_hidden_neurons: The number of neurons to use in all the intermediate layers. This
        is very simplistic but works fine for now.
        """
        super(BasicGCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, num_hidden_neurons)
        self.conv2 = GCNConv(num_hidden_neurons, num_hidden_neurons)
        self.conv3 = GCNConv(num_hidden_neurons, num_hidden_neurons)

        self.lin1 = Linear(num_hidden_neurons, num_hidden_neurons)
        self.lin2 = Linear(num_hidden_neurons, num_outputs)

    def forward(self, x, edge_index, batch, edge_weight=None):
        """
        Perform the forward pass through all of the layers. Relu is used after each layer, and
        global mean pooling is used between conv and linear layers. Finally some dropout is applied
        in between the final hidden layer and the output layer.
        :param x: The batch of node features.
        :param edge_index: The batch of edge indices
        :param batch: The tensor describing which nodes belong to which graph within the batch.
        :param edge_weight: Edge weightings, not really used at the moment.
        :return: The outputs, as well as the embeddings (last hidden layer).
        """
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = F.relu(self.conv3(x, edge_index, edge_weight))

        x = global_mean_pool(x, batch)
        embeddings = F.relu(self.lin1(x))
        x = F.dropout(embeddings, p=0.5, training=self.training)
        x = self.lin2(x)

        return x, embeddings
