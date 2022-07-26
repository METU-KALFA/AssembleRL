import gym
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_sparse  import SparseTensor
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_max_pool
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import get_device

class GraphNet(torch.nn.Module):
    def __init__(self, num_node_features, pooling=False): # num_out_features
        super(GraphNet, self).__init__()
        self.conv1 = GCNConv(num_node_features, num_node_features)
        self.conv2 = GCNConv(num_node_features, num_node_features)
        self.conv3 = GCNConv(num_node_features, num_node_features)
        self.pooling = pooling
    
    def forward(self, data):
        # print(data.x, data.adj_t)
        x, adj, batch = data.x, data.adj_t, data.batch
        x = F.relu(self.conv1(x, adj.t()))
        x = F.relu(self.conv2(x, adj.t()))
        x = self.conv3(x, adj.t())
        if self.pooling:
            x = global_max_pool(x, batch)
        return x

class GraphFeatureExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256, pooling=False):
        super(GraphFeatureExtractor, self).__init__(observation_space, features_dim)
        node_features = observation_space["node_features"]
        adjacency_mat = observation_space["edges"]
        self._n = adjacency_mat.shape[0]
        self.num_node_features = node_features.shape[1]
        self.graph_net = GraphNet(self.num_node_features, pooling).to(device=get_device("auto"))
        self.flatten = torch.nn.Flatten()
        self.pooling = pooling

    def forward(self, observations) -> torch.Tensor:
        node_features = observations["node_features"]
        adjacency_mat = observations["edges"]
        data_list = []
        for k in range(node_features.shape[0]):
            adj = SparseTensor.from_dense(adjacency_mat[k])
            data_list.append(Data(x=node_features[k], adj_t=adj))

        batch = Batch.from_data_list(data_list)
        out = self.graph_net(batch)
        if self.pooling:
            out = torch.reshape(out, (-1, self.num_node_features))
        else:    
            out = torch.reshape(out, (-1, self._n, self.num_node_features))
        out = self.flatten(out)
        return out
