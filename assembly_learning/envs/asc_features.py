import numpy as np
import torch

import torch.nn.functional as F

from assembly_learning.utils import PNU
from assembly_learning.envs import AssemblyCloudBase

class AssemblyCloudFeatures(AssemblyCloudBase):

    def __init__(self, **kwargs):
        super(AssemblyCloudFeatures, self).__init__(**kwargs)
        self._nf_size = 256

        self.pointnet2 = PNU.get_pointnet2_fe("pointnet2.pt")
        self.part_feature_dict = self._get_part_feature_dict()

    def _get_part_feature_dict(self):
        part_feature_dict = {}
        part_cloud_dict = self._cloud._get_object_cloud_dict()
        for name in self._object_name_list:
            cloud = part_cloud_dict[name]
            part_feature_dict[name] = PNU.pointcloud2feature(self.pointnet2, cloud)
        return part_feature_dict

    def _reset_graph_data(self):
        node_features = F.one_hot(torch.arange(0, self.n_objects + self.n_conn_sites*self.n_objects, 1), num_classes=self._nf_size)
        node_features = node_features.float()
        assert node_features.size(1) == self._nf_size
        node_features = node_features.numpy()
        for i, feature in enumerate(self.part_feature_dict.values()):
            node_features[i * (1+ self.n_conn_sites)] = feature 
        self.node_features = node_features
        self.adjacency_mat = np.zeros(((1+self.n_conn_sites)*self.n_objects, 
                                  (1+self.n_conn_sites)*self.n_objects), 
                                  dtype=np.uint8)
        for i in range(self.n_objects):
            for j in range(self.n_conn_sites):
                cs_id = self.n_objects + i * self.n_conn_sites + j
                self.adjacency_mat[i, cs_id] = 1
                self.adjacency_mat[cs_id, i] = 1