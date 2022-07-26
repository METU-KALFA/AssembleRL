from collections import OrderedDict

import copy
import numpy as np
import gym
import torch
import matplotlib.pyplot as plt
import networkx as nx

from gym.spaces import Discrete, Box, Dict

import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx

from assembly_learning.utils import FurnitureCloud, IcpReward, T

class AssemblyCloudBase(gym.Env):

    def __init__(self, **kwargs):
        self.action_space_type = kwargs["as_type"]  
        self.reward_type = kwargs["rew_type"] 
        self._main_object_name = kwargs["main_object"] if "partial" in self.reward_type  else None
        print("reward type:", self.reward_type)
        self.num_points = kwargs["pc_sample_size"]  
        self.partial_threshold_points = kwargs["points_threshold"] if "partial_th" in self.reward_type else None
        self.partial_threshold_num = kwargs["num_threshold"] if  "partial_th" in self.reward_type else None

        self._nf_size = 35
        
        self._max_ep_steps = kwargs["max_ep_length"]  
        self.furniture_id = kwargs["furniture_id"]  
        self._cloud = FurnitureCloud(self.furniture_id, num_samples=self.num_points)

        self._object_name_list = self._cloud._get_object_name_list()
         
        self._object_name2index_dict = {}
        for index, object_name in enumerate(self._object_name_list):
            self._object_name2index_dict[object_name] = index

        self.n_objects = len(self._object_name_list)
        self.n_conn_sites = 6
        
        self._conn_site2body_dict = self._cloud._get_site_info_dict()
        self._conn_site_rel_pose_dict = self._cloud._get_site_rel_pose_dict()
        self._cc_obj_pose_dict = self._cloud._get_cc_obj_pose_dict()
        self._conn_site_dict = self._get_conn_site_dict()
        self.icp_reward_point = IcpReward(self.furniture_id, num_samples=self.num_points)
        print("threshold:", self.partial_threshold_num)

        if self.action_space_type == 360:
            self.pair_dict = self._get_pair_dict_360()
        else:
            raise Exception("Wrong action space type")

        self.first_reset = True
        self.first_t2s = None 

    @property
    def observation_space(self):
        ob_space = OrderedDict()    
        ob_space['node_features'] = Box(
            low=-np.inf,
            high=np.inf,
            shape=((1+self.n_conn_sites)*self.n_objects, self._nf_size),
        )
        ob_space['edges'] = Box(
            low=0,
            high=1,
            dtype=np.uint8,
            shape=((1+self.n_conn_sites)*self.n_objects,(1+self.n_conn_sites)*self.n_objects),
        )
        return Dict(ob_space)

    @property
    def dof(self):
        n = self.n_objects
        if self.action_space_type == 10:
            dof = int(0.5*n*(n-1))
        elif self.action_space_type == 360:
            dof = int(0.5*n*(n-1)*6*6) 
        return dof

    @property
    def action_space(self):
        return Discrete(self.dof)

    def _update_graph_data(self, obj1_name, obj2_name, conn_site1, conn_site2):
        obj1_id = self._object_name2index_dict[obj1_name]
        obj2_id = self._object_name2index_dict[obj2_name]
        id1 = self.n_objects + obj1_id * self.n_conn_sites + conn_site1
        id2 = self.n_objects + obj2_id * self.n_conn_sites + conn_site2
        self.adjacency_mat[id1, id2] = 1
        self.adjacency_mat[id2, id1] = 1

    def _place_objects(self):       
        _pos_init = [np.array([ 0.137, -0.199,  0.133     ]), 
                          np.array([0.035, 0.103, 0.133     ]), 
                          np.array([0.336, 0.079, 0.133     ]), 
                          np.array([-0.075, -0.159,  0.133     ]), 
                          np.array([-0.265,  0.178,  0.        ]),
                          np.array([-0.465,  0.378,  0.133]),
                          np.array([-0.565,  -0.278,  -0.233]),
                          np.array([1.,  0.,  0.]),
                          np.array([0.240,  -0.578,  -0.333]),
                          np.array([-0.465,  0.378,  0.133]),
                          np.array([0.165,  0.78,  0.133])]
        _quat_init = [[0.995, 0.0434, 0.0783, -0.0034], 
                           [0.996, 0.0692, 0.0123, 0.0515], 
                           [0.998, 0.0250, 0.0272, 0.0429], 
                           [0.992, 0.0732, 0.0668, 0.0732], 
                           [0.999, 0.0167, 0.0266, 0.02576],
                           [0.999, 0.0167, 0.0266, 0.02576],
                           [0.998, 0.0250, 0.0272, 0.0429],
                           [0., 1., 0., 0],
                           [0., 0., 1., 0],
                           [0., 0., 0., 1.],
                           [0.5, 0.5, 0.5, 0.5]]
        return _pos_init, _quat_init

    def step(self, a):
        self._num_step += 1
        change = self._step_pc(a)
        reward, done, info = self._compute_reward(change)
        ob = self._get_obs()
        info["change"] = change
        return ob, reward, done, info

    def _compute_part_pose_transform(self, target_obj_name, source_obj_name, target_conn_site, source_conn_site):
        conn_site_dict = self._conn_site_dict
        target_mat = self._pose_dict[target_obj_name]["pose"]
        target_rel_mat = conn_site_dict[target_obj_name][target_conn_site]
        target_abs_mat = np.matmul(target_mat, target_rel_mat)
        source_mat = self._pose_dict[source_obj_name]["pose"]
        source_rel_mat = copy.deepcopy(conn_site_dict[source_obj_name][source_conn_site] )
        source_abs_mat = np.matmul(source_mat, source_rel_mat)
        source_abs_inv = np.linalg.inv(source_abs_mat)
        transform = np.matmul(target_abs_mat, source_abs_inv) 
        return transform

    def _step_pc(self, a):
        target_obj_name, source_obj_name, target_conn_site, source_conn_site = self.pair_dict[int(a)]
        if source_obj_name in self._connected_dict[target_obj_name]: return "no_change"
        if not target_conn_site in self._empty_cs_dict[target_obj_name]: return "no_change"
        if not source_conn_site in self._empty_cs_dict[source_obj_name]: return "no_change"
        self._update_graph_data(target_obj_name, source_obj_name, target_conn_site, source_conn_site)
        source_set = self._connected_dict[source_obj_name]
        target_set = self._connected_dict[target_obj_name]
        union_set = target_set.union(source_set)
        if "partial" in self.reward_type and  not self._main_object_name in union_set:
            return "wrong_object"
        transform = self._compute_part_pose_transform(target_obj_name, source_obj_name, target_conn_site, source_conn_site)
        for name in self._connected_dict[source_obj_name]:
            pose = copy.deepcopy(self._pose_dict[name]["pose"])
            self._pose_dict[name]["pose"] = np.matmul(transform, pose)
        for name in union_set:
            self._connected_dict[name] = union_set
        self._empty_cs_dict[target_obj_name].remove(target_conn_site)
        self._empty_cs_dict[source_obj_name].remove(source_conn_site)
        return "correct_object"

    def _render_graph(self):
        edge_index1 = []
        edge_index2 = []
        for i in range(self.adjacency_mat.shape[0]):
            for j in range(self.adjacency_mat.shape[1]):
                if self.adjacency_mat[i,j] == 1:
                    edge_index1.append(i)
                    edge_index2.append(j)
        edge_index = torch.tensor([edge_index1,edge_index2], dtype=torch.long)
        graph_data = Data(torch.tensor(self.node_features), edge_index=edge_index)
        g = to_networkx(graph_data)
        plt.figure(1,figsize=(14,14))
        node_colors = ["red"] * self.n_objects + ["blue"] * self.n_objects * self.n_conn_sites
        nx.draw_networkx(g, with_labels = False , node_color = node_colors)
        plt.show()


    def render(self, mode="cloud"):
        if mode in ["cloud", "human"]:
            self._cloud._draw_pointcloud(self._pose_dict, part_based=True)
        elif mode == "graph":
            self._render_graph()
        elif mode == "partial":
            self._cloud._draw_pointcloud(self._get_partial_pose_dict(), part_based=True)
            

    def _get_graph_ob(self):
        graph_ob = OrderedDict()
        graph_ob["node_features"] = copy.deepcopy(self.node_features)
        graph_ob["edges"] = copy.deepcopy(self.adjacency_mat)
        return graph_ob

    def _reset_graph_data(self):
        node_features = F.one_hot(torch.arange(0, self.n_objects + self.n_conn_sites*self.n_objects, 1), num_classes=self._nf_size)
        node_features = node_features.float()
        assert node_features.size(1) == self._nf_size
        node_features = node_features.numpy()
        self.node_features = node_features
        self.adjacency_mat = np.zeros(((1+self.n_conn_sites)*self.n_objects, 
                                  (1+self.n_conn_sites)*self.n_objects), 
                                  dtype=np.uint8)
        for i in range(self.n_objects):
            for j in range(self.n_conn_sites):
                cs_id = self.n_objects + i * self.n_conn_sites + j
                self.adjacency_mat[i, cs_id] = 1
                self.adjacency_mat[cs_id, i] = 1

    def _get_pair_dict_360(self):
        n = self.n_objects
        pair_dict = {}
        k = 0
        for i in range(n):
            for j in range(i+1, n):
                for t1 in range(self.n_conn_sites):
                    for t2 in range(self.n_conn_sites):
                        pair_dict[k] = [self._object_name_list[i], self._object_name_list[j], t1, t2]
                        k += 1
        return pair_dict    

    def _get_initial_pose_dict(self):
        initial_pose_dict = {}
        initial_pos_list, initial_quat_list = self._place_objects()
        for i, name in enumerate(self._object_name_list):
            pos = initial_pos_list[i]
            rot_mat = T.quat2mat(T.convert_quat(np.array(initial_quat_list[i])))
            initial_pose_dict[name] = T.make_pose(pos, rot_mat)
        return initial_pose_dict

    def reset(self):
        self._num_step = 0
        self._reset_graph_data()
        self._pose_dict = {}
        self._connected_dict = {}
        self._empty_cs_dict = {}
        self._initial_pose_dict = self._get_initial_pose_dict()
        for i,name in enumerate(self._object_name_list):
            self._pose_dict[name] = {}
            self._empty_cs_dict[name] = list(range(self.n_conn_sites))
            self._connected_dict[name] = set()
            self._connected_dict[name].add(name)
            self._pose_dict[name]["pose"] = self._initial_pose_dict[name]
            self._pose_dict[name]["conn_site"] = self._conn_site_dict[name].values()


        if self.reward_type in ["dist_change", "early_stop"]:
            self.prev_dist, _ = self.icp_reward_point(self._get_pose_dict())
        elif self.reward_type == "partial_chamfer_change":
            self.prev_dist, _, _ = self.icp_reward_point(self._get_partial_pose_dict(), partial_th=None, chamfer=True) 
            self.prev_dist *= 10 
            self.prev_dist + 0.065 
            self.prev_dist *= 5 
        elif self.reward_type == "partial_th" or self.reward_type == "partial_th_comp":
            if self.first_reset:
                (self.first_s2t, self.first_t2s), _, _ = self.icp_reward_point(self._get_partial_pose_dict(), partial_th=self.partial_threshold_points)
                self.first_reset = False
            self.prev_t2s = self.first_t2s
        else: 
            self.prev_dist = None, None
        return self._get_obs()

    def _get_init_site_pose_mat(self, site_name):
        conn_site_rel_mat = self._conn_site_rel_pose_dict[site_name]
        obj_name = self._conn_site2body_dict[site_name]
        obj_mat = self._cc_obj_pose_dict[obj_name]
        conn_site_abs_mat = np.matmul(obj_mat, conn_site_rel_mat)
        return conn_site_abs_mat

    def _get_obj_bounding_box(self, obj_name):
        min_pos = np.array([float("inf"), float("inf"), float("inf")])
        max_pos = np.array([-1*float("inf"), -1*float("inf"), -1*float("inf")])
        for site_name in self._conn_site2body_dict.keys():
            if self._conn_site2body_dict[site_name] == obj_name:
                pos = T.mat2pose(self._get_init_site_pose_mat(site_name))[0]
                min_pos = np.minimum(min_pos, pos)
                max_pos = np.maximum(max_pos, pos)
        return min_pos, max_pos

    def _get_conn_site_dict(self):
        conn_site_dict = {}
        for _, obj_name in enumerate(self._object_name_list):
            conn_site_dict[obj_name] = {}
            k = 0
            for _, site_name in enumerate(self._conn_site2body_dict.keys()):
                if 'conn_site' in site_name and self._conn_site2body_dict[site_name] == obj_name:
                    rel_pose = self._conn_site_rel_pose_dict[site_name]
                    conn_site_dict[obj_name][k] = rel_pose
                    k+=1
            print("Number of connsites:", k)
        return conn_site_dict

    def _get_obs(self):
        return self._get_graph_ob()

    def _is_done(self):
        key = self._main_object_name if "partial" in self.reward_type else list(self._connected_dict.keys())[0]
        return (self._num_step == self._max_ep_steps) or (self._connected_dict[key] == set(self._object_name_list))

    def _get_pose_dict(self):
        pose_dict = OrderedDict()
        for _, obj_name in enumerate(self._object_name_list):
            pose_mat = self._pose_dict[obj_name]["pose"]
            pose_dict[obj_name] = {"pose": pose_mat}
        return pose_dict

    def _get_partial_pose_dict(self):
        connected_set = self._connected_dict[self._main_object_name]
        pose_dict = OrderedDict()
        for i, obj_name in enumerate(connected_set):
            pose_mat = self._pose_dict[obj_name]["pose"]
            pose_dict[obj_name] = {"pose": pose_mat} 
        return pose_dict

    def reward_partial_th(self, change):
        done = self._is_done()
        if change == "no_change":
            reward = -1 
            info = {"dist": -1, "reward": -1}
        elif change == "wrong_object":
            return -10, True, {"dist": -1, "reward": -10} 
        elif change == "correct_object":
            (curr_s2t, curr_t2s), th_size, result = self.icp_reward_point(self._get_partial_pose_dict(), partial_th=self.partial_threshold_points)
            if curr_t2s > self.prev_t2s:
                info = {"dist": curr_s2t, "reward": -10, "s2t": curr_s2t, "t2s": curr_t2s, "n_th": th_size, "fs2t": self.first_s2t, "ft2s": self.first_t2s} 
                return -10, True, info 
            else:
                if th_size <= self.partial_threshold_num:
                    reward = 5  
                    for name in self._connected_dict[self._main_object_name]:
                        pose = copy.deepcopy(self._pose_dict[name]["pose"])
                        self._pose_dict[name]["pose"] = np.matmul(result.transformation, pose)
                else:
                    reward = -5
                self.prev_t2s = curr_t2s
                info = {"dist": curr_s2t, "reward": reward, "s2t": curr_s2t, "t2s": curr_t2s, "n_th": th_size, "fs2t": self.first_s2t, "ft2s": self.first_t2s} 
        return reward, done, info
    
    def reward_partial_th_end(self):
        done = self._is_done()
        reward = -1
        info = {"dist": -1, "reward": reward}
        if done:
            (curr_s2t, curr_t2s), th_size, result = self.icp_reward_point(self._get_partial_pose_dict(), 
                                    partial_th=self.partial_threshold_points)
            if th_size <= self.partial_threshold_num and np.abs(curr_s2t-curr_t2s) < 5*10^-3:
                reward = 5  
            else:
                reward = -5
            info = {"dist": curr_s2t, "reward": reward} 
        return reward, done, info
            

    def reward_partial_chamfer(self, change):
        if change == "no_change":
            return -1, self._is_done(), {"dist": -1, "reward": -1}
        elif change == "wrong_object":
            return -10, True, {"dist": -1, "reward": -10}
        else:
            dist, _, _ = self.icp_reward_point(self._get_partial_pose_dict(), partial_th=None, chamfer=True)
            reward = (-10* dist + 0.5) *2           
            return reward, self._is_done(), {"dist": dist, "reward": reward}

    def reward_end_chamfer(self):
        done = self._is_done()
        if done:
            dist, _ = self.icp_reward_point(self._get_pose_dict(), partial_th=None, chamfer=True)
            reward = (-10* dist + 0.5) *2
            dct = {"dist": dist, "reward": reward}
        else:
            reward = -1
            dct = {"dist": -reward, "reward": reward}
        return reward, done, dct
    
    def reward_partial_chamfer_change(self, change):
        if change == "no_change":
            return -1, self._is_done(), {"dist": -1, "reward": -1, "change": change}
        elif change == "wrong_object":
            return -10, True, {"dist": -1, "reward": -10, "change": change}
        else:
            current_dist, _, _ = self.icp_reward_point(self._get_partial_pose_dict(), partial_th=None, chamfer=True)
            current_dist *= 10 
            current_dist + 0.065 
            current_dist *= 5 
            reward = self.prev_dist-current_dist      
            self.prev_dist  = current_dist
            return reward, self._is_done(), {"dist": current_dist, "reward": reward, "change": change}

    def reward_partial_end_chamfer(self, change):
        done = self._is_done()
        if change == "no_change":
            return -0.2, done, {"dist": -1, "reward": -0.2, "change": change}
        elif change == "wrong_object":
            return -2, True, {"dist": -1, "reward": -2, "change": change}
        else:
            if done:
                dist, _ = self.icp_reward_point(self._get_partial_pose_dict(), partial_th=None, chamfer=True)
                reward = (-10* dist + 0.5) *2
                return reward, done, {"dist": dist, "reward": reward, "change": change}
            else:
                return -0.1, done, {"dist": -1, "reward": -0.1, "change": change}

    def reward_only_correctness(self, change):
        done = self._is_done()
        if change == "no_change":
            reward = -1 
            info = {"dist": -1, "reward": -1}
        elif change == "wrong_object":
            return -10, True, {"dist": -1, "reward": -10} 
        elif change == "correct_object":
            (curr_s2t, _), th_size, _ = self.icp_reward_point(self._get_partial_pose_dict(), partial_th=self.partial_threshold_points)
            if th_size <= self.partial_threshold_num:
                reward = 5  
            else:
                reward = -5
            info = {"dist": curr_s2t, "reward": reward} 
        return reward, done, info

    def reward_only_completeness(self, change):
        done = self._is_done()
        if change == "no_change":
            reward = -1 
            info = {"dist": -1, "reward": -1}
        elif change == "wrong_object":
            return -10, True, {"dist": -1, "reward": -10} 
        elif change == "correct_object":
            (curr_s2t, curr_t2s), _, _ = self.icp_reward_point(self._get_partial_pose_dict(), partial_th=self.partial_threshold_points)

            if curr_t2s > self.prev_t2s:
                return -10, True, {"dist": curr_s2t, "reward": -10}
            else:
                reward = 5
                self.prev_t2s = curr_t2s
                info = {"dist": curr_s2t, "reward": reward} 
        return reward, done, info

    def _compute_reward(self, change = "correct_object"):
        if  self.reward_type == "partial_th":
            ret = self.reward_partial_th(change)
        elif self.reward_type == "partial_chamfer":
            ret = self.reward_partial_chamfer(change)
        elif self.reward_type == "end_chamfer":
            ret = self.reward_end_chamfer()
        elif self.reward_type == "partial_chamfer_change":
            ret = self.reward_partial_chamfer_change(change)
        elif self.reward_type == "partial_end_chamfer":
            ret = self.reward_partial_end_chamfer(change)
        elif self.reward_type == "partial_th_end":
            ret = self.reward_partial_th_end()
        elif self.reward_type == "partial_th_corr":
            ret = self.reward_only_correctness(change)
        elif self.reward_type == "partial_th_comp":
            ret = self.reward_only_completeness(change)
        return ret


