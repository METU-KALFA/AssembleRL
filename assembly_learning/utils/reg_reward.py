import numpy as np
import open3d as o3d

import copy

from assembly_learning.utils import FurnitureCloud
import time

# contains code from http://www.open3d.org/docs/release/tutorial/pipelines
class IcpReward(FurnitureCloud):
    '''
    Defines the icp reward class, inherits registration reward class.
    Containes __call__ method for appliying icp algorithm and compute
    rmse between pointclouds as reward value.
    IcpReward(furniture_id, point=False) uses point-to-plane icp.
    otherwise uses point-to-point icp. (default)
    '''
    def __init__(self, furniture_id, num_samples=10000, threshold=5, max_iterations=2000, point=True):
        super(IcpReward, self).__init__(furniture_id, num_samples)
        self.target_cloud = self.get_target_pointcloud()

        self._threshold = threshold
        self._max_iterations = max_iterations
        self._point = point
        if self._point:
            self.estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint
        else:
            self.estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPlane

    def refine_registration(self, source, target, result):
        # function taken from http://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html
        distance_threshold = self._threshold
        result = o3d.pipelines.registration.registration_icp(
            source, target, distance_threshold,
            estimation_method=self.estimation_method(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=self._max_iterations))
        return result

    def __call__(self, pose_dict=None, source_cloud=None, target_cloud=None, partial_th=None, chamfer=False):
        if source_cloud is None:
            source_cloud = self.get_pointcloud(pose_dict)
        if target_cloud is None:
            target_cloud = self.target_cloud
        result_icp = self.refine_registration(source_cloud, target_cloud, None)

        icp_transformed_source = source_cloud.transform(result_icp.transformation)
        source_target_dist = np.asarray(icp_transformed_source.compute_point_cloud_distance(target_cloud))
        target_source_dist = np.asarray(target_cloud.compute_point_cloud_distance(icp_transformed_source))

        dist = -1
        ind_size = -1

        if chamfer:
            dist = source_target_dist.sum()/source_target_dist.size + target_source_dist.sum()/target_source_dist.size
        if (not partial_th is None) and (partial_th > 0):
            ind = np.where(source_target_dist > partial_th)[0]
            ind_size = ind.size
            dist = (source_target_dist.sum()/source_target_dist.size),(target_source_dist.sum()/target_source_dist.size) # s2t, t2s

        return dist, ind_size, result_icp

