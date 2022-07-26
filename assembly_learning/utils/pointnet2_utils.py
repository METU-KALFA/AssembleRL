import numpy as np
import torch

from assembly_learning.utils import Pointnet2FE
from assembly_learning.utils import FurnitureCloud
from torch_geometric.data import Data, Batch
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

np.random.seed(9)
torch.manual_seed(9)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(9)

def get_pointnet2_fe(pointnet2_filename="pointnet2.pt"):
    import os
    file_path = os.path.dirname(os.path.abspath(__file__))
    pointnet2_path = file_path + "/../" + pointnet2_filename
    pointnet2_fe = Pointnet2FE()
    pointnet2_fe.to(device)
    pointnet2_fe.load_state_dict(torch.load(pointnet2_path))
    return pointnet2_fe

def open3d_to_pyg(op3_cloud):
    points = np.asarray(op3_cloud.points)
    return Data(pos = torch.tensor(points, dtype=torch.float32))

def pointcloud2feature(pointnet2_fe, cloud):
    pointnet2_fe.eval()
    with torch.no_grad():
        cloud_data = open3d_to_pyg(cloud)
        cloud_batch = Batch().from_data_list([cloud_data])
        cloud_batch.to(device)
        vector = pointnet2_fe(cloud_batch)
    return vector.cpu().numpy()

def main_test():
    pc = FurnitureCloud(furniture_id = 10, num_samples=2000)
    pointnet2_fe = get_pointnet2_fe("pointnet2.pt")
    print(pointnet2_fe)
    pointnet2_fe.eval()
    part_name_list = pc._get_object_name_list()
    part_pointcloud_dict = pc._get_object_cloud_dict(pc.target_pose_dict)
    with torch.no_grad():
        for part_name in part_name_list:
            part_cloud_data = open3d_to_pyg(part_pointcloud_dict[part_name])
            part_cloud_batch = Batch().from_data_list([part_cloud_data])
            part_cloud_batch.to(device)
            part_cloud_vector = pointnet2_fe(copy.deepcopy(part_cloud_batch))
            print(part_name, part_cloud_vector)

if __name__ == "__main__":
    main_test()

