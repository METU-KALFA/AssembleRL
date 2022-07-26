import numpy as np
import open3d as o3d

import copy
import xml.etree.ElementTree as ET

import os
import copy

from assembly_learning.objects import xml_path_completion
from assembly_learning.objects import furniture_xmls
import assembly_learning.utils.transform_utils as T


class FurnitureCloud(object):
    '''
    Defines the base class for creating and
    manipulating pointclouds from furnitures. 
    Containes methods for xml parsing, mesh reading and pointcloud 
    creation from meshes. 
    '''
    def __init__(self, furniture_id, num_samples=10000):
        self._furniture_id = furniture_id
        self._num_sample = num_samples
        self.path = xml_path_completion(furniture_xmls[self._furniture_id])
        self.folder = os.path.dirname(self.path)
        self.tree = ET.parse(self.path)
        self.root = self.tree.getroot()
        self.name = self.root.get("model")
        
        self.scale_factor = self._get_scale_factor()
        self.target_pose_dict = self._get_target_pose_dict()
        self.mesh_dict = self._get_mesh_dict()

        self.visualizer = o3d.visualization.Visualizer()
            
    def _get_mat_from_body_ele(self, ele):
            pos = np.array([float(entry) for entry in ele.get("pos").split(" ")] if ele.get("pos") else [0.,0.,0.])
            quat = np.array([float(entry) for entry in ele.get("quat").split(" ")] if ele.get("quat") else [1.,0.,0.,0.])
            quat = T.convert_quat(quat) # converts from w,x,y,z to x,y,z,w
            pose = (pos,quat)
            pose_mat = T.pose2mat(pose) 
            return pose_mat

    def _get_target_pose_dict(self):
        '''
        Iterates over body elements in the xml tree.
        Returns a dictionary with body names as keys and
        dicitonaries of pos and quat as values.
        '''
        target_pose = {}
        for body in self.root.iter("body"):
            pose_mat = self._get_mat_from_body_ele(body)
            target_pose[body.get("name")] = {"pose": pose_mat}
        return target_pose

    def _get_mesh_dict(self):
        '''
        Iterates over geom elements of each body 
        element in the xml tree. Finds corresponding
        mesh file for the geom elements. Loads geom meshes,
        transforms them and then sum to obtain only one 
        mesh for the body part.
        Returns a dictionary containing body names as keys
        and corresponding body meshes as values.        
        '''
        mesh_dict = {}
        for body in self.root.iter("body"):
            body_mesh = o3d.geometry.TriangleMesh()
            bodyname = body.get("name")
            for geom in body.iter("geom"):
                if geom.get("type") == "mesh":
                    geom_rel_pos = np.array([float(entry) for entry in geom.get("pos").split(" ")] if geom.get("pos") else [0.,0.,0.])
                    geom_rel_quat = np.array([float(entry) for entry in geom.get("quat").split(" ")] if geom.get("quat") else [1.,0.,0.,0.])
                    geom_rel_quat = T.convert_quat(geom_rel_quat)
                    geom_rel_pose = (geom_rel_pos, geom_rel_quat)
                    geom_rel_pose_mat = T.pose2mat(geom_rel_pose)
                    geom_rel_pose_mat[[0,1,2],[0,1,2]] *= self.scale_factor
                    for mesh in self.root.iter("mesh"):
                        if mesh.get("name") == geom.get("mesh"):
                            filepath = os.path.join(self.folder, mesh.get("file"))
                            geom_mesh = o3d.io.read_triangle_mesh(filepath)
                            body_mesh += self._transform(copy.deepcopy(geom_mesh), geom_rel_pose_mat)
                            break
            mesh_dict[bodyname] = body_mesh
        return mesh_dict
    
    def _get_cc_obj_pose_dict(self):
        cc_obj_pose_dict = {}
        for body in self.root.iter("body"):
            cc_obj_pose_dict[body.get("name")] = self._get_mat_from_body_ele(body)
        return cc_obj_pose_dict

    def _get_site_rel_pose_dict(self):
        site_rel_pose_dict = {}
        for site in self.root.iter("site"):
            site_rel_pose_dict[site.get("name")] = self._get_mat_from_body_ele(site) 
        return site_rel_pose_dict

    def _get_site_info_dict(self):
        site_info_dict = {}
        for body in self.root.iter("body"):
            for site in body.iter("site"):
                site_info_dict[site.get("name")] = body.get("name")
        return site_info_dict

    def _get_object_name_list(self):
        object_name_list = []
        for body in self.root.iter("body"):
            object_name_list.append(body.get("name"))
        return object_name_list

    def _get_scale_factor(self):
        '''
        Returns scaling factor of the meshes from the xml tree.
        Note: assumes scale factor is same between the meshes
        and scale dimensions.
        To do:
             * Can be integrated into _get_mesh_dict function 
               for furnitures containing meshes with different scales.
        '''
        for child in self.root.iter("mesh"):
            return float(child.get("scale").split(" ")[0])
    
    def _get_conn_site_size(self):
        '''
        Rerturns size of a connection site.
        '''
        for site in self.root.iter("site"):
            if "conn_site" in site.get("name"):
                return float(site.get("size"))

    def _translate(self, mesh, pos):
        '''
        Translates the given mesh with the pos value.
        Returns the deepcopied mesh. 
        The input mesh is not affected.
        '''
        return copy.deepcopy(mesh).translate(pos)

    def _rotate(self, mesh, quat):
        '''
        Rotates the given mesh with the quat value.
        The quat's order: [w,x,y,z] (Not sure)
        Returns the deepcopied mesh. 
        The input mesh is not affected.
        '''
        R = mesh.get_rotation_matrix_from_quaternion(quat)
        return copy.deepcopy(mesh).rotate(R, mesh.get_center())

    def _scale(self, mesh, ratio):
        '''
        Scales the given mesh with the ratio value.
        Scaling center is (0,0,0).
        Returns the deepcopied mesh. 
        The input mesh is not affected.
        '''
        return copy.deepcopy(mesh).scale(ratio, center=(0,0,0))

    def _transform(self, mesh, hom_mat):
        '''
        Transforms the mesh with a 4x4 homogenous matrix.
        '''
        return copy.deepcopy(mesh).transform(hom_mat)
    
    def _create_window(self):
        self.visualizer.create_window()

    def _destroy_window(self):
        self.visualizer.destroy_window()
        
    def _custom_draw_geometries(self, geometries = []):
        self.visualizer.clear_geometries()
        for geometry in geometries:
            self.visualizer.add_geometry(geometry)
        self.visualizer.run()

    def _draw_pointcloud(self, pose_dict, part_based=True):
        if part_based:
            cloud, coordinate_frame = self._create_furniture_cloud(pose_dict)
            o3d.visualization.draw_geometries([cloud, coordinate_frame])
        else:
            mesh = self._create_furniture_mesh(pose_dict)
            cloud = mesh.sample_points_uniformly(self._num_sample)
            o3d.visualization.draw_geometries([cloud])
            
    def _draw_pointcloud2(self, clouds):
        o3d.visualization.draw_geometries(clouds)

    def _write_pointcloud(self, name, pose_dict):
        cloud, _  = self._create_furniture_cloud(pose_dict)  
        o3d.io.write_point_cloud(name, cloud)

    def _draw_registration_result(self, source, target):
        '''
        Helper function for drawing registration result.
        '''
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        o3d.visualization.draw_geometries([source_temp, target_temp])
        return target_temp + source_temp

    def _create_furniture_mesh(self, pose_dict):
        '''
        Given the pose dict containing keys as part names
        and values as dictionaries containing pos and quat,
        returns one mesh containing all parts.
        '''
        mesh = o3d.geometry.TriangleMesh()
        part_dict = self.mesh_dict
        for name in pose_dict.keys():
            part = part_dict[name] 
            dummy = list(pose_dict.keys())[0]
            if "pose" in  pose_dict[dummy].keys():
                pose_mat = pose_dict[name]["pose"]
                mesh += self._transform(copy.deepcopy(part), pose_mat)
            else:
                pos = pose_dict[name]["pos"]
                quat = pose_dict[name]["quat"]
                mesh += self._rotate(self._translate(copy.deepcopy(part), pos), quat)
        return  mesh 

    def _create_furniture_cloud(self, pose_dict):
        furniture_cloud = o3d.geometry.PointCloud()
        furniture_cf = o3d.geometry.TriangleMesh()
        part_dict = self.mesh_dict
        for name in pose_dict.keys():
            part = part_dict[name] 
            pose_mat = pose_dict[name]["pose"]
            mesh = self._transform(copy.deepcopy(part), pose_mat)
            furniture_cf += self._transform(o3d.geometry.TriangleMesh.create_coordinate_frame(size=self.scale_factor), pose_mat)
            cloud = mesh.sample_points_uniformly(self._num_sample, seed=501)
            if "color" in pose_dict[name].keys():
                color = pose_dict[name]["color"]
                cloud.paint_uniform_color(color)
            furniture_cloud += cloud
            if "conn_site" in pose_dict[name].keys():
                cs_size = self._get_conn_site_size()
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=cs_size)
                for rel_pose_mat in pose_dict[name]["conn_site"]:
                    cs_pose_mat = np.matmul(pose_mat, rel_pose_mat)
                    mesh = self._transform(copy.deepcopy(sphere), cs_pose_mat)
                    furniture_cf += self._transform(o3d.geometry.TriangleMesh.create_coordinate_frame(size=cs_size), cs_pose_mat)
                    cloud = mesh.sample_points_uniformly(self._num_sample, seed=501)
                    cloud.paint_uniform_color(np.random.rand(3))
                    furniture_cloud += cloud
        return furniture_cloud, furniture_cf

    def _create_furniture_cloud_visualization(self, pose_dict, not_point):
        furniture_cloud = o3d.geometry.PointCloud()
        furniture_cf = o3d.geometry.TriangleMesh()
        part_dict = self.mesh_dict
        for name in pose_dict.keys():
            if name == "3_part3": continue
            part = part_dict[name] 
            pose_mat = pose_dict[name]["pose"]
            mesh = self._transform(copy.deepcopy(part), pose_mat)
            furniture_cf += self._transform(o3d.geometry.TriangleMesh.create_coordinate_frame(size=self.scale_factor), pose_mat)
            cloud = mesh.sample_points_uniformly(self._num_sample, seed=501)
            if name == not_point:
                cloud.paint_uniform_color([	128./255.,0./255.,0./255.])
            else:    
                cloud.paint_uniform_color([128./255.,128./255.,128./255.])

            furniture_cloud += cloud
        return furniture_cloud, furniture_cf

    def _get_object_cloud_dict(self, pose_dict=None):
        if pose_dict is None:
            pose_dict = self.target_pose_dict
        part_cloud_dict = {}
        name_list = self._get_object_name_list()
        part_dict = self.mesh_dict
        for name in name_list:
            part = part_dict[name] 
            pose_mat = pose_dict[name]["pose"]
            mesh = self._transform(copy.deepcopy(part), pose_mat)
            cloud = mesh.sample_points_uniformly(self._num_sample, seed=501)
            part_cloud_dict[name] = cloud
        return part_cloud_dict

    def _extract_vd_pointcloud(self, cloud):
        diameter = np.linalg.norm(
            np.asarray(cloud.get_max_bound()) - np.asarray(cloud.get_min_bound()))
        camera = [0, -1, diameter]
        radius = diameter * 100
        sphere = o3d.geometry.TriangleMesh.create_box(0.1, 0.1, 0.1)
        sphere = self._translate(sphere, [0, -1, 0])
        _, pt_map = cloud.hidden_point_removal(camera, radius)
        vd_cloud = cloud.select_by_index(pt_map)
        print(np.asarray(cloud.points).shape, np.asarray(vd_cloud.points).shape)
        return vd_cloud, sphere

    def get_pointcloud(self, pose_dict):
        mesh = self._create_furniture_mesh(pose_dict)    
        cloud = mesh.sample_points_uniformly(self._num_sample, seed=501)
        return cloud
        
    def get_target_pointcloud(self):
        target_mesh = self._create_furniture_mesh(self.target_pose_dict)
        target_cloud = target_mesh.sample_points_uniformly(self._num_sample, seed=501)
        return target_cloud
    
    def get_pointcloud_from_points(self,np_points):
        points = o3d.utility.Vector3dVector(np_points)
        return o3d.geometry.PointCloud(points=points)
