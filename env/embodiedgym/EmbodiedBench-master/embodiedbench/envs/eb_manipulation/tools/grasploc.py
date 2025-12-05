import open3d as o3d
import argparse
import numpy as np
from itertools import combinations
import pickle
import os
from distutils.util import strtobool

class GraspPoint:
    def __init__(self, centroid, normal, principal, shell_r, shell_h, idx, bbox3d):
        self.centroid = centroid
        self.normal = normal
        self.principal = principal
        self.shell_r = shell_r
        self.shell_h = shell_h
        self.neighbor_idx = idx
        self.bbox3d = bbox3d # x, y, z: x is length along normal, z is length along principal

class Grasploc:
    def __init__(self, args):
        self.args = args
        self.mesh, self.pcd, self.densepcd = None, None, None
        self.grasp_points = []
        self.se3_output = None

    def run(self, origin_offset=None):
        print('preprocess')
        ret = self.preprocess()
        if ret == 0:
            self.find_grasp1()
            if self.args.vis_debug:
                self.visual_check()
            self.save_result(origin_offset)
    
    def preprocess(self):
         # sample grasp pose on surface (self.pcd), but calculate normal, principal based on volume samples (self.densepcd)
        if self.args.use_meshlab:
            try:
                meshlab_tmp_file = self.args.input_file[:-4] + '_meshlab.ply'
                # if not os.path.exists(meshlab_tmp_file) or self.args.meshlab_always:
                try:
                    os.system('meshlabserver -i {} -o {} -s {}'.format(self.args.input_file, meshlab_tmp_file, self.args.meshlab_sampling_file))
                    ply = o3d.io.read_point_cloud(meshlab_tmp_file).voxel_down_sample(voxel_size=self.args.pcd_sample_voxel_size)
                    o3d.io.write_point_cloud(meshlab_tmp_file, ply)
                    self.pcd = o3d.io.read_point_cloud(meshlab_tmp_file)
                except Exception as e:
                    print(e)
                    print('cannot use meshlab to sample on {}'.format(self.args.input_file))
                    self.pcd = o3d.io.read_triangle_mesh(self.args.input_file).sample_points_poisson_disk(number_of_points=100000).voxel_down_sample(voxel_size=self.args.pcd_sample_voxel_size)
                # self.pcd = o3d.io.read_triangle_mesh(self.args.input_file).sample_points_poisson_disk(number_of_points=40000)
                # processing of meshlab sampling output: decide required number of points in point cloud by object bbox surface size; translate point cloud to be centered at zero origin
            except Exception as e:
                print(e)
                return 1
        else:
            self.pcd = o3d.io.read_triangle_mesh(self.args.input_file).sample_points_poisson_disk(number_of_points=200000).voxel_down_sample(voxel_size=self.args.pcd_sample_voxel_size)
        self.crop_factor = 1.0
        if not self.args.no_crop:
            # bbx = o3d.geometry.AxisAlignedBoundingBox(
            #     min_bound=np.array([self.args.crop_x_min, self.args.crop_y_min, self.args.crop_z_min]),
            #     max_bound=np.array([self.args.crop_x_max, self.args.crop_y_max, self.args.crop_z_max]))
            trans = self.args.crop_box_transform[:3,3]
            rot = self.args.crop_box_transform[:3,:3]
            x_extent = self.args.crop_x_max-self.args.crop_x_min
            y_extent = self.args.crop_y_max - self.args.crop_y_min
            z_extent = self.args.crop_z_max- self.args.crop_z_min
            extent = np.array([x_extent, y_extent, z_extent])
            bbx = o3d.geometry.OrientedBoundingBox(center=trans, R=rot, extent=extent)
            original_points = len(self.pcd.points)
            self.pcd = self.pcd.crop(bbx)
            self.crop_factor = len(self.pcd.points) / original_points
        if len(self.pcd.points) == 0:
            print('mesh has no point samples in cropped area, quit')
            return 1
        else:
            return 0

    def find_grasp1(self):
        self.pcd.estimate_normals()
        pcd_tree = o3d.geometry.KDTreeFlann(self.pcd)
        ft, fd, fw, fl = self.args.finger_thickness, self.args.finger_max_distance, self.args.finger_width, self.args.finger_length
        # The minimum number of points that are required to be inside of a grasp
        self.downpcd = self.pcd.voxel_down_sample(voxel_size=self.args.grasp_sample_voxel_size)
        while len(self.downpcd.points)>2000:
            self.args.grasp_sample_voxel_size = self.args.grasp_sample_voxel_size*1.2
            self.downpcd = self.pcd.voxel_down_sample(voxel_size=self.args.grasp_sample_voxel_size)
        self.downpcd.paint_uniform_color([1, 0, 0])
        # if self.args.vis_debug:
        #     geoms = []
        #     self.pcd.paint_uniform_color([0.8, 0.8, 0.8])
        #     geoms.append(self.pcd)
        #     geoms.append(self.downpcd)
        #     o3d.visualization.draw_geometries(geoms, point_show_normal=True)
        # for pt, nm in zip(self.pcd.points[::dr], self.pcd.normals[::dr]):
        print('num of samples:', len(self.downpcd.points))
        min_num_points_in_grasp = (fd*fl + fl*ft + fd*ft) / self.args.pcd_sample_voxel_size**2 * self.args.min_num_points_between_proportion
        min_num_points_in_grasp *= self.crop_factor / 2
        while len(self.grasp_points) <= 10:
            for i, (pt, nm) in enumerate(zip(self.downpcd.points, self.downpcd.normals)):
                print(i, '/', len(self.downpcd.points))
                # grasp along negative normal, aka pointing inward object orthogonal to local surface
                # do it for positive and negative direction as calculated normal is not always pointing inwards
                for normal in [nm, -nm]:
                    # create a cuboid of the volume between two fingers ((finger distance + 2*finger width) * finger_thickness * finger length), 
                    # here finger distance can be 0-max distance
                    # rotate about normal to find axis connecting two fingers that has blank space on both sides for fingers while has points between, 
                    # (optional) the side surface direction should be consistent with finger's direction
                    # 1. generate principal samples on orthogonal plane of normal
                    normal = normal / np.linalg.norm(normal)
                    finger_percents=[0.1, 0.2,0.4,0.5]
                    for f_p in finger_percents:
                        centroid = pt + fl*f_p * normal
                        tmp = np.array([1, 0, 0])
                        principal = tmp - np.dot(tmp, normal) * normal
                        principal = principal / np.linalg.norm(principal)
                        theta = np.pi * 2 / self.args.rotation_sample
                        principals = [np.cos(i*theta) * principal + np.sin(i*theta) * np.cross(principal, normal) for i in range(self.args.rotation_sample)]
                        principals = [p / np.linalg.norm(p) for p in principals]
                        # 2. create cuboid along normal and principal direction, normal <-> finger length; principal <-> finger thickness; cross <-> grasp aperture
                        # check point num between two fingers and on two sides
                        radius = np.sqrt(ft**2 + (fd + 2*fw)**2 + fl**2) / 2
                        _, idx, _ = pcd_tree.search_radius_vector_3d(centroid, radius=radius)
                        pt_offset = np.asarray(self.pcd.points)[idx[1:], :] - centroid
                        for principal in principals:
                            cross = np.cross(normal, principal)
                            # for aperture in np.linspace(self.args.min_grasp_distance, fd, 10):
                            between_conditions = [np.abs(pt_offset @ normal) < fl / 2, np.abs(pt_offset @ principal) < ft / 2, np.abs(pt_offset @ cross) < fd / 2]
                            # extend between area distance of finger width along +/-cross and -normal
                            around_conditions = [pt_offset @ normal < fl / 2, pt_offset @ normal > -fl / 2 - fw, np.abs(pt_offset @ principal) < ft / 2, np.abs(pt_offset @ cross) < fd / 2 + fw]
                            n_between = np.sum(np.bitwise_and.reduce(between_conditions))
                            n_around = np.sum(np.bitwise_and.reduce(around_conditions)) - n_between
                            # print(n_between, min_num_points_in_grasp, n_around, self.args.max_num_points_intefering)
                            if n_between > min_num_points_in_grasp and n_around < self.args.max_num_points_intefering:
                                grasp = GraspPoint(centroid, normal, principal, fd, None, None, [ft, fd, fl])
                                self.grasp_points.append(grasp)
                                # grasp.normal = -grasp.normal
                                # grasp.principal = -grasp.principal
                                # self.grasp_points.append(grasp)
                                # print('found a grasp')
                                # self.visual_check(grasp, vis_bbox=True)
            if len(self.grasp_points)<=10:
                min_num_points_in_grasp = min_num_points_in_grasp*0.5
                print('Release the min_num_points_in_grasp to {}'.format(min_num_points_in_grasp))
            elif len(self.grasp_points) == 0:
                self.args.grasp_sample_voxel_size = self.args.grasp_sample_voxel_size*0.8
                self.downpcd = self.pcd.voxel_down_sample(voxel_size=self.args.grasp_sample_voxel_size)
                print('Grasp poses are too litte. Resample the point cloud with {}'.format(self.args.grasp_sample_voxel_size))
        print('grasp candidates:', len(self.grasp_points))
        
    def visual_check(self, grasp=None, vis_bbox=False):
        # visualization, create cylinder
        geoms = []
        self.pcd.paint_uniform_color([0.8, 0.8, 0.8])
        geoms.append(self.pcd)
        grasp_set = self.grasp_points if grasp is None else [grasp]
        n_vis = 1000
        fw = self.args.finger_width
        dr = max(len(self.grasp_points) // n_vis, 1)
        for grasp in grasp_set[::dr]:
            trans_mat = np.identity(4)
            trans_mat[:3, 2] = grasp.normal
            trans_mat[:3, 0] = grasp.principal
            trans_mat[:3, 1] = np.cross(grasp.normal, grasp.principal)
            trans_mat[:3, 3] = grasp.centroid

            # grasp_finger_connection_vis.transform(trans_corr_mat)
            # grasp_finger_connection_vis.transform(trans_mat)
            # geoms.append(grasp_finger_connection_vis)
            if vis_bbox:
                trans_corr_mat = np.identity(4)
                grasp_bbox_vis = o3d.geometry.TriangleMesh.create_box(width=grasp.bbox3d[0], height=fw, depth=grasp.bbox3d[2])
                trans_corr_mat[0, 3] = -grasp.bbox3d[0]/2
                trans_corr_mat[1, 3] = -grasp.bbox3d[1]/2 - fw
                trans_corr_mat[2, 3] = -grasp.bbox3d[2]/2
                grasp_bbox_vis.transform(trans_corr_mat)
                grasp_bbox_vis.transform(trans_mat)
                geoms.append(grasp_bbox_vis)
                grasp_bbox_vis = o3d.geometry.TriangleMesh.create_box(width=grasp.bbox3d[0], height=fw, depth=grasp.bbox3d[2])
                trans_corr_mat[0, 3] = -grasp.bbox3d[0]/2
                trans_corr_mat[1, 3] = grasp.bbox3d[1]/2
                trans_corr_mat[2, 3] = -grasp.bbox3d[2]/2
                grasp_bbox_vis.transform(trans_corr_mat)
                grasp_bbox_vis.transform(trans_mat)
                geoms.append(grasp_bbox_vis)
                grasp_bbox_vis = o3d.geometry.TriangleMesh.create_box(width=grasp.bbox3d[0], height=grasp.bbox3d[1] + 2*fw, depth=fw)
                # grasp_bbox_vis.paint_uniform_color([0, 0, 0.1])
                trans_corr_mat[0, 3] = -grasp.bbox3d[0]/2
                trans_corr_mat[1, 3] = -grasp.bbox3d[1]/2 - fw
                trans_corr_mat[2, 3] = -grasp.bbox3d[2]/2 - fw
                grasp_bbox_vis.transform(trans_corr_mat)
                grasp_bbox_vis.transform(trans_mat)
                geoms.append(grasp_bbox_vis)
                self.args.frame_size = 0.02 # to show axis with bbox
                mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=self.args.frame_size)
                tmp_trans_mat = np.identity(4)
                tmp_trans_mat[:3, 3] = grasp.centroid
                mesh_frame.transform(tmp_trans_mat)
                geoms.append(mesh_frame)
                self.args.frame_size = 0.05
            
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=self.args.frame_size)
            mesh_frame.transform(trans_mat)         
            geoms.append(mesh_frame)

        o3d.visualization.draw_geometries(geoms, point_show_normal=True)

    def save_result(self, origin_offset=None):
        self.se3_output = []
        for grasp in self.grasp_points:
            grasp_pose = np.identity(4)
            grasp_pose[:3, 3] = grasp.centroid
            grasp_pose[:3, 0] = grasp.principal
            grasp_pose[:3, 2] = grasp.normal
            grasp_pose[:3, 1] = np.cross(grasp.normal, grasp.principal)
            self.se3_output.append(grasp_pose)
        self.se3_output = np.array(self.se3_output)
        if origin_offset is not None:
            self.se3_output = np.einsum('ij,kjl->kil', np.linalg.inv(origin_offset), self.se3_output)
        with open(self.args.output_file, 'wb') as f:
            # pickle.dump(self.grasp_points, f) # only for debug
            pickle.dump(self.se3_output, f)
        print('grasp poses saved to', self.args.output_file)
    
    def load_result(self):
        with open(self.args.output_file, 'rb') as f:
            self.se3_output = pickle.load(f)
        print('grasp poses loaded from', self.args.output_file)


def define_default_args():
    parser = argparse.ArgumentParser()
    # point cloud, grasp sample density
    parser.add_argument('--use_meshlab',type=lambda x:bool(strtobool(x)), default=True, help='whether use meshlab to sample the mesh')
    parser.add_argument('--pcd_sample_voxel_size', type=int, default=0.0005, help='point cloud sample voxel grid size')
    parser.add_argument('--grasp_sample_voxel_size', type=int, default=0.005, help='grasp sample voxel grid size')

    # grasp generation params
    parser.add_argument('--min_num_points_between_proportion', type=int, default=0.2, help='The proportion of minimum points in the bbox formed by two fingers over the half surface size of the bbox divided by sampled point cloud voxel distance squared')
    parser.add_argument('--max_num_points_intefering', type=int, default=10, help='The maximum number of points allowed to interfere with the gripper')
    parser.add_argument('--min_grasp_distance', type=float, default=0.01, help='minimum distance between two fingers when the gripper close to grasp object')
    parser.add_argument('--finger_max_distance', type=float, default=0.08, help='The maximum distance between the two fingers of the gripper')
    parser.add_argument('--finger_width', type=float, default=0.02, help='finger width (the same direction as distance between two fingers)')
    parser.add_argument('--finger_thickness', type=float, default=0.02, help='distance between front and back side of fingers')
    parser.add_argument('--rotation_sample', type=int, default=16, help='grasp sample rotated about normal axis')
    parser.add_argument('--finger_length', type=float, default=0.045, help='finger length')
    
    # input point cloud grasp search region
    parser.add_argument('--crop_x_min', type=float, default=-0.1, help='x min for cropping the point cloud')
    parser.add_argument('--crop_x_max', type=float, default=0.1, help='x max for cropping the point cloud')
    parser.add_argument('--crop_y_min', type=float, default=-0.1, help='y min for cropping the point cloud')
    parser.add_argument('--crop_y_max', type=float, default=0.1, help='y max for cropping the point cloud')
    parser.add_argument('--crop_z_min', type=float, default=-0.1, help='z max for cropping the point cloud')
    parser.add_argument('--crop_z_max', type=float, default=0.1, help='z max for cropping the point cloud')
    parser.add_argument('--crop_box_transform', default=None, help='the transform of the crop box')
    parser.add_argument('--no_crop', type=lambda x:bool(strtobool(x)), default=True, help='Do not crop input point cloud (1 default; 0 to crop), set xyz min max to be the bbox around point cloud')

    # io, vis params
    parser.add_argument('--input_file', type=str, default='./flatbox.ply', help='(PLY file) object 3d model input file to search graspable poses')
    parser.add_argument('--output_file', type=str, default='./graspposes.pkl', help='file to save graspable poses using pickle')
    parser.add_argument('--meshlab_always', type=lambda x:bool(strtobool(x)), default=False, help='always use meshlab to sample regardless of whether there is previous saved file')
    parser.add_argument('--meshlab_sampling_file', type=str, default='./tools/meshlab_stratified_sampling.mlx', help='sampling script using meshlab')
    parser.add_argument('--vis_debug', type=lambda x:bool(strtobool(x)), default=False, help='visualize temporary results for debugging')
    parser.add_argument('--frame_size', type=float, default=0.01, help='coordinate frame size in visualization')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    grasp_pose_path = './vlm/grasp_poses/'
    ply_file = None
    args = define_default_args()
    args.input_file = ply_file
    args.output_file = grasp_pose_path + ply_file.split('/')[-1][:-4] + '.pkl'
    args.vis_debug = True
    args.meshlab_always=True
    gl = Grasploc(args)
    gl.run()
