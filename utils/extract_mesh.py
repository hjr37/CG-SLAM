import torch
import numpy as np
import open3d as o3d
from skimage.measure import marching_cubes
# from src.rendering import query_nn, raw_2_output_speed_mesh, query_filter, query_nn_mesh
from tqdm import tqdm
# import trimesh
# import camera.camera as camera
import glob
import yaml
import cv2


class TSDF:
    def __init__(self, depth_path, color_path, device=None, seq_name=None) -> None:
        self.camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            600, 340, 
            300.0, 300.0,
            299.75, 169.75)
        self.voxel_length = 0.015
        self.sdf_trunc = 0.1
        self.color_type = o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=self.voxel_length, sdf_trunc=self.sdf_trunc, color_type=self.color_type)
        self.img_path = sorted(glob.glob(color_path))
        self.depth_path = sorted(glob.glob(depth_path))
        self.seq_name = seq_name

    def reconstruction(self, pose, interval = None):
        print(len(self.depth_path))
        for i in tqdm(range(len(self.depth_path))):
            frame_pose = pose[i * interval]
            frame_pose = np.linalg.inv(frame_pose)
            color = o3d.io.read_image(self.img_path[i])

            depth = cv2.imread(self.depth_path[i], -1).astype(np.float32)
            depth = o3d.geometry.Image(depth)

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, depth_trunc=7.0, convert_rgb_to_intensity=False, depth_scale=6553.5)
            self.volume.integrate(rgbd, self.camera_intrinsics, frame_pose)
            mesh = self.volume.extract_triangle_mesh()
            # o3d.io.write_triangle_mesh('./reconstructions/' + self.seq_name + '_lightweight.ply', mesh)
            o3d.io.write_triangle_mesh('/mnt/nas_7/group/hujiarui/viz/office1/{}.ply'.format(i * interval), mesh)


if __name__ =='__main__':
    pose = torch.load('./output/Replica/office1/est_traj.pt')
    pose = pose.cpu().numpy()
    color_path = './output/Replica/office1/color/*.jpg'
    depth_path = './output/Replica/office1/depth/*.png'

    tsdf = TSDF(depth_path, color_path, device='cuda:0', seq_name='office1')
    tsdf.reconstruction(pose, interval=30)

    #! "Computing normals and rendering it in a .ply file"
    # mesh.compute_vertex_normals()
    # o3d.visualization.draw([mesh])