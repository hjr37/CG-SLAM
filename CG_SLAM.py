import os
import torch
import torch.optim as optim
from tqdm import tqdm
from utils.utils import *
from torch.utils.data import DataLoader
from dataset.dataset import get_dataset
from gaussian_model.gaussian_model import GaussianModel
from gaussian_utils.graphics_utils import BasicPointCloud
from gaussian_renderer import render
from gaussian_utils.graphics_utils import getProjectionMatrix
from gaussian_utils.loss_utils import ssim, consis_loss
from torch.autograd import Variable
from utils.loop_detection.loop_detector import LoopDetector
from torch.utils.tensorboard import SummaryWriter


def frame_preprocess(frame, device):
    frame['color'] = frame['color'].squeeze(0).permute(2, 0, 1).to(device)
    frame['pose'] = frame['pose'].squeeze(0).to(device)
    frame['depth'] = frame['depth'].squeeze(0).to(device)


class Viewpoint_Cam():
    def __init__(self, frame, device, proj_matrix) -> None:
        self.world_view_transform = torch.inverse(frame['pose']).transpose(0, 1).to(device)
        self.projection_matrix = proj_matrix
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
    
    @torch.no_grad()
    def update(self, viewmatrix):
        self.world_view_transform = viewmatrix
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


class CG_SLAM():
    '''
    Main class of CG-SLAM, incuding initializing, tracking and mapping.
    Args:
        cfg (dict) : dict contains config file arguments
        args (argparse.Namespace): manually input arguments
    '''
    def __init__(self, cfg, args, hyper_group) -> None:
        self.cfg = cfg
        self.args = args
        self.device = torch.device(args.device)
        self.patch_size = [1, 1]
        self.dataset = get_dataset(cfg, self.args, cfg['scale'], device=self.device)
        
        if args.output_folder is None:
            if os.path.exists(cfg['data']['output_folder']):
                self.output_folder = cfg['data']['output_folder']
            else:
                os.makedirs(cfg['data']['output_folder'])
                self.output_folder = cfg['data']['output_folder']
        else:
            self.output_folder = args.output_folder
            
        self.focus_iter = 1980 if cfg['name'] == 'ro2' else 10000
            
        
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = cfg['cam']['H'],  cfg['cam']['W'],\
                                                             cfg['cam']['fx'], cfg['cam']['fy'],\
                                                             cfg['cam']['cx'], cfg['cam']['cy']
        self.zfar = 100.0
        self.znear = 0.01
        self.update_cam()
        self.uv_list = uniform_sample(self.H, self.W, self.patch_size, self.device)
        self.K = torch.tensor([[self.fx, 0, self.cx], 
                               [0, self.fy, self.cy], 
                               [0,      0,       1]], device=self.device)
        self.inv_K = torch.inverse(self.K).T
        self.compute_tan_fov() # half_tanfovx, half_tanfovy
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, tanHalfFovX=self.half_tanfovx, tanHalfFovY=self.half_tanfovy).transpose(0,1).to(self.device)

        self.gaussians = GaussianModel(max_sh_degree=0)
        self.extent_size = cfg['extent_size']
        self.lr_scale = cfg['lr_scale']
        self.keyframe_list = []

        self.model_hyper = hyper_group['model'] 
        self.optim_hyper = hyper_group['optim'] 
        self.pipe_hyper = hyper_group['pipe'] 
        
        bg_color = [1, 1, 1] if self.model_hyper.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device=self.device)

        # SIM calculate
        conf = {
                'checkpoint_path': './utils/loop_detection/TokyoTM_struct.mat',
                'whiten': True
                }
        self.loop_detector = LoopDetector(conf, self.device) 
        
        self.weight = {
            "T_color_weight": cfg["tracking"]["T_color_weight"],
            "T_depth_weight": cfg["tracking"]["T_depth_weight"],
            "T_rot_lr": cfg["tracking"]["T_rot_lr"],
            "T_tran_lr": cfg["tracking"]["T_tran_lr"],
            "M_color_weight": cfg["mapping"]["M_color_weight"],
            "M_depth_weight": cfg["mapping"]["M_depth_weight"]
        }
        
        self.optim_hyper.scaling_lr = self.cfg['optim_hyper']['scaling_lr'] if self.cfg['optim_hyper']['scaling_lr'] != 'None' else self.optim_hyper.scaling_lr
        self.optim_hyper.position_lr_init = self.cfg['optim_hyper']['position_lr_init'] if self.cfg['optim_hyper']['position_lr_init'] != 'None' else self.optim_hyper.position_lr_init
        self.optim_hyper.position_lr_final = self.cfg['optim_hyper']['position_lr_final'] if self.cfg['optim_hyper']['position_lr_final'] != 'None' else self.optim_hyper.position_lr_final
        self.optim_hyper.feature_lr = self.cfg['optim_hyper']['feature_lr'] if self.cfg['optim_hyper']['feature_lr'] != 'None' else self.optim_hyper.feature_lr


    def run(self):
        tb_writer = SummaryWriter()
        data_loader = DataLoader(self.dataset, num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=10)
        e_T, e_R = 0., 0.
        slam_bar = tqdm(total=len(data_loader) - 1, desc='Training progress', colour='green')
        delta_pose = None

        trajectory_path = os.path.join(self.output_folder, "trajectory_est.txt")
        if os.path.exists(trajectory_path):
            with open(trajectory_path, "w"):
                pass

        for iteration, frame in enumerate(data_loader):
            frame_preprocess(frame, self.device)
            viewpoint_cam = Viewpoint_Cam(frame, self.device, self.projection_matrix)
            gt_image = frame['color'].type(torch.float32)
            gt_depth = frame['depth'].type(torch.float32)
            
            if iteration != 0:
                slam_bar.update(1)

            if iteration == 0:
                self.initialize(frame, viewpoint_cam, gt_image, gt_depth)
                with torch.no_grad():
                    key_des = self.loop_detector.get_frame_des(frame['color'])
                    self.loop_detector.add_des(key_des.detach())
                self.keyframe_list.append({'view_cam': viewpoint_cam, 'color': gt_image, 'depth': gt_depth, 'gt_pose': frame['pose']})
                last_pose = frame['pose'].detach()
                slam_bar.update(0)
                slam_bar.set_postfix({'Err_t': e_T, 'Err_R': e_R})

                flattened_pose = torch.flatten(last_pose)
                flattened_pose = " ".join([str(num.item()) for num in flattened_pose])
                with open(trajectory_path, "a") as file:
                    file.write(flattened_pose + "\n")

                continue

            optim_c2w = self.tracking_so3(last_pose, viewpoint_cam, gt_image, frame['pose'], gt_depth, delta_pose=delta_pose, id=iteration)    

            with torch.no_grad():
                e_T, e_R = CalPoseError(optim_c2w, frame['pose'])
                if tb_writer is not None:
                    tb_writer.add_scalar(self.cfg['dataset'] + 'Error_T', e_T, iteration)
                    tb_writer.add_scalar(self.cfg['dataset'] + 'Error_R', e_R, iteration)
                slam_bar.set_postfix({'Err_t': e_T, 'Err_R': e_R})

            if len(self.keyframe_list) < self.cfg['mapping']['window_size'] - 1:
                window_size = len(self.keyframe_list) + 1
            else:
                window_size = self.cfg['mapping']['window_size']

            if iteration % self.cfg['mapping']['interval'] == 0 or iteration > self.focus_iter:
                with torch.no_grad():
                    key_des = self.loop_detector.get_frame_des(frame['color'])
                    keyframe_ids = self.loop_detector.keyframe_selection(key_des, num=window_size - 2)
                    self.loop_detector.add_des(key_des.detach())

                viewpoint_cam.update(torch.inverse(optim_c2w.detach()).T)
                self.grow_points(frame, optim_c2w.detach(), viewpoint_cam) 
                ba_c2w = self.mappingBA_warp({'view_cam': viewpoint_cam, 'color': gt_image, 'depth': gt_depth}, pose_guess=optim_c2w.detach()[None, ...], keyframe_ids=keyframe_ids)
                if ba_c2w is not None:
                    optim_c2w = ba_c2w

                viewpoint_cam.update(torch.inverse(optim_c2w.detach()).T)
                self.keyframe_list.append({'view_cam': viewpoint_cam, 'color': gt_image, 'depth': gt_depth, 'gt_pose': frame['pose']})
                
            delta_pose = torch.inverse(last_pose) @ optim_c2w # n to n-1
            last_pose = optim_c2w.detach()    

            flattened_pose = torch.flatten(last_pose)
            flattened_pose = " ".join([str(num.item()) for num in flattened_pose])
            with open(trajectory_path, "a") as file:
                file.write(flattened_pose + "\n")

        tb_writer.close()


    def initialize(self, frame, viewpoint_cam, gt_image, gt_depth):
        init_bar = tqdm(range(self.cfg['mapping']['init']), desc="Initialization progress")
        
        # project 3d world gaussian central points
        positions, rgb = self.projection(self.uv_list, frame, frame['pose'])
        gau_w_3ds = BasicPointCloud(points=positions.clone(), colors=rgb.clone())
        
        # create gaussians from point cloud (spatial_lr_scale should match scene scale)
        self.gaussians.create_from_pcd(pcd=gau_w_3ds, spatial_lr_scale=self.lr_scale)
        self.gaussians.training_setup(self.optim_hyper)
        
        # Initialization Optimization
        w2cT = torch.tensor(viewpoint_cam.world_view_transform, requires_grad=False)
        for iteration in range(1, self.cfg['mapping']['init'] + 1):

            render_pkg = render(viewpoint_cam, self.gaussians, self.pipe_hyper, self.background, viewmatrix=w2cT, fov=(self.half_tanfovx, self.half_tanfovy), HW=(self.H, self.W), gt_depth=gt_depth, track_off=True, map_off=False)
            image, viewspace_point_tensor, visibility_filter, radii, depth, depth_median, depth_var, gau_uncertainty, num_related_pixels = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["depth"], render_pkg["depth_median"], render_pkg["depth_var"], render_pkg['gau_uncertainty'], render_pkg['num_related_pixels']
            
            Ll1 = torch.abs((image - gt_image)).mean()
            DL = torch.abs((depth[0] - gt_depth)).mean()
            loss_consis = consis_loss(depth[0], depth_median[0])
            loss_var = torch.abs(depth_var).mean()
            
            loss = (self.weight['M_color_weight'] - self.optim_hyper.lambda_dssim) * Ll1 + self.optim_hyper.lambda_dssim * (1.0 - ssim(image, gt_image)) + self.weight['M_depth_weight'] * DL + 0.1 * self.aniso_loss(r_threshold=1.0, visibility_filter=visibility_filter) + 0.25 * loss_consis + 0.15 * loss_var
            
            loss.backward()

            with torch.no_grad():
                if iteration % 10 == 0:
                    init_bar.set_postfix({"Loss": f"{loss:.{7}f}"})
                    init_bar.update(10)
                self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])

                self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration  == 0.5 * self.cfg['mapping']['init']:
                    size_threshold = 20
                    self.gaussians.densify_and_prune(self.optim_hyper.densify_grad_threshold, 0.005, self.extent_size, size_threshold)
                
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none = True)
                self.gaussians.update_learning_rate(iteration)

        init_bar.close()

    def tracking_so3(self, pose_guess, viewpoint_cam, gt_image, gt_pose, gt_depth, delta_pose=None, id=None):
        
        pose_guess = pose_guess[None, ...]
        gt_pose = gt_pose[None, ...]

        if delta_pose is not None:
            # constant speed consumption
            pose_guess = pose_guess @ delta_pose
        
        Tran = Variable(pose_guess[:, :3, 3].to(self.device), requires_grad=True)
        rot = Variable(matrix_to_axis_angle(pose_guess[:, :3, :3]).to(self.device), requires_grad=True)
        if id == 1:
            pose_optimizer = optim.Adam([{'params':[rot], 'lr':0.003},
                                        {'params':[Tran], 'lr':0.00215}], betas=(0.9, 0.999))
            iters = 200
        else:
            pose_optimizer = optim.Adam([{'params':[rot], 'lr':self.weight["T_rot_lr"]},
                                        {'params':[Tran], 'lr':self.weight["T_tran_lr"]}], betas=(0.9, 0.999))
            
            iters = self.cfg['tracking']['iters']

        for iteration in range(0, iters):
            
            pose_optimizer.zero_grad()
            w2cT = torch.inverse(at_to_transform_matrix(rot, Tran).to(self.device)[0]).T
            
            viewpoint_cam.update(w2cT.detach())

            render_pkg = render(viewpoint_cam, self.gaussians, self.pipe_hyper, self.background, viewmatrix=w2cT, fov=(self.half_tanfovx, self.half_tanfovy), HW=(self.H, self.W), gt_depth=gt_depth, track_off=False, map_off=True)
            image, depth, opacity_map = render_pkg["render"], render_pkg["depth"], render_pkg["opacity_map"]
            
            with torch.no_grad():
                opacity_mask = opacity_map.detach() > 0.99
                tmp_d = torch.abs(depth[0].detach() - gt_depth)[opacity_mask[0]]
                tmp_rgb = torch.sum(torch.abs(image.detach() - gt_image), dim=0)[opacity_mask[0]]
                mask_d = torch.logical_and(opacity_mask[0], (torch.abs(depth[0].detach() - gt_depth) < 10 * tmp_d.median()))
                mask_rgb = torch.logical_and(opacity_mask[0], (torch.sum(torch.abs(image.detach() - gt_image), dim=0) < 10 * tmp_rgb.median()))

            loss_color = (torch.abs((image - gt_image))).permute(1, 2, 0)[mask_rgb.detach()].mean()
            loss_depth = (torch.abs((depth[0] - gt_depth)))[mask_d.detach()].mean()
           
            loss = self.weight["T_color_weight"] * loss_color + self.weight["T_depth_weight"] * loss_depth
                    
            loss.backward()
            with torch.no_grad():
                pose_optimizer.step()

        return at_to_transform_matrix(rot, Tran).to(self.device)[0]


    def mappingBA_warp(self, current_view_dict, pose_guess=None, keyframe_ids=None):
        if pose_guess is None:
            raise ValueError("pose_guess is required for mappingBA_warp.")

        keyframe_ids = list(keyframe_ids or [])
        ba_frame_list = [self.keyframe_list[-1]] + [self.keyframe_list[i] for i in keyframe_ids]

        Tran = Variable(pose_guess[:, :3, 3].to(self.device), requires_grad=True)
        rot = Variable(matrix_to_axis_angle(pose_guess[:, :3, :3]).to(self.device), requires_grad=True)
        pose_optimizer = optim.Adam([{'params':[rot], 'lr':0.0015}, 
                                     {'params':[Tran], 'lr':0.00215}], betas=(0.9, 0.999))
        
        track_off = True
        for iteration in range(1, self.cfg['mapping']['iters'] + 1):
            if iteration  > (0.5 * self.cfg['mapping']['iters']):
                track_off = False

            loss_list = []
            self.gaussians.optimizer.zero_grad()
            pose_optimizer.zero_grad()

            cur_w2cT = torch.inverse(at_to_transform_matrix(rot, Tran).to(self.device)[0]).T
            current_view_dict['view_cam'].update(cur_w2cT.detach())
            cur_render_pkg = render(current_view_dict['view_cam'], self.gaussians, self.pipe_hyper, self.background, viewmatrix=cur_w2cT, fov=(self.half_tanfovx, self.half_tanfovy), HW=(self.H, self.W), gt_depth=current_view_dict['depth'], track_off=track_off, map_off=False)    
            cur_image, cur_viewspace_point_tensor, cur_visibility_filter, cur_radii, cur_depth, cur_depth_median, cur_depth_var, cur_gau_uncertainty, cur_num_related_pixels = cur_render_pkg["render"], cur_render_pkg["viewspace_points"], cur_render_pkg["visibility_filter"], cur_render_pkg["radii"], cur_render_pkg["depth"], cur_render_pkg["depth_median"], cur_render_pkg['depth_var'], cur_render_pkg['gau_uncertainty'], cur_render_pkg['num_related_pixels']

            Ll1 = torch.abs((cur_image - current_view_dict['color'])).mean()
            DL = torch.abs((cur_depth[0] - current_view_dict['depth'])).mean()
            loss_consis = consis_loss(cur_depth[0], cur_depth_median[0])
            loss_var = torch.abs(cur_depth_var).mean()
            
            loss = (self.weight['M_color_weight'] - self.optim_hyper.lambda_dssim) * Ll1 + self.optim_hyper.lambda_dssim * (1.0 - ssim(cur_image, current_view_dict['color'])) + self.weight['M_depth_weight'] * DL + 0.1 * self.aniso_loss(r_threshold=1.0, visibility_filter=cur_visibility_filter) + 0.25 * loss_consis + 0.15 * loss_var

            
            loss_list.append(loss)
          
            for seq, frame_dict in enumerate(ba_frame_list):
                w2cT = torch.tensor(frame_dict['view_cam'].world_view_transform, requires_grad=False)
                render_pkg = render(frame_dict['view_cam'], self.gaussians, self.pipe_hyper, self.background, viewmatrix=w2cT, fov=(self.half_tanfovx, self.half_tanfovy), HW=(self.H, self.W), gt_depth=frame_dict['depth'], track_off=track_off, map_off=False)    
                image, visibility_filter, depth, depth_median, depth_var, gau_uncertainty, num_related_pixels = render_pkg["render"], render_pkg["visibility_filter"], render_pkg["depth"], render_pkg["depth_median"], render_pkg['depth_var'], render_pkg['gau_uncertainty'], render_pkg['num_related_pixels']

                Ll1 = torch.abs((image - frame_dict['color'])).mean()
                DL = torch.abs((depth[0] - frame_dict['depth'])).mean()
                loss_consis = consis_loss(depth[0], depth_median[0])
                loss_var = torch.abs(depth_var).mean()
                
                loss = (self.weight['M_color_weight'] - self.optim_hyper.lambda_dssim) * Ll1 + self.optim_hyper.lambda_dssim * (1.0 - ssim(image, frame_dict['color'])) + self.weight['M_depth_weight'] * DL + 0.1 * self.aniso_loss(r_threshold=1.0, visibility_filter=visibility_filter) + 0.25 * loss_consis + 0.15 * loss_var
                
                loss_list.append(loss)
                cur_gau_uncertainty += gau_uncertainty
                cur_num_related_pixels += num_related_pixels
          
            loss_ba = torch.stack(loss_list, dim=0).mean()
            loss_ba.backward()

            with torch.no_grad():              
                self.gaussians.max_radii2D[cur_visibility_filter] = torch.max(self.gaussians.max_radii2D[cur_visibility_filter], cur_radii[cur_visibility_filter])
                self.gaussians.add_densification_stats(cur_viewspace_point_tensor, cur_visibility_filter)

                if iteration  == 0.5 * self.cfg['mapping']['iters']:
                    size_threshold = 20
                    self.gaussians.densify_and_prune(self.optim_hyper.densify_grad_threshold, 0.005, self.extent_size, size_threshold)
                self.gaussians.optimizer.step()
                
                if iteration  > (0.5 * self.cfg['mapping']['iters']):
                    pose_optimizer.step()

        return at_to_transform_matrix(rot, Tran).to(self.device)[0]



    def update_cam(self):
        """
        Update the camera intrinsics according to pre-processing config, 
        such as resize or edge crop.
        """
        # resize the input images to crop_size (variable name used in lietorch)
        print("CG-SLAM Light mode:", self.args.light)
        if self.args.light:
            if 'crop_size' in self.cfg['cam']:
                crop_size = self.cfg['cam']['crop_size']
                sx = crop_size[1] / self.W
                sy = crop_size[0] / self.H
                self.fx = sx * self.fx
                self.fy = sy * self.fy
                self.cx = sx * self.cx
                self.cy = sy * self.cy
                self.W = crop_size[1]
                self.H = crop_size[0]
            else:
                raise ValueError("Please set 'crop_size' in config file to enable light mode.")

        # cropping will change H, W, cx, cy, so need to change here
        if self.cfg['cam']['crop_edge'] > 0:
            self.H -= self.cfg['cam']['crop_edge'] * 2
            self.W -= self.cfg['cam']['crop_edge'] * 2
            self.cx -= self.cfg['cam']['crop_edge']
            self.cy -= self.cfg['cam']['crop_edge']
    
    def compute_tan_fov(self):
        self.half_tanfovx = 0.5 * self.W / abs(self.fx)
        self.half_tanfovy = 0.5 * self.H / abs(self.fy)

    def projection(self, uv, frame, last_pose):
        depth, rgb = get_depth_rgb(uv, frame['depth'], frame['color'])
        gau_cam_3ds = torch.cat([uv * depth, depth], dim=-1) @ self.inv_K

        rgb = rgb[gau_cam_3ds[:, -1] > 0]
        gau_cam_3ds = gau_cam_3ds[gau_cam_3ds[:, -1] > 0]  # filter zero depth

        gau_cam_3ds_homo = torch.cat([gau_cam_3ds, torch.ones(gau_cam_3ds.shape[0], 1, device=self.device)], dim=-1)
        return (gau_cam_3ds_homo @ last_pose.t())[..., :3], rgb

    @torch.no_grad()
    def grow_points(self, map_frame, last_pose, viewpoint_cam):

        w2cT = torch.tensor(viewpoint_cam.world_view_transform, requires_grad=False)
        render_pkg = render(viewpoint_cam, self.gaussians, self.pipe_hyper, self.background, viewmatrix=w2cT, fov=(self.half_tanfovx, self.half_tanfovy), HW=(self.H, self.W), gt_depth=map_frame['depth'], track_off=True, map_off=True)
        visibility_filter, opacity_map = render_pkg["visibility_filter"], render_pkg['opacity_map']

        opacity_mask = opacity_map[0] < 0.5
        indices = torch.nonzero(opacity_mask, as_tuple=False)[:, [1, 0]]
        positions, rgb = self.projection(indices, map_frame, last_pose)
        added_gaus = BasicPointCloud(points=positions.clone(), colors=rgb.clone())        
        self.gaussians.add_from_pcd(added_gaus, visibility_filter)
    

    def aniso_loss(self, r_threshold, visibility_filter):
        r_threshold = torch.tensor(r_threshold, device=self.device)
        gau_scales = self.gaussians.get_scaling[visibility_filter]
        max_scales, _ = torch.max(gau_scales, dim=-1)
        min_scales, _ = torch.min(gau_scales, dim=-1)
        ratio = torch.div(max_scales, min_scales)

        aniso_loss = (torch.where(ratio > r_threshold, ratio, r_threshold) - r_threshold).mean()

        return aniso_loss
