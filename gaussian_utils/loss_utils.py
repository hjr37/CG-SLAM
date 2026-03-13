#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from utils.utils import mask_in_image
import numpy as np

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def grad_loss(output, gt):
    def one_grad(shift):
        oy = output[:, shift:] - output[:, :-shift]
        ox = output[:, :, shift:] - output[:, :, :-shift]
        gy = gt[:, shift:] - gt[:, :-shift]
        gx = gt[:, :, shift:] - gt[:, :, :-shift]
        loss = (ox - gx).abs().mean() + (oy - gy).abs().mean()
        return loss
    loss = (one_grad(1) + one_grad(2) + one_grad(3)) / 3.
    return loss


def warp_loss(coords, cam_src, cam_dest, inv_K, K, W, H, num, device):
        #! randomly N pixels
        select_inds = np.random.choice(coords.shape[0], size=[num], replace=False)  # (N_rand,)
        select_coords = coords[select_inds].to(device)  # (N_rand, 2)
        #! -----------------
        # 2d -> 3D
        depths = cam_src['depth'][:, select_coords[:,1].to(torch.int64), select_coords[:,0].to(torch.int64)].view(-1, 1)
        prev_rgbs = cam_src['color'][:, select_coords[:,1].to(torch.int64), select_coords[:,0].to(torch.int64)].permute(1,0)
        src_3ds = torch.cat([select_coords * depths, depths], dim=-1) @ inv_K
        
        prev_rgbs = prev_rgbs[src_3ds[:, -1] > 0]
        src_3ds = src_3ds[src_3ds[:, -1] > 0]
        src_3ds_homo = torch.cat([src_3ds, torch.ones([src_3ds.shape[0], 1], device=device)], dim=-1)
        # reprojection
        dest_3ds = (src_3ds_homo @ cam_src['c2wT'] @ cam_dest['w2cT'])[:, :3]
        prev_rgbs = prev_rgbs[dest_3ds[:, -1] > 0]
        dest_3ds = dest_3ds[dest_3ds[:, -1] > 0]

        dest_3ds_homo = dest_3ds / dest_3ds[:, -1:]
        dest_2ds = (dest_3ds_homo @ K.T)[:, :2]
        # mask and grid_sample
        mask = mask_in_image(dest_2ds, (W, H), 0)
        dest_2ds = dest_2ds[mask]
        prev_rgbs = prev_rgbs[mask]

        dest_2ds[..., 0] = 2 * (dest_2ds[..., 0] / (W - 1)) - 1
        dest_2ds[..., 1] = 2 * (dest_2ds[..., 1] / (H - 1)) - 1
        dest_rgbs = F.grid_sample(cam_dest['gt_color'].unsqueeze(0), dest_2ds.view([1, 1, -1, 2]), align_corners=True).squeeze()
        # dest_rgbs = torch.cat([dest_rgbs, torch.zeros([3, 73], device=self.device)], dim=-1)
        
        # cv2.imwrite('./test_src_patch.jpg', cv2.cvtColor(to8b(prev_rgbs.reshape(200,200,3).detach().cpu().numpy()), cv2.COLOR_RGB2BGR))
        # cv2.imwrite('./test_dest_patch.jpg', cv2.cvtColor(to8b(dest_rgbs.permute(1,0).reshape(200,200,3).detach().cpu().numpy()), cv2.COLOR_RGB2BGR))
        # loss compute
        warp_loss = torch.abs(prev_rgbs.T-dest_rgbs).mean()
        return warp_loss

def consis_loss(mean_depth, median_depth):
    loss_consis = torch.abs(mean_depth - median_depth).mean()
    return loss_consis
