import numpy as np
import torch
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix, axis_angle_to_matrix, quaternion_to_axis_angle, se3_exp_map, se3_log_map
from skimage.color import rgb2gray
from skimage import filters
from torch_scatter import scatter_min, scatter_mean



def as_intrinsics_matrix(intrinsics):
    """
    Get matrix representation of intrinsics.

    """
    K = np.eye(3)
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[2]
    K[1, 2] = intrinsics[3]

    return K


def uniform_sample(H, W , patch_size, device):  #neural point cloud pixel sample

    v = np.arange(H/patch_size[0])
    u = np.arange(W/patch_size[1])

    grid_x, grid_y = np.meshgrid(u,v)
    uv_list = np.stack([grid_x, grid_y], -1).reshape(-1,2)  # (x,y)
    uv_list = torch.tensor(uv_list, dtype=torch.int64, device=device)

    return uv_list 

def get_depth_rgb(xy, depth, color):
    # depth[depth > 8.0] = 0
    # depth[depth < 0.2] = 0
    depth_list = depth[xy[:,1], xy[:,0]]
    color_list = color[:, xy[:,1], xy[:,0]]
    
    return depth_list.view(-1,1), color_list.permute(1,0)

def get_camera_from_tensor(inputs, device):
    """
    Convert quaternion and translation to transformation matrix.

    """
    N = len(inputs.shape)
    if N == 1:
        inputs = inputs.unsqueeze(0)
    quad, T = inputs[:, :4], inputs[:, 4:]
    R = quad2rotation(quad)
    RT = torch.cat([R, T[:, :, None]], 2)
    if N == 1:
        RT = RT[0]
        RT = torch.cat([RT,torch.tensor([[0., 0., 0., 1.]], device=device)], 0)
    return RT

def quad2rotation(quad):
    """
    Convert quaternion to rotation in batch. Since all operation in pytorch, support gradient passing.

    Args:
        quad (tensor, batch_size*4): quaternion.

    Returns:
        rot_mat (tensor, batch_size*3*3): rotation.
    """
    bs = quad.shape[0]
    qr, qi, qj, qk = quad[:, 0], quad[:, 1], quad[:, 2], quad[:, 3]
    two_s = 2.0 / (quad * quad).sum(-1)
    rot_mat = torch.zeros(bs, 3, 3).to(quad.get_device())
    rot_mat[:, 0, 0] = 1 - two_s * (qj ** 2 + qk ** 2)
    rot_mat[:, 0, 1] = two_s * (qi * qj - qk * qr)
    rot_mat[:, 0, 2] = two_s * (qi * qk + qj * qr)
    rot_mat[:, 1, 0] = two_s * (qi * qj + qk * qr)
    rot_mat[:, 1, 1] = 1 - two_s * (qi ** 2 + qk ** 2)
    rot_mat[:, 1, 2] = two_s * (qj * qk - qi * qr)
    rot_mat[:, 2, 0] = two_s * (qi * qk - qj * qr)
    rot_mat[:, 2, 1] = two_s * (qj * qk + qi * qr)
    rot_mat[:, 2, 2] = 1 - two_s * (qi ** 2 + qj ** 2)
    return rot_mat

def get_tensor_from_frame(RT, Tquad=False):
    """
    Convert transformation matrix to quaternion and translation.

    """
    gpu_id = -1
    if type(RT) == torch.Tensor:
        if RT.get_device() != -1:  # RT.get_device() == -1  on  cpu
            RT = RT.detach().cpu()
            gpu_id = RT.get_device()
        RT = RT.numpy()
    from mathutils import Matrix
    R, T = RT[:3, :3], RT[:3, 3]
    rot = Matrix(R)
    quad = rot.to_quaternion()
    if Tquad:
        tensor = np.concatenate([T, quad], 0)
    else:
        tensor = np.concatenate([quad, T], 0)
    tensor = torch.from_numpy(tensor).float()
    if gpu_id != -1:
        tensor = tensor.to(gpu_id)
    return tensor

def CalPoseError(pose_optimize, pose_gt):
    
    if isinstance(pose_optimize, torch.Tensor):
        pose_optimize = pose_optimize.detach().cpu()
    if isinstance(pose_gt, torch.Tensor):
        pose_gt = pose_gt.detach().cpu()

    R_gt = pose_gt[:3, :3]
    t_gt = pose_gt[:3, 3]

    R = pose_optimize[:3, :3]
    t = pose_optimize[:3, 3]

    e_t = np.linalg.norm(t_gt - t, axis=0)
    cos = np.clip((np.trace(np.dot(R_gt.T, R)) - 1) / 2, -1., 1.)
    e_R = np.rad2deg(np.abs(np.arccos(cos)))
    return e_t, e_R


def get_sample_uv_with_grad(H0, H1, W0, W1, n, image):
    """
    Sample n uv coordinates from an image region H0..H1, W0..W1
    image (numpy.ndarray): color image or estimated normal image

    """
    intensity = rgb2gray(image.cpu().numpy())
    grad_y = filters.sobel_h(intensity)
    grad_x = filters.sobel_v(intensity)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    img_size = (image.shape[0], image.shape[1])
    selected_index = np.argpartition(grad_mag, -5*n, axis=None)[-5*n:]
    indices_h, indices_w = np.unravel_index(selected_index, img_size)
    mask = (indices_h >= H0) & (indices_h < H1) & (
        indices_w >= W0) & (indices_w < W1)
    indices_h, indices_w = indices_h[mask], indices_w[mask]
    selected_index = np.ravel_multi_index(
        np.array((indices_h, indices_w)), img_size)
    samples = np.random.choice(
        range(0, indices_h.shape[0]), size=n, replace=False)

    return selected_index[samples]

@torch.no_grad()
def initial_filter(cell_size, init_points,  device):
    xyz_min, xyz_max = torch.min(init_points, dim=-2)[0], torch.max(init_points, dim=-2)[0]
    space_edge = torch.max(xyz_max - xyz_min) * 1.05
    xyz_mid = (xyz_max + xyz_min) / 2
    space_min = xyz_mid - space_edge / 2

    construct_vox_sz =torch.tensor(cell_size, device=device)
    vox_res = space_edge / construct_vox_sz

    xyz_shift = init_points - space_min[None, ...]
    _, inv_idx = torch.unique(torch.floor(xyz_shift / construct_vox_sz[None, ...]).to(torch.int32), dim=0, return_inverse=True)
    xyz_centroid = scatter_mean(init_points, inv_idx, dim=0)
    xyz_centroid_prop = xyz_centroid[inv_idx,:]
    xyz_residual = torch.norm(init_points - xyz_centroid_prop, dim=-1)

    _, min_idx = scatter_min(xyz_residual, inv_idx, dim=0)

    return min_idx

@torch.no_grad()
def growing_filter(cell_size, existing_points, grow_points, device):
    
    total_points = torch.cat([existing_points, grow_points], dim=0)
    xyz_min, xyz_max = torch.min(total_points, dim=-2)[0], torch.max(total_points, dim=-2)[0]
    space_edge = torch.max(xyz_max - xyz_min) * 1.05
    xyz_mid = (xyz_max + xyz_min) / 2
    space_min = xyz_mid - space_edge / 2

    construct_vox_sz =torch.tensor(cell_size, device=device)
    vox_res = space_edge / construct_vox_sz

    grow_xyz_shift = grow_points - space_min[None, ...]
    _, inv_idx = torch.unique(torch.floor(grow_xyz_shift / construct_vox_sz[None, ...]).to(torch.int32), dim=0, return_inverse=True)
    grow_xyz_centroid = scatter_mean(grow_points, inv_idx, dim=0)
    grow_xyz_centroid_prop = grow_xyz_centroid[inv_idx,:]
    grow_xyz_residual = torch.norm(grow_points - grow_xyz_centroid_prop, dim=-1)

    _, min_idx = scatter_min(grow_xyz_residual, inv_idx, dim=0)

    return min_idx


def matrix_to_axis_angle(rot):
    """
    :param rot: [N, 3, 3]
    :return:
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(rot))

def at_to_transform_matrix(rot, trans):
    """
    :param rot: axis-angle [bs, 3]
    :param trans: translation vector[bs, 3]
    :return: transformation matrix [b, 4, 4]
    """
    bs = rot.shape[0]
    T = torch.eye(4).to(rot)[None, ...].repeat(bs, 1, 1)
    R = axis_angle_to_matrix(rot)
    T[:, :3, :3] = R
    T[:, :3, 3] = trans
    return T

def se3_to_T(se3: torch.Tensor):
    return se3_exp_map(se3).permute(0, 2, 1)

def T_to_se3(transform_matrix: torch.Tensor):
    return se3_log_map(transform_matrix.permute(0, 2, 1))


def mask_in_image(pts, image_size, pad: int = 1):
    w, h = image_size
    image_size_ = torch.tensor([w - pad - 1, h - pad - 1]).to(pts)
    return torch.all((pts >= pad) & (pts <= image_size_), -1)


def get_pointcloud(depth, intrinsics, w2c, sampled_indices):
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # Compute indices of sampled pixels
    xx = (sampled_indices[:, 1] - CX)/FX
    yy = (sampled_indices[:, 0] - CY)/FY
    depth_z = depth[0, sampled_indices[:, 0], sampled_indices[:, 1]]

    # Initialize point cloud
    pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
    pts4 = torch.cat([pts_cam, torch.ones_like(pts_cam[:, :1])], dim=1)
    c2w = torch.inverse(w2c)
    pts = (c2w @ pts4.T).T[:, :3]

    # Remove points at camera origin
    A = torch.abs(torch.round(pts, decimals=4))
    B = torch.zeros((1, 3)).cuda().float()
    _, idx, counts = torch.cat([A, B], dim=0).unique(
        dim=0, return_inverse=True, return_counts=True)
    mask = torch.isin(idx, torch.where(counts.gt(1))[0])
    invalid_pt_idx = mask[:len(A)]
    valid_pt_idx = ~invalid_pt_idx
    pts = pts[valid_pt_idx]

    return pts


def keyframe_selection_overlap(gt_depth, w2c, intrinsics, keyframe_list, k, pixels=1600):
        """
        Select overlapping keyframes to the current camera observation.

        Args:
            gt_depth (tensor): ground truth depth image of the current frame.
            w2c (tensor): world to camera matrix (4 x 4).
            keyframe_list (list): a list containing info for each keyframe.
            k (int): number of overlapping keyframes to select.
            pixels (int, optional): number of pixels to sparsely sample 
                from the image of the current camera. Defaults to 1600.
        Returns:
            selected_keyframe_list (list): list of selected keyframe id.
        """
        # Radomly Sample Pixel Indices from valid depth pixels
        width, height = gt_depth.shape[2], gt_depth.shape[1]
        valid_depth_indices = torch.where(gt_depth[0] > 0)
        valid_depth_indices = torch.stack(valid_depth_indices, dim=1)
        indices = torch.randint(valid_depth_indices.shape[0], (pixels,))
        sampled_indices = valid_depth_indices[indices]

        # Back Project the selected pixels to 3D Pointcloud
        pts = get_pointcloud(gt_depth, intrinsics, w2c, sampled_indices)

        list_keyframe = []
        for keyframeid, keyframe in enumerate(keyframe_list):
            # Get the estimated world2cam of the keyframe
            est_w2c = keyframe['est_w2c']
            # Transform the 3D pointcloud to the keyframe's camera space
            pts4 = torch.cat([pts, torch.ones_like(pts[:, :1])], dim=1)
            transformed_pts = (est_w2c @ pts4.T).T[:, :3]
            # Project the 3D pointcloud to the keyframe's image space
            points_2d = torch.matmul(intrinsics, transformed_pts.transpose(0, 1))
            points_2d = points_2d.transpose(0, 1)
            points_z = points_2d[:, 2:] + 1e-5
            points_2d = points_2d / points_z
            projected_pts = points_2d[:, :2]
            # Filter out the points that are outside the image
            edge = 20
            mask = (projected_pts[:, 0] < width-edge)*(projected_pts[:, 0] > edge) * \
                (projected_pts[:, 1] < height-edge)*(projected_pts[:, 1] > edge)
            mask = mask & (points_z[:, 0] > 0)
            # Compute the percentage of points that are inside the image
            percent_inside = mask.sum()/projected_pts.shape[0]
            list_keyframe.append(
                {'id': keyframeid, 'percent_inside': percent_inside})

        # Sort the keyframes based on the percentage of points that are inside the image
        list_keyframe = sorted(
            list_keyframe, key=lambda i: i['percent_inside'], reverse=True)
        # Select the keyframes with percentage of points inside the image > 0
        selected_keyframe_list = [keyframe_dict['id']
                                  for keyframe_dict in list_keyframe if keyframe_dict['percent_inside'] > 0.0]
        selected_keyframe_list = list(np.random.permutation(
            np.array(selected_keyframe_list))[:k])

        return selected_keyframe_list

@torch.no_grad()
def cal_uncertainty(orig_uncertainty, num_related_pixels):
    
    mean_uncertainty = torch.div(orig_uncertainty, num_related_pixels+1e-5)
    sort_uncertainty, indices = torch.sort(mean_uncertainty, dim=0, descending=True)
    # sort_uncertainty, indices = torch.sort(orig_uncertainty, dim=0, descending=True)

    return sort_uncertainty, indices