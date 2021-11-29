import os
from PIL import Image
import numpy as np
from collections import OrderedDict

from tqdm.auto import tqdm

import torchvision
import torch


def to_numpy(x):
    x_ = np.array(x)
    x_ = x_.astype(np.float32)
    return x_


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def get_euler_angles(R):
    sy = torch.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    singular = sy < 1e-6

    if not singular:
        x = torch.atan2(R[2, 1], R[2, 2])
        y = torch.atan2(-R[2, 0], sy)
        z = torch.atan2(R[1, 0], R[0, 0])
    else:
        x = torch.atan2(-R[1, 2], R[1, 1])
        y = torch.atan2(-R[2, 0], sy)
        z = 0

    return torch.tensor([x, y, z], dtype=x.dtype)


def get_image_transform(transform):
    # fix for this issue: https://github.com/pytorch/vision/issues/2194
    if transform is not None and isinstance(transform, torchvision.transforms.Compose) and (transform.transforms[-1], torchvision.transforms.ToTensor):
        transform = torchvision.transforms.Compose([
            *transform.transforms[:-1],
            torchvision.transforms.Lambda(to_numpy),
            torchvision.transforms.ToTensor()
        ])
    elif isinstance(transform, torchvision.transforms.ToTensor):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Lambda(to_numpy),
            torchvision.transforms.ToTensor()
        ])

    return transform


def unproject(cam2world, intrinsic, depth):
    # get dimensions
    bs, _, H, W = depth.shape

    # create meshgrid with image dimensions (== pixel coordinates of source image)
    y = torch.linspace(0, H - 1, H).type_as(depth).int()
    x = torch.linspace(0, W - 1, W).type_as(depth).int()
    xx, yy = torch.meshgrid(x, y)
    xx = torch.transpose(xx, 0, 1).repeat(bs, 1, 1)
    yy = torch.transpose(yy, 0, 1).repeat(bs, 1, 1)

    # get intrinsics and depth in correct format to match image dimensions
    fx = intrinsic[:, 0, 0].unsqueeze(1).unsqueeze(1).expand_as(xx)
    cx = intrinsic[:, 0, 2].unsqueeze(1).unsqueeze(1).expand_as(xx)
    fy = intrinsic[:, 1, 1].unsqueeze(1).unsqueeze(1).expand_as(yy)
    cy = intrinsic[:, 1, 2].unsqueeze(1).unsqueeze(1).expand_as(yy)
    depth = depth.squeeze()

    # inverse projection (K_inv) on pixel coordinates --> 3D point-cloud
    x = (xx - cx) / fx * depth
    y = (yy - cy) / fy * depth

    # combine each point into an (x,y,z,1) vector
    coords = torch.zeros(bs, H, W, 4).type_as(depth).float()
    coords[:, :, :, 0] = x
    coords[:, :, :, 1] = y
    coords[:, :, :, 2] = depth
    coords[:, :, :, 3] = 1

    # extrinsic view projection to target view
    coords = coords.view(bs, -1, 4)
    coords = torch.bmm(coords, cam2world)
    coords = coords.view(bs, H, W, 4)

    return coords


def reproject(cam2world_tar, intrinsic, coords, H, W):
    # add homogenous 4D coordinate
    coords = torch.cat([coords, torch.ones_like(coords[:, :1])], dim=1)
    # add batch dim
    coords = coords.unsqueeze(0)

    # calculate src2tar extrinsic matrix
    world2cam_tar = torch.inverse(cam2world_tar)
    #src2tar = torch.transpose(torch.bmm(world2cam_tar, cam2world_src), 1, 2)

    # get intrinsics and depth in correct format to match image dimensions
    fx = intrinsic[:,0,0]
    cx = intrinsic[:,0,2]
    fy = intrinsic[:,1,1]
    cy = intrinsic[:,1,2]

    # extrinsic view projection to target view
    coords = torch.bmm(coords, world2cam_tar)

    # projection (K) on 3D point-cloud --> pixel coordinates
    z_tar = coords[:, :, 2]
    x = coords[:, :, 0] / (1e-8 + z_tar) * fx + cx
    y = coords[:, :, 1] / (1e-8 + z_tar) * fy + cy

    def make_grid(x, y, d):
        """
        converts pixel coordinates from [0..W] or [0..H] to [-1..1] and stacks them together.
        :param x: x pixel coordinates with shape NxHxW
        :param y: y pixel coordinates with shape NxHxW
        :return: (x,y) pixel coordinate grid with shape NxHxWx2
        """
        x = (2.0 * x / W) - 1.0
        y = (2.0 * y / H) - 1.0
        grid = torch.stack((x, y, d), dim=2)
        return grid

    # create (x,y) pixel coordinate grid with reprojected float coordinates
    map_x = x.float()
    map_y = y.float()
    map = make_grid(map_x, map_y, z_tar)

    return map.squeeze().reshape(-1, 3)
