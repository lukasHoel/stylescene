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


def get_color_encoding(seg_classes):
    if seg_classes.lower() == 'nyu40':
        """Color palette for nyu40 labels """
        return OrderedDict([
            ('void', (0, 0, 0)),
            ('wall', (174, 199, 232)),
            ('floor', (152, 223, 138)),
            ('cabinet', (31, 119, 180)),
            ('bed', (255, 187, 120)),
            ('chair', (188, 189, 34)),
            ('sofa', (140, 86, 75)),
            ('table', (255, 152, 150)),
            ('door', (214, 39, 40)),
            ('window', (197, 176, 213)),
            ('bookshelf', (148, 103, 189)),
            ('picture', (196, 156, 148)),
            ('counter', (23, 190, 207)),
            ('blinds', (178, 76, 76)),
            ('desk', (247, 182, 210)),
            ('shelves', (66, 188, 102)),
            ('curtain', (219, 219, 141)),
            ('dresser', (140, 57, 197)),
            ('pillow', (202, 185, 52)),
            ('mirror', (51, 176, 203)),
            ('floormat', (200, 54, 131)),
            ('clothes', (92, 193, 61)),
            ('ceiling', (78, 71, 183)),
            ('books', (172, 114, 82)),
            ('refrigerator', (255, 127, 14)),
            ('television', (91, 163, 138)),
            ('paper', (153, 98, 156)),
            ('towel', (140, 153, 101)),
            ('showercurtain', (158, 218, 229)),
            ('box', (100, 125, 154)),
            ('whiteboard', (178, 127, 135)),
            ('person', (120, 185, 128)),
            ('nightstand', (146, 111, 194)),
            ('toilet', (44, 160, 44)),
            ('sink', (112, 128, 144)),
            ('lamp', (96, 207, 209)),
            ('bathtub', (227, 119, 194)),
            ('bag', (213, 92, 176)),
            ('otherstructure', (94, 106, 211)),
            ('otherfurniture', (82, 84, 163)),
            ('otherprop', (100, 85, 144)),
        ])
    elif seg_classes.lower() == 'scannet20':
        return OrderedDict([
            ('unlabeled', (0, 0, 0)),
            ('wall', (174, 199, 232)),
            ('floor', (152, 223, 138)),
            ('cabinet', (31, 119, 180)),
            ('bed', (255, 187, 120)),
            ('chair', (188, 189, 34)),
            ('sofa', (140, 86, 75)),
            ('table', (255, 152, 150)),
            ('door', (214, 39, 40)),
            ('window', (197, 176, 213)),
            ('bookshelf', (148, 103, 189)),
            ('picture', (196, 156, 148)),
            ('counter', (23, 190, 207)),
            ('desk', (247, 182, 210)),
            ('curtain', (219, 219, 141)),
            ('refrigerator', (255, 127, 14)),
            ('showercurtain', (158, 218, 229)),
            ('toilet', (44, 160, 44)),
            ('sink', (112, 128, 144)),
            ('bathtub', (227, 119, 194)),
            ('otherfurniture', (82, 84, 163)),
        ])


def nyu40_to_scannet20(label):
    """Remap a label image from the 'nyu40' class palette to the 'scannet20' class palette """

    # Ignore indices 13, 15, 17, 18, 19, 20, 21, 22, 23, 25, 26. 27. 29. 30. 31. 32, 35. 37. 38, 40
    # Because, these classes from 'nyu40' are absent from 'scannet20'. Our label files are in
    # 'nyu40' format, hence this 'hack'. To see detailed class lists visit:
    # http://kaldir.vc.in.tum.de/scannet_benchmark/labelids_all.txt ('nyu40' labels)
    # http://kaldir.vc.in.tum.de/scannet_benchmark/labelids.txt ('scannet20' labels)
    # The remaining labels are then to be mapped onto a contiguous ordering in the range [0,20]

    # The remapping array comprises tuples (src, tar), where 'src' is the 'nyu40' label, and 'tar' is the
    # corresponding target 'scannet20' label
    remapping = [(0, 0), (13, 0), (15, 0), (17, 0), (18, 0), (19, 0), (20, 0), (21, 0), (22, 0), (23, 0), (25, 0),
                 (26, 0), (27, 0),
                 (29, 0), (30, 0), (31, 0), (32, 0), (35, 0), (37, 0), (38, 0), (40, 0), (14, 13), (16, 14), (24, 15),
                 (28, 16), (33, 17),
                 (34, 18), (36, 19), (39, 20)]
    for src, tar in remapping:
        label[np.where(label == src)] = tar
    return label


def enet_weighing(dataloader, num_classes, c=1.02):
    """Computes class weights as described in the ENet paper:

        w_class = 1 / (ln(c + p_class)),

    where c is usually 1.02 and p_class is the propensity score of that
    class:

        propensity_score = freq_class / total_pixels.

    References: https://arxiv.org/abs/1606.02147

    Keyword arguments:
    - dataloader (``data.Dataloader``): A data loader to iterate over the
    dataset.
    - num_classes (``int``): The number of classes.
    - c (``int``, optional): AN additional hyper-parameter which restricts
    the interval of values for the weights. Default: 1.02.

    """
    class_count = 0
    total = 0
    print("Create class weights...")
    for batch in tqdm(dataloader):
        label = batch[1]
        label = label.cpu().numpy()

        # Flatten label
        flat_label = label.flatten()

        # Sum up the number of pixels of each class and the total pixel
        # counts for each label
        class_count += np.bincount(flat_label, minlength=num_classes)
        total += flat_label.size

    # Compute propensity score and then the weights for each class
    propensity_score = class_count / total

    class_weights = 1 / (np.log(c + propensity_score))

    class_weights[class_count == 0] = 0

    return class_weights


def median_freq_balancing(dataloader, num_classes):
    """Computes class weights using median frequency balancing as described
    in https://arxiv.org/abs/1411.4734:

        w_class = median_freq / freq_class,

    where freq_class is the number of pixels of a given class divided by
    the total number of pixels in images where that class is present, and
    median_freq is the median of freq_class.

    Keyword arguments:
    - dataloader (``data.Dataloader``): A data loader to iterate over the
    dataset.
    whose weights are going to be computed.
    - num_classes (``int``): The number of classes

    """
    class_count = 0
    total = 0
    print("Create class weights...")
    for _, label in tqdm(dataloader):
        label = label.cpu().numpy()

        # Flatten label
        flat_label = label.flatten()

        # Sum up the class frequencies
        bincount = np.bincount(flat_label, minlength=num_classes)

        # Create of mask of classes that exist in the label
        mask = bincount > 0
        # Multiply the mask by the pixel count. The resulting array has
        # one element for each class. The value is either 0 (if the class
        # does not exist in the label) or equal to the pixel count (if
        # the class exists in the label)
        total += mask * flat_label.size

        # Sum up the number of pixels found for each class
        class_count += bincount

    # Compute the frequency and its median
    freq = class_count / total
    med = np.median(freq)

    return med / freq


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


def reproject(cam2world_src, cam2world_tar, W, H, intrinsic, depth_src, depth_tar, color_tar, mask_tar):
    # get batch_size
    bs = mask_tar.shape[0]

    # calculate src2tar extrinsic matrix
    world2cam_tar = torch.inverse(cam2world_tar)
    src2tar = torch.transpose(torch.bmm(world2cam_tar, cam2world_src), 1, 2)

    # create meshgrid with image dimensions (== pixel coordinates of source image)
    y = torch.linspace(0, H - 1, H).type_as(color_tar).int()
    x = torch.linspace(0, W - 1, W).type_as(color_tar).int()
    xx, yy = torch.meshgrid(x, y)
    xx = torch.transpose(xx, 0, 1).repeat(bs, 1, 1)
    yy = torch.transpose(yy, 0, 1).repeat(bs, 1, 1)

    # get intrinsics and depth in correct format to match image dimensions
    fx = intrinsic[:,0,0].unsqueeze(1).unsqueeze(1).expand_as(xx)
    cx = intrinsic[:,0,2].unsqueeze(1).unsqueeze(1).expand_as(xx)
    fy = intrinsic[:,1,1].unsqueeze(1).unsqueeze(1).expand_as(yy)
    cy = intrinsic[:,1,2].unsqueeze(1).unsqueeze(1).expand_as(yy)
    depth_src = depth_src.squeeze()

    # inverse projection (K_inv) on pixel coordinates --> 3D point-cloud
    x = (xx - cx) / fx * depth_src
    y = (yy - cy) / fy * depth_src

    # combine each point into an (x,y,z,1) vector
    coords = torch.zeros(bs, H, W, 4).type_as(color_tar).float()
    coords[:, :, :, 0] = x
    coords[:, :, :, 1] = y
    coords[:, :, :, 2] = depth_src
    coords[:, :, :, 3] = 1

    # extrinsic view projection to target view
    coords = coords.view(bs, -1, 4)
    coords = torch.bmm(coords, src2tar)
    coords = coords.view(bs, H, W, 4)

    # projection (K) on 3D point-cloud --> pixel coordinates
    z_tar = coords[:, :, :, 2]
    x = coords[:, :, :, 0] / (1e-8 + z_tar) * fx + cx
    y = coords[:, :, :, 1] / (1e-8 + z_tar) * fy + cy

    # mask invalid pixel coordinates because of invalid source depth
    mask0 = (depth_src == 0)

    # mask invalid pixel coordinates after projection:
    # these coordinates are not visible in target view (out of screen bounds)
    mask1 = (x < 0) + (y < 0) + (x >= W - 1) + (y >= H - 1)

    # create 4 target pixel coordinates which map to the nearest integer coordinate
    # (left, top, right, bottom)
    lx = torch.floor(x).float()
    ly = torch.floor(y).float()
    rx = (lx + 1).float()
    ry = (ly + 1).float()

    def make_grid(x, y):
        """
        converts pixel coordinates from [0..W] or [0..H] to [-1..1] and stacks them together.
        :param x: x pixel coordinates with shape NxHxW
        :param y: y pixel coordinates with shape NxHxW
        :return: (x,y) pixel coordinate grid with shape NxHxWx2
        """
        x = (2.0 * x / W) - 1.0
        y = (2.0 * y / H) - 1.0
        grid = torch.stack((x, y), dim=3)
        return grid

    # combine to (x,y) pixel coordinates: (top-left, ..., bottom-right)
    ll = make_grid(lx, ly)
    lr = make_grid(lx, ry)
    rl = make_grid(rx, ly)
    rr = make_grid(rx, ry)

    # calculate difference between depth in target view after reprojection and gt depth in target view
    z_tar = z_tar.unsqueeze(1)
    sample_z1 = torch.abs(z_tar - torch.nn.functional.grid_sample(depth_tar, ll,
                                                                  mode="nearest",
                                                                  padding_mode='border',
                                                                  align_corners=True))
    sample_z2 = torch.abs(z_tar - torch.nn.functional.grid_sample(depth_tar, lr,
                                                                  mode="nearest",
                                                                  padding_mode='border',
                                                                  align_corners=True))
    sample_z3 = torch.abs(z_tar - torch.nn.functional.grid_sample(depth_tar, rl,
                                                                  mode="nearest",
                                                                  padding_mode='border',
                                                                  align_corners=True))
    sample_z4 = torch.abs(z_tar - torch.nn.functional.grid_sample(depth_tar, rr,
                                                                  mode="nearest",
                                                                  padding_mode='border',
                                                                  align_corners=True))

    # mask invalid pixel coordinates because of too high difference in depth
    mask2 = torch.minimum(torch.minimum(sample_z1, sample_z2), torch.minimum(sample_z3, sample_z4)) > 0.1
    mask2 = mask2.int().squeeze()

    # combine all masks
    mask_remap = (1 - (mask0 + mask1 + mask2 > 0).int()).float().unsqueeze(1)

    # create (x,y) pixel coordinate grid with reprojected float coordinates
    map_x = x.float()
    map_y = y.float()
    map = make_grid(map_x, map_y)

    # warp target rgb/mask to the new pixel coordinates based on the reprojection
    # also mask the results
    color_tar_to_src = torch.nn.functional.grid_sample(color_tar, map,
                                                                  mode="bilinear",
                                                                  padding_mode='border',
                                                                  align_corners=True)
    mask_tar = mask_tar.float().unsqueeze(1)
    mask = torch.nn.functional.grid_sample(mask_tar, map,
                                            mode="bilinear",
                                            padding_mode='border',
                                            align_corners=True)
    mask = (mask > 0.99) * mask_remap
    mask = mask.bool()
    color_tar_to_src *= mask

    return color_tar_to_src, mask.squeeze(1)