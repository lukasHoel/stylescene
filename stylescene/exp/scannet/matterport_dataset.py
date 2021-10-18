import os
import cv2
import numpy as np
import torchvision
from PIL import Image
from os.path import join
from scannet.abstract_dataset import Abstract_Dataset
from tqdm.auto import tqdm
import torch


class MatterportDataset(Abstract_Dataset):

    sort_keys = {
        # an example for default naming scheme is 0e92a69a50414253a23043758f111cec_i0_0.jpg where i0_0 gives the order
        # an example that also works with default naming scheme is 5b9b2794954e4694a45fc424a8643081_i0_0.jpg.rendered_depth.npy
        "default": lambda x: [x.split(".")[0].split('_')[0], int(x.split(".")[0].split('_')[1][1]) * 100 + int(x.split(".")[0].split('_')[2])]
    }

    def __init__(self,
                 root_path,
                 transform_rgb=None,
                 transform_depth=None,
                 style_path=None,
                 resize=False,
                 resize_size=(256, 256),
                 region_index=0,
                 cache=False,
                 train=False,
                 verbose=False):

        self.region_index = region_index

        Abstract_Dataset.__init__(self,
                                  root_path=root_path,
                                  transform_rgb=transform_rgb,
                                  transform_depth=transform_depth,
                                  style_path=style_path,
                                  resize=resize,
                                  resize_size=resize_size,
                                  cache=cache,
                                  train=train,
                                  verbose=verbose)

    def get_scenes(self):
        return os.listdir(self.root_path)

    def get_colors(self, scene_path, extensions=["jpg", "png"]):
        """
        Return absolute paths to all colors images for the scene (sorted!)
        """
        color_path = join(scene_path, 'rendered', f'region_{self.region_index}', 'color')
        sort_key = MatterportDataset.sort_keys["default"]
        if not os.path.exists(color_path) or not os.path.isdir(color_path):
            return []

        colors = os.listdir(color_path)
        colors = [c for c in colors if any(c.endswith(x) for x in extensions)]
        colors = sorted(colors, key=sort_key)
        colors = [join(color_path, f) for f in colors]

        return colors

    def get_depth(self, scene_path):
        """
        Return absolute paths to all depth images for the scene (sorted!)
        """
        # load rendered opengl depth
        def load_rendered_depth(scene_path):
            uv_path = join(scene_path, "rendered", f'region_{self.region_index}', 'rendered_depth')

            if not os.path.exists(uv_path) or not os.path.isdir(uv_path):
                return []

            files = sorted(os.listdir(uv_path), key=MatterportDataset.sort_keys['default'])
            return [join(uv_path, f) for f in files if "npy" in f and 'depth' in f]
        rendered_depth_npy = load_rendered_depth(scene_path)

        # load original sensor depth
        depth_path = join(scene_path, 'rendered', f'region_{self.region_index}', 'depth')
        if not os.path.exists(depth_path) or not os.path.isdir(depth_path):
            self.rendered_depth = True
            return rendered_depth_npy

        depth = sorted(os.listdir(depth_path), key=MatterportDataset.sort_keys['default'])
        depth = [join(depth_path, f) for f in depth]

        # choose opengl depth, if sensor depth not available
        if len(depth) == 0:
            self.rendered_depth = True
            return rendered_depth_npy
        else:
            self.rendered_depth = False
            return depth

    def get_extrinsics(self, scene_path):
        """
        Return absolute paths to all extrinsic images for the scene (sorted!)
        """
        extrinsics_path = join(scene_path, "rendered", f'region_{self.region_index}', 'pose')

        if not os.path.exists(extrinsics_path) or not os.path.isdir(extrinsics_path):
            return []

        extrinsics = sorted(os.listdir(extrinsics_path), key=MatterportDataset.sort_keys['default'])
        extrinsics = [join(extrinsics_path, f) for f in extrinsics if 'intrinsic' not in f]

        return extrinsics

    def get_intrinsics(self, scene_path):
        """
        Return 3x3 numpy array as intrinsic K matrix for the scene and (W,H) image dimensions if available
        """
        intrinsics = np.identity(4, dtype=np.float32)
        w = 0
        h = 0
        intr_path = join(scene_path, 'rendered', f'region_{self.region_index}', 'pose')
        file = [join(intr_path, f) for f in os.listdir(intr_path) if ".intrinsics.txt" in f]
        if len(file) > 0:
            file = file[0]
            self.intrinsics_file = file
            with open(file) as f:
                lines = f.readlines()
                for i, l in enumerate(lines):
                    l = l.strip()
                    elems = l.split(' ')
                    if i < 3:
                        intrinsics[i][0] = float(elems[0])
                        intrinsics[i][1] = float(elems[1])
                        intrinsics[i][2] = float(elems[2])
                    elif i == 3:
                        w = int(elems[0])
                        h = int(elems[1])
                    else:
                        raise ValueError('index too large', i, lines)

        return intrinsics, (w,h)

    def load_extrinsics(self, idx):
        """
        load the extrinsics item from self.extrinsics

        :param idx: the item to load

        :return: the extrinsics as numpy array
        """

        extrinsics = open(self.extrinsics[idx], "r").readlines()
        extrinsics = [[float(item) for item in line.split(" ")] for line in extrinsics]
        extrinsics = np.array(extrinsics, dtype=np.float32)

        return extrinsics

    def load_depth(self, idx):
        file = self.depth_images[idx]
        if not self.rendered_depth:
            d = np.asarray(Image.open(file)) / 4000.0
        else:
            d = np.load(file)
            d = d[:, :, :1]  # only keep first channel, but slice it instead of '[:,:,0]' to keep the dim

        return d
