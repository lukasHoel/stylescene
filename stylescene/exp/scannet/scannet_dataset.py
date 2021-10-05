# taken from: https://github.com/krrish94/ENet-ScanNet/blob/master/data/scannet.py

import os
import numpy as np
from PIL import Image
from os.path import join
from scannet.abstract_dataset import Abstract_Dataset


class ScanNetDataset(Abstract_Dataset):

    orig_sizes = {
        # in format (h, w)
        "rgb": (240, 320),
        "label": (240, 320),
        "uv": (480, 640)
    }

    stylized_images_sort_key = {
        "default": lambda x: int(x.split(".")[0]),
    }

    def __init__(self,
                 root_path,
                 transform_rgb=None,
                 transform_depth=None,
                 style_path=None,
                 resize=False,
                 resize_size=(256, 256),
                 cache=False,
                 verbose=False):

        Abstract_Dataset.__init__(self,
                                  root_path=root_path,
                                  style_path=style_path,
                                  transform_rgb=transform_rgb,
                                  transform_depth=transform_depth,
                                  resize=resize,
                                  resize_size=resize_size,
                                  cache=cache,
                                  verbose=verbose)

    def get_scenes(self):
        return os.listdir(self.root_path)

    def get_colors(self, scene_path, extensions=["jpg", "png"]):
        """
        Return absolute paths to all colors images for the scene (sorted!)
        """
        color_path = join(scene_path, "color")
        sort_key = ScanNetDataset.stylized_images_sort_key["default"]
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

        # load original sensor depth
        depth_path = join(scene_path, "depth")
        if not os.path.exists(depth_path) or not os.path.isdir(depth_path):
            return []

        depth = sorted(os.listdir(depth_path), key=lambda x: int(x.split(".")[0]))
        depth = [join(depth_path, f) for f in depth]

        return depth

    def get_extrinsics(self, scene_path):
        """
        Return absolute paths to all extrinsic images for the scene (sorted!)
        """
        extrinsics_path = join(scene_path, "pose")

        if not os.path.exists(extrinsics_path) or not os.path.isdir(extrinsics_path):
            return []

        extrinsics = sorted(os.listdir(extrinsics_path), key=lambda x: int(x.split(".")[0]))
        extrinsics = [join(extrinsics_path, f) for f in extrinsics]

        return extrinsics

    def get_intrinsics(self, scene_path):
        """
        Return 3x3 numpy array as intrinsic K matrix for the scene and (W,H) image dimensions if available
        """
        intrinsics = np.identity(4, dtype=np.float32)
        w = 0
        h = 0
        file = [join(scene_path, f) for f in os.listdir(scene_path) if ".txt" in f]
        if len(file) == 1:
            file = file[0]
            self.intrinsics_file = file
            with open(file) as f:
                lines = f.readlines()
                for l in lines:
                    l = l.strip()
                    if "fx_color" in l:
                        fx = float(l.split(" = ")[1])
                        intrinsics[0,0] = fx
                    if "fy_color" in l:
                        fy = float(l.split(" = ")[1])
                        intrinsics[1,1] = fy
                    if "mx_color" in l:
                        mx = float(l.split(" = ")[1])
                        intrinsics[0,2] = mx
                    if "my_color" in l:
                        my = float(l.split(" = ")[1])
                        intrinsics[1,2] = my
                    if "colorWidth" in l:
                        w = int(l.split(" = ")[1])
                    if "colorHeight" in l:
                        h = int(l.split(" = ")[1])

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
        d = np.asarray(Image.open(file)) / 1000.0
        return d
