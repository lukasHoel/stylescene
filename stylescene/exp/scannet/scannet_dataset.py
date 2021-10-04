# taken from: https://github.com/krrish94/ENet-ScanNet/blob/master/data/scannet.py

import os
import cv2
import numpy as np
import torchvision
from PIL import Image
from os.path import join
from scannet.abstract_dataset import Abstract_Dataset, Abstract_DataModule
from tqdm.auto import tqdm
import torch


class ScanNet_DataModule(Abstract_DataModule):

    def __init__(self,
                 root_path: str,
                 batch_size: int = 32,
                 num_workers: int = 1,
                 transform_rgb=None,
                 transform_label=None,
                 transform_uv=None,
                 crop=False,
                 crop_size=(-1, -1),
                 crop_random=True,
                 resize=False,
                 resize_size=(256, 256),
                 test_crop=False,
                 test_crop_random=False,
                 load_uvs=False,
                 paired=False,
                 paired_index_threshold=10,
                 test_noise=False,
                 noise_suffix="_noise",
                 shuffle: bool = False,
                 sampler_mode: str = "random",
                 index_repeat: int = 1,
                 nearest_neighbors: int = 0,
                 ignore_unlabeled: bool = True,
                 class_weight: bool = True,
                 create_instance_map: bool = False,
                 depth_scale_std_factor: float = 1,
                 depth_scale_mean_factor: float = 0,
                 verbose: bool = False,
                 cache: bool = False):

        root_paths = {
            "train": join(root_path, "train/images"),
            "val": join(root_path, "val/images"),
            "test": join(root_path, "val/images") # TODO real test directory
        }

        Abstract_DataModule.__init__(self,
                                     dataset=ScanNetDataset,
                                     root_path=root_paths,
                                     transform_rgb=transform_rgb,
                                     transform_label=transform_label,
                                     transform_uv=transform_uv,
                                     crop=crop,
                                     crop_size=crop_size,
                                     crop_random=crop_random,
                                     resize=resize,
                                     resize_size=resize_size,
                                     test_crop_random=test_crop_random,
                                     test_crop=test_crop,
                                     load_uvs=load_uvs,
                                     test_noise=test_noise,
                                     noise_suffix=noise_suffix,
                                     verbose=verbose,
                                     batch_size=batch_size,
                                     num_workers=num_workers,
                                     cache=cache,
                                     ignore_unlabeled=ignore_unlabeled,
                                     class_weight=class_weight,
                                     create_instance_map=create_instance_map,
                                     paired=paired,
                                     paired_index_threshold=paired_index_threshold,
                                     depth_scale_std_factor=depth_scale_std_factor,
                                     depth_scale_mean_factor=depth_scale_mean_factor,
                                     shuffle=shuffle,
                                     sampler_mode=sampler_mode,
                                     index_repeat=index_repeat,
                                     split_mode="folder",
                                     nearest_neighbors=nearest_neighbors)

    def after_create_dataset(self, d, root_path, crop, crop_random, noise):
        if isinstance(d, ScanNetDataset):
            d.create_data()


class ScanNetDataset(Abstract_Dataset):

    orig_sizes = {
        # in format (h, w)
        "rgb": (240, 320),
        "label": (240, 320),
        "uv": (480, 640)
    }

    stylized_images_sort_key = {
        "default": lambda x: int(x.split(".")[0]),
        "styled-X": lambda x: int(x.split(".")[0].split("-")[1]),
        "styled_X_X": lambda x: int(x.split(".")[0].split("_")[1])*100+int(x.split(".")[0].split("_")[2])
    }

    def __init__(self,
                 root_path,
                 transform_rgb=None,
                 transform_label=None,
                 transform_uv=None,
                 crop=False,
                 crop_size=(-1, -1),
                 crop_random=True,
                 resize=False,
                 resize_size=(256, 256),
                 create_instance_map=False,
                 load_noise=False,
                 noise_suffix="_noise",
                 load_uvs=False,
                 load_stylized_images=False,
                 stylized_images_path=None,
                 stylized_images_sort_key="default",
                 depth_scale_std_factor=1,
                 depth_scale_mean_factor=0,
                 load_uv_mipmap=False,
                 load_uv_pyramid=False,
                 pyramid_levels=5,
                 min_pyramid_depth=0.25,
                 min_pyramid_height=256,
                 cache=False,
                 verbose=False):

        self.set_stylized_image_mode(load_stylized_images, stylized_images_path, stylized_images_sort_key)
        self.set_uv_mipmap_mode(load_uv_mipmap)
        self.set_uv_pyramid_mode(load_uv_pyramid, min_pyramid_depth, min_pyramid_height)
        self.set_pyramid_levels(pyramid_levels)

        self.depth_weights = False

        Abstract_Dataset.__init__(self,
                                  root_path=root_path,
                                  transform_rgb=transform_rgb,
                                  transform_label=transform_label,
                                  transform_uv=transform_uv,
                                  crop=crop,
                                  crop_size=crop_size,
                                  crop_random=crop_random,
                                  resize=resize,
                                  resize_size=resize_size,
                                  load_noise=load_noise,
                                  noise_suffix=noise_suffix,
                                  create_instance_map=create_instance_map,
                                  depth_scale_std_factor=depth_scale_std_factor,
                                  depth_scale_mean_factor=depth_scale_mean_factor,
                                  load_uvs=load_uvs,
                                  cache=cache,
                                  verbose=verbose)

    def set_stylized_image_mode(self, stylized_images: bool, stylized_images_path: str, stylized_images_sort_key: str):
        self.stylized_images = stylized_images
        self.stylized_images_path = stylized_images_path
        if stylized_images_sort_key not in ScanNetDataset.stylized_images_sort_key.keys():
            raise ValueError(f"Unsupported stylized_images_sort_key: {stylized_images_sort_key}")
        self.stylized_images_sort_key = ScanNetDataset.stylized_images_sort_key[stylized_images_sort_key]

    def set_uv_mipmap_mode(self, uv_mipmap: bool):
        self.uv_mipmap = uv_mipmap

    def set_uv_pyramid_mode(self, load_uv_pyramid: bool,
                            min_pyramid_depth: float = 0.25,
                            min_pyramid_height: int = 256):
        self.uv_pyramid = load_uv_pyramid
        self.min_pyramid_depth = min_pyramid_depth
        self.min_pyramid_height = min_pyramid_height

    def set_pyramid_levels(self, pyramid_levels):
        self.pyramid_levels = pyramid_levels

    def set_mean_std_depth(self, mean_depth, std_depth):
        self.mean_depth = mean_depth
        self.std_depth = std_depth
        self.depth_weights = True

    def set_texel_statistics(self, statistics):
        self.min_angle_map, self.min_angle_map_texture, self.mean_texel_count, self.total_uv_mask, self.texel_count = statistics

    def get_texel_statistics(self):
        return [
            self.min_angle_map,
            self.min_angle_map_texture,
            self.mean_texel_count,
            self.total_uv_mask,
            self.texel_count
        ]

    def get_scenes(self):
        return os.listdir(self.root_path)

    def get_colors(self, scene_path, extensions=["jpg", "png"]):
        """
        Return absolute paths to all colors images for the scene (sorted!)
        """
        if not self.stylized_images:
            color_path = join(scene_path, "color")
            sort_key = ScanNetDataset.stylized_images_sort_key["default"]
        else:
            color_path = self.stylized_images_path
            sort_key = self.stylized_images_sort_key
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
            if self.load_noise:
                uv_path = join(scene_path, f"uv{self.noise_suffix}")
            else:
                uv_path = join(scene_path, "uv")

            if self.uv_mipmap:
                uv_path += "_mipmap"

            if not os.path.exists(uv_path) or not os.path.isdir(uv_path):
                return []

            files = sorted(os.listdir(uv_path), key=lambda x: int(x.split(".")[0]))
            return [join(uv_path, f) for f in files if "npy" in f and 'depth' in f]
        rendered_depth_npy = load_rendered_depth(scene_path)

        # load original sensor depth
        depth_path = join(scene_path, "depth")
        if not os.path.exists(depth_path) or not os.path.isdir(depth_path):
            return []

        depth = sorted(os.listdir(depth_path), key=lambda x: int(x.split(".")[0]))
        depth = [join(depth_path, f) for f in depth]

        # choose opengl depth, if sensor depth not available
        if len(depth) == 0:
            self.rendered_depth = True
            return rendered_depth_npy
        else:
            self.rendered_depth = False
            return depth

    def get_labels(self, scene_path):
        """
        Return absolute paths to all label images for the scene (sorted!)
        """
        label_path = join(scene_path, "label")

        if not os.path.exists(label_path) or not os.path.isdir(label_path):
            return []

        labels = sorted(os.listdir(label_path), key=lambda x: int(x.split(".")[0]))
        labels = [join(label_path, f) for f in labels]

        return labels

    def get_instances(self, scene_path):
        """
        Return absolute paths to all instance images for the scene (sorted!)
        """
        instance_path = join(scene_path, "instance")

        if not os.path.exists(instance_path) or not os.path.isdir(instance_path):
            return []

        instances = sorted(os.listdir(instance_path), key=lambda x: int(x.split(".")[0]))
        instances = [join(instance_path, f) for f in instances]

        return instances

    def get_extrinsics(self, scene_path):
        """
        Return absolute paths to all extrinsic images for the scene (sorted!)
        """
        if self.load_noise:
            extrinsics_path = join(scene_path, f"pose{self.noise_suffix}")
        else:
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

    def get_uvs(self, scene_path):
        """
        Return absolute paths to all uvmap images for the scene (sorted!)
        """
        def load_folder(folder):
            if not os.path.exists(folder) or not os.path.isdir(folder):
                return []

            files = sorted(os.listdir(folder), key=lambda x: int(x.split(".")[0]))
            uvs_npy = [join(folder, f) for f in files if "npy" in f and not 'angle' in f and not 'depth' in f]
            uvs_png = [join(folder, f) for f in files if "png" in f and not 'angle' in f and not 'depth' in f]
            return uvs_npy, uvs_png

        if not self.uv_pyramid:
            if self.load_noise:
                uv_path = join(scene_path, f"uv{self.noise_suffix}")
            else:
                uv_path = join(scene_path, "uv")

            if self.uv_mipmap:
                uv_path += "_mipmap"

            uvs_npy, uvs_png = load_folder(uv_path)

            if len(uvs_npy) >= len(uvs_png):
                self.uv_npy = True
                return uvs_npy
            else:
                self.uv_npy = False
                return uvs_png
        else:
            if self.uv_mipmap:
                raise NotImplementedError('combination of uv_pyramid and mipmap not supported!')

            pyramid_folders = [f for f in os.listdir(scene_path) if 'uv_' in f and is_float(f.split('_')[1])]
            if self.load_noise:
                pyramid_folders = [f for f in pyramid_folders if self.noise_suffix in f]
            else:
                pyramid_folders = [f for f in pyramid_folders if self.noise_suffix not in f]
            pyramid_folders = sorted(pyramid_folders, key=lambda x: float(x.split('_')[1]))

            # filter same folders, that only differ in suffix '.0', e.g. 256 vs 256.0
            # --> they should be the same, so just use one of them
            duplicates = []
            for i, f in enumerate(pyramid_folders):
                size = float(f.split('_')[1])
                if i < len(pyramid_folders) - 1:
                    next_size = float(pyramid_folders[i+1].split('_')[1])
                    if size == next_size:
                        duplicates.append(i + 1)
            pyramid_folders = [f for i, f in enumerate(pyramid_folders) if i not in duplicates]

            # save unfiltered level steps
            self.all_levels = np.array([float(x.split('_')[1]) for x in pyramid_folders])

            # filter minimum uv size
            pyramid_folders = [f for f in pyramid_folders if float(f.split('_')[1]) >= self.min_pyramid_height]

            # filter how many levels we want to have
            pyramid_folders = pyramid_folders[:self.pyramid_levels]

            # save the filtered level steps
            self.levels = np.array([float(x.split('_')[1]) for x in pyramid_folders])

            # return the paths to the levels
            pyramid_folders = [join(scene_path, f) for f in pyramid_folders]
            self.uv_npy = True
            return [load_folder(f)[0] for f in pyramid_folders]

    def get_angles(self, scene_path):
        """
        Return absolute paths to all angle images for the scene (sorted!)
        """
        def load_folder(folder):
            if not os.path.exists(folder) or not os.path.isdir(folder):
                return []

            files = sorted(os.listdir(folder), key=lambda x: int(x.split(".")[0]))
            angles_npy = [join(folder, f) for f in files if "npy" in f and 'angle' in f]
            angles_png = [join(folder, f) for f in files if "png" in f and 'angle' in f]
            return angles_npy, angles_png

        # always return angle maps without pyramid_levels, because we only need one angle map anyways
        if self.load_noise:
            uv_path = join(scene_path, f"uv{self.noise_suffix}")
        else:
            uv_path = join(scene_path, "uv")

        if self.uv_mipmap:
            uv_path += "_mipmap"

        angles_npy, angles_png = load_folder(uv_path)

        if len(angles_npy) >= len(angles_png):
            self.angles_npy = True
            return angles_npy
        else:
            self.angles_npy = False
            return angles_png

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

    def load_uvmap(self, idx, pyramid_idx=0):
        """
        load the uvmap item from self.uv_maps

        :param idx: the item to load

        :return: the uvmap as PIL image or numpy array
        """

        if self.uv_pyramid:
            file = self.uv_maps[pyramid_idx][idx]
        else:
            file = self.uv_maps[idx]
        if self.uv_npy:
            return np.load(file)
        else:
            return Image.open(file)

    def load_anglemap(self, idx):
        """
        load the angle_map item from self.angle_maps

        :param idx: the item to load

        :return: the angle_map as PIL image or numpy array
        """
        file = self.angle_maps[idx]
        if self.angles_npy:
            angle = np.load(file)
            angle = angle[:, :, :1]  # only keep first channel, but slice it instead of '[:,:,0]' to keep the dim
            return angle
        else:
            # if we implement this, we should only keep the first channel here as well...
            raise ValueError("not implemented")

    def load_depth(self, idx):
        file = self.depth_images[idx]
        if not self.rendered_depth:
            d = np.asarray(Image.open(file)) / 1000.0
        else:
            d = np.load(file)
            d = d[:, :, :1]  # only keep first channel, but slice it instead of '[:,:,0]' to keep the dim

        return d

    def calculate_mask(self, uvmap, depth=None):
        """
        calculate the uvmap mask item from uvmap (valid values == 1)

        :param idx: the uvmap from which to calculate the mask

        :return: the mask as PIL image
        """

        mask = np.asarray(uvmap)
        if self.uv_npy:
            mask_bool = mask[:, :, 0] != 0
            mask_bool += mask[:, :, 1] != 0
            mask = mask_bool
        else:
            mask = mask[:, :, 2] == 0

        if self.uv_pyramid and depth is not None:
            depth = cv2.resize(depth, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
            mask = mask * (depth > 0)

        mask = Image.fromarray(mask)

        return mask

    def calculate_texel_statistics(self, w, h):
        raise NotImplementedError()

    def calculate_fair_index_repeat(self, w=1024, h=1024, k=500, max_repeat=50):
        raise NotImplementedError()

    def calculate_depth_weight(self, w=0, h=0, global_calculation=True):
        raise NotImplementedError()

    def calculate_depth_weight_global(self):
        raise NotImplementedError()

    def calculate_depth_weight_per_texel(self, w, h):
        raise NotImplementedError()

    def calculate_depth_level(self, uv, depth, transform_uv, transform_depth):
        raise NotImplementedError()


def is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    # execute only if run as a script
    import matplotlib.pyplot as plt
    import torch

    transform_rgb = torchvision.transforms.Compose([
        #torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.ToTensor()
    ])

    transform_label = torchvision.transforms.Compose([
        #torchvision.transforms.Resize((256, 256), Image.NEAREST),
        torchvision.transforms.ToTensor()
    ])

    transform_uv = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: x * 2 - 1),  # from [0,1] to [-1,1]
        torchvision.transforms.Lambda(lambda x: x[:2])  # delete third channel, uv is present in first two only
    ])

    transform_label = torchvision.transforms.ToTensor()

    d = ScanNetDataset(root_path="/home/lukas/datasets/ScanNet/train/images/",
                       verbose=True,
                       transform_rgb=transform_rgb,
                       transform_label=transform_label,
                       transform_uv=transform_uv,
                       load_uvs=False)

    dm = ScanNet_DataModule(root_path="/home/lukas/datasets/ScanNet",
                            transform_uv=transform_uv,
                            transform_label=transform_label,
                            transform_rgb=transform_rgb,
                            class_weight=False)
    dm.setup()

    print(f"Dataset has {d.num_classes} distinct classes")

    for idx, (rgb, classes, instances, extrinsics, _) in enumerate(d):
        print("ITEM: ", idx)
        print("Extrinsics: ", extrinsics)
        labels = d.get_labels_in_image(classes)
        print(f"Item has {len(labels)} labels: {labels}")

        instance_masks = d.get_instance_masks(instances, False)
        # show instance masks
        """
        for k, instance_mask in instance_masks.items():
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(instance_mask)
            ax[1].imshow(torchvision.transforms.ToPILImage()(rgb))
            plt.show()
        """

        colored_label_image = d.get_color_image(classes)

        fig, ax = plt.subplots(1, 4)
        ax[0].imshow(torchvision.transforms.ToPILImage()(rgb))
        ax[1].imshow(torchvision.transforms.ToPILImage()(classes.int()))
        ax[2].imshow(colored_label_image)
        ax[3].imshow(torchvision.transforms.ToPILImage()(instances.int()))
        plt.show()
