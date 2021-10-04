import numpy as np

from PIL import Image
from os.path import join
from tqdm.auto import tqdm

import torch
import torchvision
import random

from scannet.scannet_dataset import ScanNetDataset
from scannet.abstract_dataset import Abstract_DataModule


class ScanNet_Single_Scene_DataModule(Abstract_DataModule):
    def __init__(self,
                 root_path: str,
                 batch_size: int = 32,
                 num_workers: int = 1,
                 transform_rgb=None,
                 transform_label=None,
                 transform_uv=None,
                 crop=False,
                 crop_size=(-1,-1),
                 crop_random=True,
                 resize=False,
                 resize_size=(256, 256),
                 test_crop=False,
                 test_crop_random=False,
                 load_uvs=False,
                 load_stylized_images=False,
                 stylized_images_path=None,
                 stylized_images_sort_key="default",
                 load_uv_mipmap=False,
                 load_uv_pyramid=False,
                 pyramid_levels=5,
                 min_pyramid_depth=0.25,
                 min_pyramid_height=32,
                 test_noise=False,
                 noise_suffix="_noise",
                 scene=None,
                 min_images=1000,
                 max_images=-1,
                 shuffle: bool = False,
                 sampler_mode: str = "random",
                 index_repeat: int = 1,
                 paired=False,
                 paired_index_threshold=10,
                 split: list = [0.8, 0.2, 0.2],
                 split_mode: str = "skip",
                 nearest_neighbors: int = 0,
                 depth_scale_std_factor: float = 1,
                 depth_scale_mean_factor: float = 0,
                 ignore_unlabeled: bool = True,
                 class_weight: bool = True,
                 create_instance_map: bool = False,
                 verbose: bool = False,
                 cache: bool = False):

        Abstract_DataModule.__init__(self,
                                     dataset=ScanNet_Single_House_Dataset,
                                     root_path=join(root_path, "train/images"),
                                     transform_rgb=transform_rgb,
                                     transform_label=transform_label,
                                     transform_uv=transform_uv,
                                     crop=crop,
                                     crop_size=crop_size,
                                     crop_random=crop_random,
                                     resize=resize,
                                     resize_size=resize_size,
                                     test_crop=test_crop,
                                     test_crop_random=test_crop_random,
                                     load_uvs=load_uvs,
                                     test_noise=test_noise,
                                     noise_suffix=noise_suffix,
                                     use_scene_filter=True,
                                     scene=scene,
                                     min_images=min_images,
                                     max_images=max_images,
                                     verbose=verbose,
                                     batch_size=batch_size,
                                     num_workers=num_workers,
                                     cache=cache,
                                     ignore_unlabeled=ignore_unlabeled,
                                     class_weight=class_weight,
                                     create_instance_map=create_instance_map,
                                     shuffle=shuffle,
                                     sampler_mode=sampler_mode,
                                     index_repeat=index_repeat,
                                     paired=paired,
                                     paired_index_threshold=paired_index_threshold,
                                     depth_scale_std_factor=depth_scale_std_factor,
                                     depth_scale_mean_factor=depth_scale_mean_factor,
                                     split=split,
                                     split_mode=split_mode,
                                     nearest_neighbors=nearest_neighbors)

        self.load_stylized_images = load_stylized_images
        self.stylized_images_path = stylized_images_path
        self.stylized_images_sort_key = stylized_images_sort_key
        self.load_uv_mipmap = load_uv_mipmap
        self.load_uv_pyramid = load_uv_pyramid
        self.min_pyramid_depth = min_pyramid_depth
        self.min_pyramid_height = min_pyramid_height
        self.pyramid_levels = pyramid_levels

    def after_create_dataset(self, d, root_path, crop, crop_random, noise):
        if isinstance(d, ScanNetDataset):
            d.set_stylized_image_mode(self.load_stylized_images, self.stylized_images_path, self.stylized_images_sort_key)
            d.set_uv_mipmap_mode(self.load_uv_mipmap)
            d.set_uv_pyramid_mode(self.load_uv_pyramid, self.min_pyramid_depth, self.min_pyramid_height)
            d.set_pyramid_levels(self.pyramid_levels)
            d.create_data()


class ScanNet_Single_House_Dataset(ScanNetDataset):

    def __init__(self,
                 root_path,
                 scene=None,
                 min_images=1000,
                 max_images=-1,
                 transform_rgb=None,
                 transform_label=None,
                 transform_uv=None,
                 crop=False,
                 crop_size=(-1,-1),
                 crop_random=True,
                 resize=False,
                 resize_size=(256, 256),
                 load_noise=False,
                 noise_suffix="_noise",
                 load_uvs=False,
                 load_uv_pyramid=False,
                 pyramid_levels=5,
                 min_pyramid_depth=0.25,
                 min_pyramid_height=32,
                 load_stylized_images=False,
                 create_instance_map=False,
                 depth_scale_std_factor=1,
                 depth_scale_mean_factor=0,
                 cache=False,
                 verbose=False):

        self.input_scene = scene
        self.min_images = min_images
        self.max_images = max_images
        self.create_instance_map = create_instance_map

        ScanNetDataset.__init__(self,
                                root_path=root_path,
                                transform_rgb=transform_rgb,
                                transform_label=transform_label,
                                transform_uv=transform_uv,
                                crop=crop,
                                crop_size=crop_size,
                                crop_random=crop_random,
                                resize=resize,
                                resize_size=resize_size,
                                load_uvs=load_uvs,
                                load_uv_pyramid=load_uv_pyramid,
                                min_pyramid_depth=min_pyramid_depth,
                                min_pyramid_height=min_pyramid_height,
                                pyramid_levels=pyramid_levels,
                                load_noise=load_noise,
                                noise_suffix=noise_suffix,
                                load_stylized_images=load_stylized_images,
                                depth_scale_std_factor=depth_scale_std_factor,
                                depth_scale_mean_factor=depth_scale_mean_factor,
                                create_instance_map=False,  # only create it afterwards when needed because we shrinken scenes anyways after this
                                cache=cache,
                                verbose=verbose)

    def create_data(self):
        self.scene_dict = self.parse_scenes()[-1]
        self.rgb_images, self.label_images, self.instance_images, self.extrinsics, self.intrinsics, self.intrinsic_image_sizes, self.depth_images, self.uv_maps, self.angle_maps, self.size, self.scene = self.get_scene(self.input_scene, self.min_images, self.max_images)

        print(f"Using scene: {self.scene}. Input was: {self.input_scene}")

        # use this to finally set the self.uvs_npy, self.angle_npy and self.rendered_depth attributes correctly to the state of the chosen scene
        self.get_depth(join(self.root_path, self.scene))
        self.get_uvs(join(self.root_path, self.scene))
        self.get_angles(join(self.root_path, self.scene))

        if self.create_instance_map:
            self.instance_map, self.inverse_instance_map = self.get_instance_map()
        else:
            self.instance_map = None
            self.inverse_instance_map = None

    def get_scene(self, scene, min_images, max_images):
        items = self.get_scene_items(scene)
        if self.in_range(min_images, max_images, items):
            return self.parse_scene(scene)
        else:
            return self.find_house(min_images, max_images)

    def get_scene_items(self, scene):
        if scene is None:
            return None
        elif scene not in self.scene_dict:
            return 0
        else:
            return self.scene_dict[scene]["items"]

    def in_range(self, min, max, value):
        return (value is not None) and (min == -1 or value >= min) and (max == -1 or value <= max)

    def parse_scene(self, scene):
        h = self.scene_dict[scene]
        if self.load_uvs:
            return h["color"], h["label"], h["instance"], h["extrinsics"], h["intrinsics"], h["image_size"], h["depth"], h["uv_map"], h["angle_map"], len(h["color"]), scene
        else:
            return h["color"], h["label"], h["instance"], h["extrinsics"], h["intrinsics"], h["image_size"], h["depth"], [], [], len(h["color"]), scene

    def find_house(self, min_images, max_images):
        max = -1
        min = -1
        scenes = [s for s in self.scene_dict.keys()]
        random.shuffle(scenes)
        if self.verbose:
            scenes = tqdm(scenes)
            print(f"Searching for a house with more than {min_images} images")
        for h in scenes:
            size = self.get_scene_items(h)
            if max == -1 or size > max:
                max = size
            if min == -1 or size < min:
                min = size
            if self.in_range(min_images, max_images, size):
                if self.verbose:
                    print(f"Using scene '{h}' which has {size} images")
                return self.parse_scene(h)
        raise ValueError(f"No scene found with {min_images} <= i <= {max_images} images. Min/Max available: {min}/{max}")