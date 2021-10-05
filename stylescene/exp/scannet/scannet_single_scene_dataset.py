import os.path

from tqdm.auto import tqdm

import random

from scannet.scannet_dataset import ScanNetDataset


class ScanNet_Single_House_Dataset(ScanNetDataset):

    def __init__(self,
                 root_path,
                 style_path=None,
                 scene=None,
                 min_images=1000,
                 max_images=-1,
                 transform_rgb=None,
                 transform_depth=None,
                 train=False,
                 resize=False,
                 resize_size=(256, 256),
                 cache=False,
                 verbose=False):

        self.input_scene = scene
        self.min_images = min_images
        self.max_images = max_images

        ScanNetDataset.__init__(self,
                                root_path=root_path,
                                style_path=style_path,
                                transform_rgb=transform_rgb,
                                transform_depth=transform_depth,
                                train=train,
                                resize=resize,
                                resize_size=resize_size,
                                cache=cache,
                                verbose=verbose)

        style_name = os.path.basename(style_path)
        import datetime
        self.name = self.__class__.__name__ + "_" + scene + "_" + style_name + "_" + str(datetime.datetime.now())

    def create_data(self):
        self.scene_dict = self.parse_scenes()[-1]
        self.rgb_images, self.extrinsics, self.intrinsics, self.intrinsic_image_sizes, self.depth_images, \
        self.size, self.scene = self.get_scene(self.input_scene, self.min_images, self.max_images)

        print(f"Using scene: {self.scene}. Input was: {self.input_scene}")

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
        return h["color"], h["extrinsics"], h["intrinsics"], h["image_size"], h["depth"], len(h["color"]), scene

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