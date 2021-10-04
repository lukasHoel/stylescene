import torchvision
import torch
import numpy as np
import os

from torch.utils.data import Dataset, SubsetRandomSampler, SequentialSampler, Sampler
from torch.utils.data import DataLoader

from PIL import Image

from scannet.utils import get_image_transform, get_color_encoding, get_euler_angles, enet_weighing, unproject

from tqdm.auto import tqdm

import os.path
from os.path import join

import pytorch_lightning as pl

from typing import Optional

from abc import ABC, abstractmethod

import random

from typing import Union
import cv2

import torch.nn.functional as F


class Abstract_Dataset(Dataset, ABC):

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
                 load_labels=False,
                 load_instances=False,
                 depth_scale_std_factor=1,
                 depth_scale_mean_factor=0,
                 cache=False,
                 verbose=False):
        # save all constructor arguments
        self.transform_rgb = transform_rgb
        self.transform_label = get_image_transform(transform_label)
        self.transform_uv = transform_uv
        self.crop = crop
        self.crop_size = crop_size
        self.crop_random = crop_random
        self.resize = resize
        self.resize_size = resize_size
        if isinstance(resize_size, int):
            # self.resize_size = (resize_size, resize_size)
            pass
        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size)
        self.verbose = verbose
        self.root_path = root_path
        self.load_uvs = load_uvs
        self.load_labels = load_labels
        self.load_instances = load_instances
        self.load_noise = load_noise
        self.create_instance_map = create_instance_map
        self.use_cache = cache
        self.cache = {}
        self.label_to_color = list(get_color_encoding("nyu40").items())
        self.label_to_color = list(get_color_encoding("nyu40").items())
        self.num_classes = len(self.label_to_color)
        self.noise_suffix = noise_suffix
        self.depth_scale_std_factor = depth_scale_std_factor
        self.depth_scale_mean_factor = depth_scale_mean_factor

        from stylization.vgg_models import encoder3
        self.enc_net = encoder3()
        self.enc_net.load_state_dict(torch.load("stylization/vgg_r31.pth"))

        self.name = 'Scannet'
        self.logging_rate = 1

        # create data for this dataset
        self.create_data()

        if self.use_cache:
            print("Preloading all into cache")
            for i in tqdm(range(self.size)):
                self.__getitem__(i)
            print("Finished preloading")

    def create_data(self):
        self.rgb_images, self.label_images, self.instance_images, self.uv_maps, self.angle_maps, self.extrinsics, self.intrinsics, self.intrinsic_image_sizes, self.depth_images, self.size, self.scene_dict = self.parse_scenes()

        if self.create_instance_map:
            self.instance_map, self.inverse_instance_map = self.get_instance_map()
        else:
            self.instance_map = None
            self.inverse_instance_map = None

    def get_class_count(self):
        return self.num_classes

    def get_instance_count(self):
        if self.instance_map is None:
            self.instance_map, self.inverse_instance_map = self.get_instance_map()

        return len(self.instance_map.keys())

    def get_instance_map(self):
        instance_map = {}
        inverse_instance_map = {}
        counter = 0
        print("Creating instance map...")
        for i in tqdm(range(self.size)):
            item = self.__getitem__(i)
            instance = item[2]
            instances = [i.detach().cpu().numpy().item() for i in torch.unique(instance)]

            for i in instances:
                if i not in instance_map:
                    instance_map[i] = {
                        "idx": counter,
                        "priority": 1,
                        "name": counter
                    }
                    inverse_instance_map[counter] = {
                        "instance": i,
                        "priority": 1,
                        "name": counter
                    }
                    counter += 1

        return instance_map, inverse_instance_map

    @abstractmethod
    def get_scenes(self):
        """
        Return names to all scenes for the dataset.
        """
        pass

    @abstractmethod
    def get_colors(self, scene_path):
        """
        Return absolute paths to all colors images for the scene (sorted!)
        """
        pass

    @abstractmethod
    def get_depth(self, scene_path):
        """
        Return absolute paths to all depth images for the scene (sorted!)
        """
        pass

    @abstractmethod
    def get_labels(self, scene_path):
        """
        Return absolute paths to all label images for the scene (sorted!)
        """
        pass

    @abstractmethod
    def get_instances(self, scene_path):
        """
        Return absolute paths to all instance images for the scene (sorted!)
        """
        pass

    @abstractmethod
    def get_extrinsics(self, scene_path):
        """
        Return absolute paths to all extrinsic images for the scene (sorted!)
        """
        pass

    @abstractmethod
    def get_uvs(self, scene_path):
        """
        Return absolute paths to all uvmap images for the scene (sorted!)
        """
        pass

    @abstractmethod
    def get_angles(self, scene_path):
        """
        Return absolute paths to all angle images for the scene (sorted!)
        """
        pass

    def get_intrinsics(self, scene_path):
        """
        Return 3x3 numpy array as intrinsic K matrix for the scene and (W,H) image dimensions if available
        """
        return np.identity(4, dtype=np.float32), (0, 0)

    def parse_scenes(self):
        rgb_images = []
        depth_images = []
        label_images = []
        instance_images = []
        uv_maps = []
        angle_maps = []
        extrinsics_matrices = []
        intrinsic_matrices = []
        intrinsic_image_sizes = []
        scene_dict = {}

        scenes = self.get_scenes()
        if self.verbose:
            print("Collecting images...")
            scenes = tqdm(scenes)

        for scene in scenes:
            scene_path = join(self.root_path, scene)
            if os.path.isdir(scene_path):
                scene_dict[scene] = {
                    "path": scene_path,
                    "items": 0,
                }

                colors = self.get_colors(scene_path)

                if self.load_labels:
                    labels = self.get_labels(scene_path)
                else:
                    labels = colors

                if self.load_instances:
                    instances = self.get_instances(scene_path)
                else:
                    instances = colors

                depth = self.get_depth(scene_path)

                extrinsics = self.get_extrinsics(scene_path)
                intrinsics, image_size = self.get_intrinsics(scene_path)
                intrinsics = [intrinsics for i in range(len(colors))]
                image_size = [image_size for i in range(len(colors))]

                if self.load_uvs:
                    uvs = self.get_uvs(scene_path)
                    angles = self.get_angles(scene_path)
                else:
                    uvs = []
                    angles = []

                if len(colors) > 0 and len(colors) == len(labels) and len(labels) == len(instances) and \
                        (len(instances) == len(uvs) or not self.load_uvs or (
                                hasattr(self, 'uv_pyramid') and self.uv_pyramid and all(len(instances) == len(uv) for uv in uvs))) \
                        and (len(instances) == len(angles) or not self.load_uvs) and len(instances) == len(extrinsics):
                    rgb_images.extend(colors)
                    depth_images.extend(depth)
                    label_images.extend(labels)
                    instance_images.extend(instances)

                    if hasattr(self, 'uv_pyramid') and self.uv_pyramid:
                        if len(uv_maps) == 0:
                            uv_maps.extend(uvs)
                        else:
                            for x,y in zip(uv_maps, uvs):
                                x.extend(y)
                    else:
                        uv_maps.extend(uvs)
                    angle_maps.extend(angles)
                    extrinsics_matrices.extend(extrinsics)
                    intrinsic_matrices.extend(intrinsics)
                    intrinsic_image_sizes.extend(image_size)
                    scene_dict[scene]["items"] = len(colors)
                    scene_dict[scene]["color"] = colors
                    scene_dict[scene]["depth"] = depth
                    scene_dict[scene]["label"] = labels
                    scene_dict[scene]["instance"] = instances
                    scene_dict[scene]["extrinsics"] = extrinsics
                    scene_dict[scene]["intrinsics"] = intrinsics
                    scene_dict[scene]["image_size"] = image_size

                    if self.load_uvs:
                        scene_dict[scene]["uv_map"] = uvs
                        scene_dict[scene]["angle_map"] = angles
                elif self.verbose:
                    print(
                        f"Scene {scene_path} rendered incomplete --> is skipped. colors: {len(colors)}, labels: {len(labels)}, instances: {len(instances)}, uvs: {len(uvs)}, angles: {len(angles)}, extr: {len(extrinsics)}")

        assert (len(rgb_images) == len(label_images))
        assert (len(label_images) == len(instance_images))
        assert (len(instance_images) == len(uv_maps) or not self.load_uvs or (hasattr(self, 'uv_pyramid') and self.uv_pyramid and all(len(instances) == len(uv) for uv in uvs)))
        assert (len(instance_images) == len(angle_maps) or not self.load_uvs)
        assert (len(instance_images) == len(extrinsics_matrices))

        return rgb_images, label_images, instance_images, uv_maps, angle_maps, extrinsics_matrices, intrinsic_matrices, intrinsic_image_sizes, depth_images, len(
            rgb_images), scene_dict

    def get_labels_in_image(self, label_image):
        if isinstance(label_image, torch.Tensor):
            label_image = torchvision.transforms.ToPILImage()(label_image.cpu().int())

        if not isinstance(label_image, Image.Image):
            raise ValueError(f"image must be of type {torch.Tensor} or {Image.Image}, but was: {label_image}")

        labels = {}
        for i in range(label_image.size[0]):
            for j in range(label_image.size[1]):
                pixel = label_image.getpixel((i, j))
                if pixel not in labels:
                    # is ordered dict: [pixel] accesses the pixel-th item, [0] accesses the key (which is the label)
                    labels[pixel] = self.label_to_color[pixel][0]

        return labels

    def get_label_masks(self, label_image):
        if isinstance(label_image, torch.Tensor):
            label_image = torchvision.transforms.ToPILImage()(label_image.cpu().int())

        if not isinstance(label_image, Image.Image):
            raise ValueError(f"image must be of type {torch.Tensor} or {Image.Image}, but was: {label_image}")

        masks = {}
        for i in range(label_image.size[0]):
            for j in range(label_image.size[1]):
                pixel = label_image.getpixel((i, j))
                if pixel not in masks:
                    masks[pixel] = np.zeros((label_image.size[1], label_image.size[0]))
                masks[pixel][j, i] = 1

        return masks

    def get_instance_masks(self, instance_image, global_instance_id=True):
        if isinstance(instance_image, torch.Tensor):
            instance_image = torchvision.transforms.ToPILImage()(instance_image.cpu().int())

        if not isinstance(instance_image, Image.Image):
            raise ValueError(f"image must be of type {torch.Tensor} or {Image.Image}, but was: {instance_image}")

        masks = {}
        for i in range(instance_image.size[0]):
            for j in range(instance_image.size[1]):
                pixel = instance_image.getpixel((i, j))

                if global_instance_id:
                    if self.instance_map is None:
                        raise ValueError(f"Cannot use global_instance_id when no instance_map was created!")
                    pixel = self.instance_map[pixel]["global_id"]

                if pixel not in masks:
                    masks[pixel] = np.zeros((instance_image.size[1], instance_image.size[0]))
                masks[pixel][j, i] = 1

        return masks

    def get_color_image(self, label_image):
        if isinstance(label_image, torch.Tensor):
            label_image = torchvision.transforms.ToPILImage()(label_image.cpu().int())

        if not isinstance(label_image, Image.Image):
            raise ValueError(f"image must be of type {torch.Tensor} or {Image.Image}, but was: {label_image}")

        color_image = Image.new("RGB", label_image.size)
        colored_pixels = color_image.load()

        for i in range(color_image.size[0]):
            for j in range(color_image.size[1]):
                pixel = label_image.getpixel((i, j))
                # is ordered dict: [pixel] accesses the pixel-th item, [1] accesses the value (which is the color)
                colored_pixels[i, j] = self.label_to_color[pixel][1]

        return color_image

    def get_nearest_neighbors(self, train_dataset, train_indices, test_indices, n=1, weights=[1.0, 1.0], verbose=False):
        # get all train extrinsics
        train_extrinsics = [train_dataset.__getitem__(i, only_cam=True)[0] for i in train_indices]
        train_r = [get_euler_angles(r) for r in train_extrinsics]
        train_t = [e[:3, 3] for e in train_extrinsics]

        # get all test extrinsics
        test_extrinsics = [self.__getitem__(i, only_cam=True)[0] for i in test_indices]
        test_r = [get_euler_angles(r) for r in test_extrinsics]
        test_t = [e[:3, 3] for e in test_extrinsics]
        test = zip(test_indices, test_r, test_t)
        if verbose:
            print(
                f"Calculating {n} nearest neighbors for {len(test_indices)} test images within {len(train_indices)} train images")
            test = tqdm(test, total=len(test_indices))

        # dict of lists: i-th entry contains the "n nearest neighbors list" for the test index i
        # an entry in the "n nearest neighbors list" has the form {"i": train_index, "d": distance to test_index}
        neighbors = {i: [] for i in test_indices}

        if n > 0:
            for test_idx, r1, t1 in test:
                for train_idx, r2, t2 in zip(train_indices, train_r, train_t):
                    # calculate distance (weighted between R and T)
                    dr = torch.sum((r2 - r1) ** 2)
                    dt = torch.sum((t2 - t1) ** 2)
                    d = weights[0] * dr + weights[1] * dt

                    # search insertion index
                    insert_index = 0
                    for neighbor in neighbors[test_idx]:
                        if neighbor["d"] > d:
                            break
                        insert_index += 1

                    # only insert if it is one of the n shortest distances
                    if insert_index < n:
                        # add neighbor at the correct index
                        neighbors[test_idx].insert(insert_index, {"i": train_idx, "d": d})

                        # remove neighbors that are no longer among the shortest n
                        neighbors[test_idx] = neighbors[test_idx][:n]

        return neighbors

    @abstractmethod
    def load_extrinsics(self, idx):
        """
        load the extrinsics item from self.extrinsics

        :param idx: the item to load

        :return: the extrinsics as numpy array
        """
        pass

    def load_intrinsics(self, idx):
        """
        load the intrinsics item from self.intrinsics

        :param idx: the item to load

        :return: the intrinsics as numpy array
        """
        return self.intrinsics[idx]

    @abstractmethod
    def load_uvmap(self, idx, pyramid_idx=0):
        """
        load the uvmap item from self.uv_maps

        :param idx: the item to load

        :return: the uvmap as PIL image or numpy array
        """
        pass

    @abstractmethod
    def load_anglemap(self, idx):
        """
        load the angle_map item from self.angle_maps

        :param idx: the item to load

        :return: the angle_map as PIL image or numpy array
        """
        pass

    @abstractmethod
    def calculate_mask(self, uvmap, depth=None):
        """
        calculate the uvmap mask item from uvmap (valid values == 1)

        :param idx: the uvmap from which to calculate the mask

        :return: the mask as PIL image
        """
        pass

    @abstractmethod
    def calculate_depth_level(self, uv, depth, transform_uv, transform_depth):
        """
        calculate the depth level per pixel from uv map and depth images.

        :param uv: the uvmap as numpy array
        :param depth: the depth as PIL image

        :return: the depth level as numpy array
        """
        pass

    def prepare_getitem(self, idx):
        """
        Implementations can prepare anything necessary for loading this idx, i.e. load a .hdf5 file
        :param idx:
        :return:
        """
        pass

    def finalize_getitem(self, idx):
        """
        Implementations can finalize anything necessary after loading this idx, i.e. close a .hdf5 file
        :param idx:
        :return:
        """
        pass

    def calculate_fair_index_repeat(self, w=1024, h=1024, k=500, max_repeat=50):
        '''
        calculate for each item how often it should be repeated that each texel can be seen roughly as often as others.

        @param w: texture size width
        @param h: texture size height
        @param k: use top-k texels that appear the least to calculate index_repeat per frame
        @param max_repeat: maximum number of repeats per frame
        @return: list of self.size where each item signals how often it should be repeated
        '''
        return [1 for i in range(self.size)]

    def load_rgb(self, idx):
        return Image.open(self.rgb_images[idx])

    def load_depth(self, idx):
        return Image.open(self.depth_images[idx])

    def load_label(self, idx):
        return Image.open(self.label_images[idx])

    def load_instance(self, idx):
        return Image.open(self.instance_images[idx])

    def modify_intrinsics_matrix(self, intrinsics, intrinsics_image_size, rgb_image_size):
        if intrinsics_image_size != rgb_image_size:
            intrinsics = np.array(intrinsics)
            intrinsics[0, 0] = (intrinsics[0, 0] / intrinsics_image_size[0]) * rgb_image_size[0]
            intrinsics[1, 1] = (intrinsics[1, 1] / intrinsics_image_size[1]) * rgb_image_size[1]
            intrinsics[0, 2] = (intrinsics[0, 2] / intrinsics_image_size[0]) * rgb_image_size[0]
            intrinsics[1, 2] = (intrinsics[1, 2] / intrinsics_image_size[1]) * rgb_image_size[1]

        return intrinsics

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        if item not in self.cache:
            self.prepare_getitem(item)

            extrinsics = self.load_extrinsics(item)
            extrinsics = torch.from_numpy(extrinsics)
            intrinsics = self.load_intrinsics(item)
            intrinsics = torch.from_numpy(intrinsics)

            rgb = self.load_rgb(item)
            depth = self.load_depth(item)

            if self.resize:
                if isinstance(self.resize_size, int):
                    w, h = rgb.size
                    h_new = self.resize_size
                    w_new = round(w * h_new / h)
                    resize_size = (w_new, h_new)
                else:
                    resize_size = self.resize_size

                rgb = rgb.resize(resize_size)
                if isinstance(depth, np.ndarray):
                    depth = cv2.resize(depth, resize_size, interpolation=cv2.INTER_LINEAR)
                else:
                    depth = depth.resize(resize_size)

            # fix intrinsics to resized item
            intrinsics = self.modify_intrinsics_matrix(intrinsics, self.intrinsic_image_sizes[item], rgb.size)
            intrinsics = torch.from_numpy(intrinsics)

            if self.transform_rgb:
                rgb = self.transform_rgb(rgb)

            if self.transform_label:
                depth = self.transform_label(depth)

            # TODO create
            # points2: 3d projection of each pixel, use the unproject function. then also normalize it by biggest coordinate value in xyz
            # feats: vgg features of the color image
            # points: reprojection to another view, sampling map for each pixel in (-1, 1). TODO can just make uniform sample map for test time
            # tgt: color image
            # src: zeros_like(tgt)
            # style: style image

            # raise ValueError(points.shape, emb.shape)  # ((226, 273, 490, 3), (226, 256, 72, 124))

            feats = self.enc_net(rgb.unsqueeze(0))
            H2, W2 = feats.shape[2:4]
            feats = feats.permute(1, 0, 2, 3)
            feats = feats.reshape(256, -1)

            points2 = unproject(extrinsics.unsqueeze(0), intrinsics.unsqueeze(0), depth.unsqueeze(0))
            H1, W1 = points2.shape[1:3]
            y = [int(round(y)) for y in np.array(list(range(H2))) / (H2 - 1) * (H1 - 1)]
            x = [int(round(x)) for x in np.array(list(range(W2))) / (W2 - 1) * (W1 - 1)]
            points2 = np.take(points2, y, axis=1)
            points2 = np.take(points2, x, axis=2)

            points2 = points2[:, :, :, :3]  # only need xyz
            points2 = points2.reshape(-1, 3)

            # TODO FIXME points: simple uniform sampling map, if we want to train this and use reprojection, we have to use reprojected sampling map!
            h, w = rgb.shape[1:]
            w_range = torch.arange(0, w, dtype=torch.float) / (w - 1.0) * 2.0 - 1.0
            h_range = torch.arange(0, h, dtype=torch.float) / (h - 1.0) * 2.0 - 1.0
            v, u = torch.meshgrid(h_range, w_range)
            uv_id = torch.stack([u, v, depth.squeeze()], 2)
            points = uv_id
            H1, W1 = points.shape[0:2]
            y = [int(round(y)) for y in np.array(list(range(H2))) / (H2 - 1) * (H1 - 1)]
            x = [int(round(x)) for x in np.array(list(range(W2))) / (W2 - 1) * (W1 - 1)]
            points = np.take(points, y, axis=0)
            points = np.take(points, x, axis=1)
            points = points.reshape(-1, 3)

            style = Image.open("/home/hoellein/datasets/styles/3style/14-2.jpg")
            style = self.transform_rgb(style)
            # TODO: if eval style has another dimension with multiple style images.
            # TODO: if train this is not present
            style = style.unsqueeze(0)

            result = {
                "tgt": rgb,
                "src": torch.zeros_like(rgb),
                "style": style,
                "feats": feats,
                "points": points,
                "points2": points2
            }

            if self.use_cache:
                self.cache[item] = result

            self.finalize_getitem(item)
            return result
        else:
            return self.cache[item]


class Abstract_DataModule(pl.LightningDataModule):
    split_modes = [
        "folder",  # we are given three different root_paths to train/val/test subdirectories
        "skip",  # leave out every n-th image from the train_val set s.t. we have split[2] test images
        "sequential",  # the last split[2] percent images are the test images
        "noise",  # use an extra noise folder at test time + use all images for train/val (split[2] argument must be 0)
    ]

    sampler_modes = [
        "random",
        # uses SubsetRandomSampler (might be in addition to the self.shuffle argument -> initially shuffled + other order every epoch)
        "sequential",
        # uses SequentialSampler (might be in addition to the self.shuffle argument -> initially shuffled + same order every epoch)
        "repeat",
        # uses custom RepeatedSampler and the self.index_repeat argument to sequentially iterate and repeat each train-item self.index_repeat times (might be in addition to self.shuffle argument --> initially shuffled + same repeated order every epoch)
        "repeat_fair"
        # uses custom RepeatedSampler to sequentially iterate and repeat each train-item as often that each texel is seen the same amount of time roughly
    ]

    def __init__(self,
                 dataset,
                 root_path: Union[str, dict],
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
                 test_noise=False,
                 noise_suffix="_noise",
                 use_scene_filter=False,
                 scene=None,
                 min_images=1000,
                 max_images=-1,
                 shuffle: bool = False,
                 sampler_mode: str = "random",
                 index_repeat: int = 1,
                 split: list = [0.8, 0.2, 0.0],
                 split_mode: str = "noise",
                 paired=False,
                 paired_index_threshold=10,
                 nearest_neighbors: int = 0,
                 depth_scale_std_factor: float = 1,
                 depth_scale_mean_factor: float = 0,
                 ignore_unlabeled: bool = True,
                 class_weight: bool = True,
                 create_instance_map: bool = False,
                 verbose: bool = False,
                 cache: bool = False):
        super().__init__()

        self.dataset_class = dataset
        self.root_path = root_path
        self.transform_rgb = transform_rgb
        self.transform_label = transform_label
        self.transform_uv = transform_uv
        self.crop = crop
        self.crop_size = crop_size
        self.resize = resize
        self.resize_size = resize_size
        self.crop_random = crop_random
        self.test_crop = test_crop
        self.test_crop_random = test_crop_random
        self.load_uvs = load_uvs
        self.test_noise = test_noise
        self.verbose = verbose
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache = cache
        self.ignore_unlabeled = ignore_unlabeled
        self.class_weight = class_weight
        self.create_instance_map = create_instance_map
        self.shuffle = shuffle
        self.sampler_mode = sampler_mode
        self.index_repeat = index_repeat
        self.split = split
        self.split_mode = split_mode
        self.paired = paired
        self.paired_index_threshold = paired_index_threshold
        self.nearest_neighbors = nearest_neighbors
        self.noise_suffix = noise_suffix
        self.depth_scale_std_factor = depth_scale_std_factor
        self.depth_scale_mean_factor = depth_scale_mean_factor

        self.use_scene_filter = use_scene_filter
        self.scene = scene  # the input scene
        self.selected_scene = None  # the scene after creating the first dataset (should be the same scene for all other creations)
        self.min_images = min_images
        self.max_images = max_images

    def create_dataset(self, root_path, crop, crop_random, noise) -> Abstract_Dataset:
        if not self.use_scene_filter:
            d = self.dataset_class(root_path=root_path,
                                   transform_rgb=self.transform_rgb,
                                   transform_label=self.transform_label,
                                   transform_uv=self.transform_uv,
                                   crop=crop,
                                   crop_size=self.crop_size,
                                   crop_random=crop_random,
                                   resize=self.resize,
                                   resize_size=self.resize_size,
                                   load_noise=noise,
                                   noise_suffix=self.noise_suffix,
                                   load_uvs=self.load_uvs,
                                   create_instance_map=self.create_instance_map,
                                   depth_scale_std_factor=self.depth_scale_std_factor,
                                   depth_scale_mean_factor=self.depth_scale_mean_factor,
                                   verbose=self.verbose,
                                   cache=self.cache)
        else:
            d = self.dataset_class(root_path=root_path,
                                   scene=self.selected_scene if self.selected_scene else self.scene,
                                   min_images=self.min_images,
                                   max_images=self.max_images,
                                   transform_rgb=self.transform_rgb,
                                   transform_label=self.transform_label,
                                   transform_uv=self.transform_uv,
                                   crop=crop,
                                   crop_size=self.crop_size,
                                   crop_random=crop_random,
                                   resize=self.resize,
                                   resize_size=self.resize_size,
                                   load_noise=noise,
                                   noise_suffix=self.noise_suffix,
                                   load_uvs=self.load_uvs,
                                   create_instance_map=self.create_instance_map,
                                   depth_scale_std_factor=self.depth_scale_std_factor,
                                   depth_scale_mean_factor=self.depth_scale_mean_factor,
                                   verbose=self.verbose,
                                   cache=self.cache)
            self.selected_scene = d.scene
            d.input_scene = d.scene

        self.after_create_dataset(d, root_path, crop, crop_random, noise)

        if self.paired:
            d = PairedDataset(d, self.paired_index_threshold)

        return d

    @abstractmethod
    def after_create_dataset(self, d, root_path, crop, crop_random, noise):
        pass

    def setup(self, stage: Optional[str] = None):
        # create datasets based on the specified path and further arguments
        if isinstance(self.root_path, dict):
            train_path = self.root_path["train"]
            val_path = self.root_path["val"]
            test_path = self.root_path["test"]

            self.train_dataset = self.create_dataset(train_path, self.crop, self.crop_random, False)
            self.val_dataset = self.create_dataset(val_path, self.crop, self.crop_random, False)
            self.test_dataset = self.create_dataset(test_path, self.test_crop, self.test_crop_random, self.test_noise)
        else:
            self.train_dataset = self.create_dataset(self.root_path, self.crop, self.crop_random, False)
            self.val_dataset = self.create_dataset(self.root_path, self.crop, self.crop_random, False)
            self.test_dataset = self.create_dataset(self.root_path, self.test_crop, self.test_crop_random,
                                                    self.test_noise)

        # num classes as defined in the dataset
        self.num_classes = self.train_dataset.num_classes if not self.paired else self.train_dataset.dataset.num_classes

        # create train/val/test split from the loaded datasets based on the split_mode
        if self.split_mode == "folder":
            self.train_indices = [i for i in range(self.train_dataset.__len__())]
            self.val_indices = [i for i in range(self.val_dataset.__len__())]
            self.test_indices = [i for i in range(self.test_dataset.__len__())]

            if self.shuffle:
                np.random.shuffle(self.train_indices)
                np.random.shuffle(self.val_indices)

        else:
            if isinstance(self.root_path, dict):
                raise ValueError(
                    f"Cannot use multiple root_path arguments (train/val/test) when split_mode is not 'folder'!")

            # create train/val/test split
            len = self.train_dataset.__len__()
            indices = [i for i in range(len)]
            train_split = int(self.split[0] * len)

            if self.split_mode == "sequential":
                val_split = int((self.split[0] + self.split[1]) * len)
                train_val_indices = indices[:val_split]
                test_indices = indices[val_split:]

            elif self.split_mode == "skip":
                test_count = int(len * self.split[2])
                train_val_count = len - test_count

                if train_val_count > test_count:
                    test_every_nth = len // test_count
                    test_indices = indices[::test_every_nth]
                    train_val_indices = [i for i in indices]
                    del train_val_indices[::test_every_nth]
                else:
                    train_val_every_nth = len // train_val_count
                    train_val_indices = indices[::train_val_every_nth]
                    test_indices = [i for i in indices]
                    del test_indices[::train_val_every_nth]

            elif self.split_mode == "noise":
                # we use a different self.test_dataset that has equal amount of indices, but other uvmaps (with noise)
                train_val_indices = indices
                test_indices = indices.copy()

            else:
                raise ValueError(
                    f"Unsupported split_mode: {self.split_mode}. Supported are: {Abstract_DataModule.split_modes}")

            if self.shuffle:
                np.random.shuffle(train_val_indices)

            self.train_indices = train_val_indices[:train_split]
            self.val_indices = train_val_indices[train_split:]
            self.test_indices = test_indices

        # class weights calculated over all data in train set (classes that are not present at all will have weight of 0)
        if self.class_weight:
            self.class_weights = enet_weighing(
                DataLoader(self.train_dataset if not self.paired else self.train_dataset.dataset,
                           batch_size=self.batch_size, num_workers=self.num_workers),
                self.num_classes)
        else:
            self.class_weights = np.ones(self.num_classes)

        # convert to Tensor
        self.class_weights = torch.from_numpy(self.class_weights).float()

        # set the class_weight of the "void" label to 0 --> the void label is the first one, see utils.get_color_encoding()
        if self.ignore_unlabeled:
            self.class_weights[0] = 0.0

        # all indices that have a zero weight should be ignored
        self.ignore_index = tuple((self.class_weights == 0.0).nonzero().view(-1).numpy())

        if self.verbose:
            print(f"Class weights: {self.class_weights}")
            print(f"Ignore Index: {self.ignore_index}")

    def train_dataloader(self):
        if self.sampler_mode == "sequential":
            sampler = SequentialSampler(self.train_dataset)
        elif self.sampler_mode == "random":
            sampler = SubsetRandomSampler(self.train_indices)
        elif self.sampler_mode == "repeat":
            sampler = RepeatingSampler(self.train_indices, self.index_repeat)
        elif self.sampler_mode == "repeat_fair":
            repeats = self.train_dataset.calculate_fair_index_repeat(1024, 1024)
            sampler = RepeatingSampler(self.train_indices, repeats)
        else:
            raise ValueError(f"Unsupported sampler mode: {self.sampler_mode}")
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, sampler=sampler)

    def val_dataloader(self):
        if self.sampler_mode == "sequential":
            sampler = SequentialSampler(self.val_dataset)
        elif self.sampler_mode == "random":
            sampler = SubsetRandomSampler(self.val_indices)
        elif self.sampler_mode == "repeat" or self.sampler_mode == "repeat_fair":
            # validation does not need to be repeated
            sampler = RepeatingSampler(self.val_indices, 1)
        else:
            raise ValueError(f"Unsupported sampler mode: {self.sampler_mode}")
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, sampler=sampler)

    def test_dataloader(self):
        if self.verbose:
            print(f"Test dataset uses noisy trajectory: {self.test_noise}")

        dataset = NearestNeighborDataset(test_dataset=self.test_dataset.dataset if self.paired else self.test_dataset,
                                         train_dataset=self.train_dataset.dataset if self.paired else self.train_dataset,
                                         train_indices=self.train_indices,
                                         test_indices=self.test_indices,
                                         n=self.nearest_neighbors,
                                         verbose=self.verbose)

        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers)


class RepeatingSampler(Sampler):
    def __init__(self, indices, index_repeat):
        super().__init__(indices)
        if isinstance(index_repeat, int):
            self.indices = [item for item in indices for i in range(index_repeat)]
        elif isinstance(index_repeat, list):
            self.indices = [item for item in indices for i in range(index_repeat[item])]
        else:
            raise ValueError('unsupported index_repeat type', index_repeat)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class NearestNeighborDataset(Dataset):
    """
    Returns the item along with all nearest neighbor items in a train set.
    """

    def __init__(self,
                 test_dataset: Abstract_Dataset,
                 train_dataset: Abstract_Dataset,
                 test_indices,
                 train_indices,
                 n=1,
                 weights=[1.0, 1.0],
                 verbose=False):
        super().__init__()

        self.test_dataset = test_dataset
        self.test_indices = test_indices
        self.train_dataset = train_dataset
        self.train_indices = train_indices
        self.size = len(test_indices)
        self.neighbors = self.test_dataset.get_nearest_neighbors(train_dataset, train_indices, test_indices,
                                                                 n, weights, verbose)

    def __getitem__(self, item):
        index = self.test_indices[item]
        test_item = self.test_dataset[index]
        neighbors = self.neighbors[index]

        # we laod the train neighbor from the test dataset, the assumption here is:
        # - the i-th item in the test_dataset is the i-th item in the train_dataset
        # - the difference will be only in the uvmaps, as the ones from the test_dataset contains noise if noise should be loaded
        # - but this is acceptable because we are not interested in the uvmap in following computations that use this dataset
        # - we cannot use the train_dataset as that might do some form of (random) cropping
        # - it could be argued that using the train_dataset is still preferably because the cropping was also used during real training
        #   but still cropping could be random and not the same as during training so we might as well show the complete image (i.e. what is retrieved from he test dataset)
        neighbors = [self.test_dataset[n["i"]] for n in neighbors]

        return test_item, neighbors

    def __len__(self):
        return self.size


class PairedDataset(Dataset):
    def __init__(self,
                 dataset: Abstract_Dataset,
                 index_threshold=10):
        super().__init__()

        if dataset.crop:
            raise ValueError(
                "Cropping should be disabled when using paired_dataset because reprojections might be all-zero in that case")

        self.dataset = dataset
        self.index_threshold = index_threshold

    def sample_pair(self, i, max_index, min_index=0):
        start = max(min_index, i - self.index_threshold)
        end = min(max_index, i + self.index_threshold)
        pair_index = random.choice([j for j in range(start, end) if j != i])
        return pair_index

    def __getitem__(self, item):
        item_a = self.dataset[item]
        item_b = self.dataset[self.sample_pair(item, self.dataset.__len__())]

        return item_a, item_b

    def __len__(self):
        return self.dataset.__len__()
