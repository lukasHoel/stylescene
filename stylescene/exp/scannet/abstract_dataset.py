import torch
import numpy as np
import os

from torch.utils.data import Dataset

from PIL import Image

from scannet.utils import unproject, reproject, get_euler_angles, get_image_transform

from tqdm.auto import tqdm

import os.path
from os.path import join

from abc import ABC, abstractmethod

import cv2


class Abstract_Dataset(Dataset, ABC):

    def __init__(self,
                 root_path,
                 transform_rgb=None,
                 transform_depth=None,
                 style_path=None,
                 resize=False,
                 resize_size=(256, 256),
                 train=False,
                 cache=False,
                 verbose=False):
        # save all constructor arguments
        self.transform_rgb = transform_rgb
        self.transform_depth = get_image_transform(transform_depth)
        self.resize = resize
        self.resize_size = resize_size
        if isinstance(resize_size, int):
            # self.resize_size = (resize_size, resize_size)
            pass
        self.verbose = verbose
        self.root_path = root_path
        self.style_path = style_path
        self.use_cache = cache
        self.cache = {}
        self.train = train

        from stylization.vgg_models import encoder3
        self.enc_net = encoder3()
        self.enc_net.load_state_dict(torch.load("stylization/vgg_r31.pth"))

        import datetime
        self.name = self.__class__.__name__ + "_" + str(datetime.datetime.now())
        self.logging_rate = 1

        # create data for this dataset
        self.create_data()
        #self.calc_points()

        if self.use_cache:
            print("Preloading all into cache")
            for i in tqdm(range(self.size)):
                self.__getitem__(i)
            print("Finished preloading")

    def create_data(self):
        self.rgb_images, self.extrinsics, self.intrinsics, self.intrinsic_image_sizes, \
        self.depth_images, self.size, self.scene_dict = self.parse_scenes()

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
    def get_extrinsics(self, scene_path):
        """
        Return absolute paths to all extrinsic images for the scene (sorted!)
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
                depth = self.get_depth(scene_path)

                extrinsics = self.get_extrinsics(scene_path)
                intrinsics, image_size = self.get_intrinsics(scene_path)
                intrinsics = [intrinsics for i in range(len(colors))]
                image_size = [image_size for i in range(len(colors))]

                if len(colors) > 0 and len(colors) == len(extrinsics) and len(extrinsics) == len(depth):
                    rgb_images.extend(colors)
                    depth_images.extend(depth)
                    extrinsics_matrices.extend(extrinsics)
                    intrinsic_matrices.extend(intrinsics)
                    intrinsic_image_sizes.extend(image_size)
                    scene_dict[scene]["items"] = len(colors)
                    scene_dict[scene]["color"] = colors
                    scene_dict[scene]["depth"] = depth
                    scene_dict[scene]["extrinsics"] = extrinsics
                    scene_dict[scene]["intrinsics"] = intrinsics
                    scene_dict[scene]["image_size"] = image_size
                elif self.verbose:
                    print(
                        f"Scene {scene_path} rendered incomplete --> is skipped. colors: {len(colors)}, extr: {len(extrinsics)}, depth: {len(depth)}")

        return rgb_images, extrinsics_matrices, intrinsic_matrices, intrinsic_image_sizes, depth_images, len(rgb_images), scene_dict

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

    def load_rgb(self, idx):
        return Image.open(self.rgb_images[idx])

    def load_depth(self, idx):
        return Image.open(self.depth_images[idx])

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

    def calc_points(self):

        feats_list = []
        points_list = []

        for i in tqdm(range(self.size)):
            if i % 10 != 0:
                pass
                #continue

            rgb, depth, extrinsics, intrinsics = self.__getitem__(i, raw=True)

            feats = self.enc_net(rgb.unsqueeze(0)).detach().cpu()
            H2, W2 = feats.shape[2:4]
            feats = feats.permute(1, 0, 2, 3)
            feats = feats.reshape(256, -1)
            feats_list.append(feats)

            points2 = unproject(extrinsics.unsqueeze(0), intrinsics.unsqueeze(0), depth.unsqueeze(0)).detach().cpu()
            points_list.append(points2)

        self.feats = torch.cat(feats_list, dim=1)

        points = torch.cat(points_list, dim=0)
        H1, W1 = points.shape[1:3]
        y = [int(round(y)) for y in np.array(list(range(H2))) / (H2 - 1) * (H1 - 1)]
        x = [int(round(x)) for x in np.array(list(range(W2))) / (W2 - 1) * (W1 - 1)]
        points = np.take(points, y, axis=1)
        points = np.take(points, x, axis=2)
        self.points = points[:, :, :, :3].reshape(-1, 3)

        self.H2 = H2
        self.W2 = W2

        if self.train:
            skip = max(int(self.points.shape[0] // 100000), 1)
            self.points = self.points[::skip, :].float()
            self.feats = self.feats[:, ::skip].float()

        print('finished calc points', self.points.shape, self.feats.shape)

    def __getitem__(self, item, only_pose=False, raw=False):
        if item not in self.cache or only_pose:
            self.prepare_getitem(item)

            extrinsics = self.load_extrinsics(item)
            extrinsics = torch.from_numpy(extrinsics)

            if only_pose:
                return extrinsics

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

            if self.transform_depth:
                depth = self.transform_depth(depth)

            if raw:
                return rgb, depth, extrinsics, intrinsics

            # TODO create
            # points2: 3d projection of each pixel, use the unproject function. then also normalize it by biggest coordinate value in xyz
            # feats: vgg features of the color image
            # points: reprojection to another view, sampling map for each pixel in (-1, 1). TODO can just make uniform sample map for test time
            # tgt: color image
            # src: zeros_like(tgt)
            # style: style image

            # raise ValueError(points.shape, emb.shape)  # ((226, 273, 490, 3), (226, 256, 72, 124))

            feats = self.enc_net(rgb.unsqueeze(0)).detach()
            H2, W2 = feats.shape[2:4]
            feats = feats.permute(1, 0, 2, 3)
            feats = feats.reshape(256, -1)

            points2 = unproject(extrinsics.unsqueeze(0), intrinsics.unsqueeze(0), depth.unsqueeze(0)).detach()
            H1, W1 = points2.shape[1:3]
            y = [int(round(y)) for y in np.array(list(range(H2))) / (H2 - 1) * (H1 - 1)]
            x = [int(round(x)) for x in np.array(list(range(W2))) / (W2 - 1) * (W1 - 1)]
            points2 = np.take(points2, y, axis=1)
            points2 = np.take(points2, x, axis=2)

            points2 = points2[:, :, :, :3]  # only need xyz
            points2 = points2.reshape(-1, 3)

            # TODO FIXME points: simple uniform sampling map, if we want to train this and use reprojection, we have to use reprojected sampling map!
            if not self.train or True:
                h, w = rgb.shape[1:]
                w_range = torch.arange(0, w, dtype=torch.float) / (w - 1.0) * 2.0 - 1.0
                h_range = torch.arange(0, h, dtype=torch.float) / (h - 1.0) * 2.0 - 1.0
                v, u = torch.meshgrid(h_range, w_range)
                uv_id = torch.stack([u, v, depth.squeeze()], 2)
                points = uv_id.detach()

                H1, W1 = points.shape[0:2]
                y = [int(round(y)) for y in np.array(list(range(H2))) / (H2 - 1) * (H1 - 1)]
                x = [int(round(x)) for x in np.array(list(range(W2))) / (W2 - 1) * (W1 - 1)]
                points = np.take(points, y, axis=0)
                points = np.take(points, x, axis=1)
                points = points.reshape(-1, 3)

            """
            # how is the sampling map defined: we want to map the color image (tgt) TO ANOTHER VIEW
            # (a) choose any neighboring view of the color image
            # (b) use reproject function to find a map that maps the color image to that view
            # (c) return this is the new points
            neighbor = item + 2 if self.train else item
            if neighbor >= self.__len__():
                neighbor = item - 2
            extr_src = self.__getitem__(neighbor, only_pose=True)
            h, w = rgb.shape[1:]
            points = reproject(extr_src.unsqueeze(0), intrinsics.unsqueeze(0), self.points, h, w)
            """

            style = Image.open(self.style_path)
            style = self.transform_rgb(style)

            # if eval style has another dimension with multiple style images.
            # if train this is not present
            if not self.train:
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
