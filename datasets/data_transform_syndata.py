import math
from typing import Dict, List

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import special_ortho_group
import torch
import torch.utils.data

from . import se3
from . import so3
from .build_graphs import farthest_point_sampling

from sklearn.neighbors import NearestNeighbors
import scipy.optimize as opt
import open3d


def nearest_neighbor(src, dst, k=1):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    # assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()



class SplitSourceRef:
    """Clones the point cloud into separate source and reference point clouds"""
    def __call__(self, sample: Dict):
        sample['points_raw'] = sample.pop('points')
        if isinstance(sample['points_raw'], torch.Tensor):
            sample['points_src'] = sample['points_raw'].detach()
            if 'points_ref' in sample: # keeping custom ref, it exists
                sample['points_ref'] = sample['points_ref'].detach()
            else:        
                sample['points_ref'] = sample['points_raw'].detach()
        else:  # is numpy
            sample['points_src'] = sample['points_raw'].copy()
            sample['points_ref'] = sample['points_raw'].copy()

        sample['points_flow'] = sample['points_ref'] - sample['points_src']
        return sample

class SplitSourceRef2:
    """Clones the point cloud into separate source and reference point clouds.
    
    If the input was a pair, they are kept.
    """
    def __call__(self, sample: Dict):
        sample['points_raw1'] = sample.pop('points1')
        sample['points_raw2'] = sample.pop('points2')
        if isinstance(sample['points_raw1'], torch.Tensor) and isinstance(sample['points_raw2'], torch.Tensor):
            sample['points_src'] = sample['points_raw1'].detach()
            sample['points_ref'] = sample['points_raw2'].detach()
            
        else:  # is numpy
            sample['points_src'] = sample['points_raw1'].copy()
            sample['points_ref'] = sample['points_raw2'].copy()
        return sample


class Resampler:
    def __init__(self, num: int):
        """Resamples a point cloud containing N points to one containing M

        Guaranteed to have no repeated points if M <= N.
        Otherwise, it is guaranteed that all points appear at least once.

        Args:
            num (int): Number of points to resample to, i.e. M

        """
        self.num = num

    def __call__(self, sample):

        if 'deterministic' in sample and sample['deterministic']:
            np.random.seed(sample['idx'])

        if 'points' in sample:
            sample['points'] = self._resample(sample['points'], self.num)
        else:
            if 'crop_proportion' not in sample:
                src_size, ref_size = self.num, self.num
            elif len(sample['crop_proportion']) == 1:
                src_size = math.ceil(sample['crop_proportion'][0] * self.num)
                ref_size = self.num
            elif len(sample['crop_proportion']) == 2:
                src_size = math.ceil(sample['crop_proportion'][0] * self.num)
                ref_size = math.ceil(sample['crop_proportion'][1] * self.num)
            else:
                raise ValueError('Crop proportion must have 1 or 2 elements')

            if 'tranflag' in sample:
                sample['points_src'] = self._resample(sample['points_src'], src_size)
                sample['points_ref'] = self._resample(sample['points_ref'], ref_size)
            else:
                sample['points_src'] = self._resample(sample['points_src'], src_size)
                sample['points_ref'] = sample['points_src']

        return sample

    @staticmethod
    def _resample(points, k):
        """Resamples the points such that there is exactly k points.

        If the input point cloud has <= k points, it is guaranteed the
        resampled point cloud contains every point in the input.
        If the input point cloud has > k points, it is guaranteed the
        resampled point cloud does not contain repeated point.
        """

        if k <= points.shape[0]:
            rand_idxs = np.random.choice(points.shape[0], k, replace=False)
            return points[rand_idxs, :]
        elif points.shape[0] == k:
            return points
        else:
            rand_idxs = np.concatenate([np.random.choice(points.shape[0], points.shape[0], replace=False),
                                        np.random.choice(points.shape[0], k - points.shape[0], replace=True)])
            return points[rand_idxs, :]


class Resampler2(Resampler):
    def __init__(self, num: int):
        """Resamples point clouds containing N points to containing M

        Guaranteed to have no repeated points if M <= N.
        Otherwise, it is guaranteed that all points appear at least once.

        Args:
            num (int): Number of points to resample to, i.e. M

        """
        self.num = num

    def __call__(self, sample):

        if 'deterministic' in sample and sample['deterministic']:
            np.random.seed(sample['idx'])

        if 'points1' in sample and 'points2' in sample:
            sample['points1'], sample['points2'] = self._resample2(sample['points1'], sample['points2'], self.num)
        else:
            if 'crop_proportion' not in sample:
                src_size, ref_size = self.num, self.num
            elif len(sample['crop_proportion']) == 1:
                src_size = math.ceil(sample['crop_proportion'][0] * self.num)
                ref_size = self.num
            elif len(sample['crop_proportion']) == 2:
                src_size = math.ceil(sample['crop_proportion'][0] * self.num)
                ref_size = math.ceil(sample['crop_proportion'][1] * self.num)
            else:
                raise ValueError('Crop proportion must have 1 or 2 elements')

            if 'tranflag' in sample:
                sample['points_src'], sample['points_ref'] = self._resample2(sample['points_src'], sample['points_ref'], src_size)
            else:
                sample['points_src'], sample['points_ref'] = self._resample2(sample['points_src'], sample['points_ref'], src_size)

        return sample

    @staticmethod
    def _resample2(points1, points2, k):
        """Resamples the points such that there is exactly k points.

        If the input point cloud has <= k points, it is guaranteed the
        resampled point cloud contains every point in the input.
        If the input point cloud has > k points, it is guaranteed the
        resampled point cloud does not contain repeated point.
        """

        if k <= points1.shape[0]:
            rand_idxs = np.random.choice(points1.shape[0], k, replace=False)
            return points1[rand_idxs, :], points2[rand_idxs, :]
        elif points1.shape[0] == k:
            return points1, points2
        else:
            rand_idxs = np.concatenate([np.random.choice(points1.shape[0], points1.shape[0], replace=False),
                                        np.random.choice(points1.shape[0], k - points1.shape[0], replace=True)])
            return points1[rand_idxs, :], points2[rand_idxs, :]



class FixedResampler(Resampler):
    """Fixed resampling to always choose the first N points.
    Always deterministic regardless of whether the deterministic flag has been set
    """
    @staticmethod
    def _resample(points, k):
        multiple = k // points.shape[0]
        remainder = k % points.shape[0]

        resampled = np.concatenate((np.tile(points, (multiple, 1)), points[:remainder, :]), axis=0)
        return resampled

class FixedResampler2(Resampler2):
    """Fixed resampling to always choose the first N points.
    Always deterministic regardless of whether the deterministic flag has been set
    """
    @staticmethod
    def _resample(points, k):
        multiple = k // points.shape[0]
        remainder = k % points.shape[0]

        resampled = np.concatenate((np.tile(points, (multiple, 1)), points[:remainder, :]), axis=0)
        return resampled

class UniformOutlier:

    def __init__(self, percentage=0.2):
        self.percentage = percentage

    def uniform_outlier(self, cloud, begin_idx):
        ''' Remove last % of points and include outliers insted
        Points from src are removed as well
        '''
        maxes = np.max(cloud, axis=0)
        mins = np.min(cloud, axis=0)

        return np.random.uniform(low=mins, high=maxes, size=(cloud.shape[0] - begin_idx, cloud.shape[1]))

    def __call__(self, sample):
        if 'points' in sample:
            start_idx = int((1 - self.percentage) * sample['points'].shape[0])
            outliers = self.uniform_outlier(sample['points'], start_idx)
            sample['points'][start_idx:, :] = outliers
        else:
            start_idx = int((1 - self.percentage) * sample['points_src'].shape[0])
            outliers = self.uniform_outlier(sample['points_ref'], start_idx)

            sample['points_src'] = sample['points_src'][:start_idx, :]
            sample['points_flow'] = sample['points_ref'][:start_idx, :] - sample['points_src']
            sample['points_ref'][start_idx:, :] = outliers

        return sample


class RandomHoles:

    def __init__(self, nholes=10, k=10):
        self.nholes = nholes
        self.k = k
        self.rng = Generator(PCG64(seed=0))

    def hole_mask(self, points):
        mask = np.ones(points.shape[0])

        initial = self.rng.integers(low=0, high=points.shape[0])
        seed_idxs, _ = farthest_point_sampling(points, self.nholes+1, skip_initial=True, initial_idx=initial)

        _, knn = nearest_neighbor(points[seed_idxs[0]], points, self.k)

        for i in range(knn.shape[0]):
            mask[knn[i]] = 0
        return mask.astype(np.bool)

    def __call__(self, sample):

        if 'deterministic' in sample and sample['deterministic']:
            np.random.seed(sample['idx'])

        flow = sample['points_ref'] - sample['points_src']
        sample['points_flow'] = flow

        mask_ref = self.hole_mask(sample['points_ref'])
        
        sample['perm_mat'] = np.identity(sample['points_src'].shape[0])
        
        sample['mask_src'] = np.ones(sample['points_src'].shape[0]).astype(np.bool) # mask_src
        sample['mask_ref'] = mask_ref

        return sample

class RandomJitter:
    """ generate perturbations """
    def __init__(self, scale=0.02, clip=0.1):
        self.scale = scale
        self.clip = clip

    def jitter(self, pts):

        noise = np.clip(np.random.normal(0.0, scale=self.scale, size=(pts.shape[0], 3)),
                        a_min=-self.clip, a_max=self.clip)
        pts[:, :3] += noise  # Add noise to xyz

        return pts

    def __call__(self, sample):

        if 'points' in sample:
            sample['points'] = self.jitter(sample['points'])
        else:
            #sample['points_src'] = self.jitter(sample['points_src'])
            sample['points_ref'] = self.jitter(sample['points_ref'])

        return sample


class RandomJitterOriginal:
    """ generate perturbations """
    def __init__(self, scale=0.01, clip=0.05):
        self.scale = scale
        self.clip = clip

    def jitter(self, pts):

        noise = np.clip(np.random.normal(0.0, scale=self.scale, size=(pts.shape[0], 3)),
                        a_min=-self.clip, a_max=self.clip)
        pts[:, :3] += noise  # Add noise to xyz

        return pts

    def __call__(self, sample):

        if 'points' in sample:
            sample['points'] = self.jitter(sample['points'])
        else:
            sample['points_src'] = self.jitter(sample['points_src'])
            sample['points_ref'] = self.jitter(sample['points_ref'])

        return sample

from numpy.random import Generator, PCG64


def uniform_2_sphere(num: int = None, rng = None):
    """Uniform sampling on a 2-sphere

    Source: https://gist.github.com/andrewbolster/10274979

    Args:
        num: Number of vectors to sample (or None if single)
        rng: Random Number Generator, this proved to be more
        stable during evaluation than the existing np.random.uniform.

    Returns:
        Random Vector (np.ndarray) of size (num, 3) with norm 1.
        If num is None returned value will have size (3,)

    """
    if num is not None:
        phi = np.random.uniform(0.0, 2 * np.pi, num)
        cos_theta = np.random.uniform(-1.0, 1.0, num)
    else:
        if rng:
            phi = rng.random() * 2 * np.pi
            cos_theta = 2 * rng.random() - 1
        else:
            phi = np.random.uniform(0.0, 2 * np.pi)
            cos_theta = np.random.uniform(-1.0, 1.0)

    theta = np.arccos(cos_theta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return np.stack((x, y, z), axis=-1)

class RandomCrop:
    """Randomly crops the *source* and *reference* point clouds, approximately retaining half the points

    A direction is randomly sampled from S2, and we retain points which lie within the
    half-space oriented in this direction.
    If p_keep != 0.5, we shift the plane until approximately p_keep points are retained
    This class does not crop directly the cloud, instead it creates a occlusion map, this is necessary 
    to guide the posterior shuffling.
    """
    def __init__(self, p_keep: List = None):
        if p_keep is None:
            p_keep = [0.7, 0.7]  # Crop both clouds to 70%
        self.p_keep = np.array(p_keep, dtype=np.float32)
        self.target_size = 0
        self.rng = Generator(PCG64(seed=0)) # 4 horse # 6 armadillo1 # 2 armadillo2 # 4 armadillo4 # 0 others 

    # Since I do not normalize the points, they will be normalized to create the mask
    @staticmethod
    def normalize_cloud(cloud):
        maxes = np.abs(np.max(cloud))
        mins = np.abs(np.min(cloud))
        maxes = max(maxes, mins)

        cloud = 0.5 * cloud / maxes
        return cloud

    # @staticmethod
    def crop(self, original_points, p_keep):
        rand_xyz = uniform_2_sphere(rng=self.rng)

        points = RandomCrop.normalize_cloud(original_points)
        centroid = np.mean(points[:, :3], axis=0)
        points_centered = points[:, :3] - centroid

        dist_from_plane = np.dot(points_centered, rand_xyz)
        if p_keep == 0.5:
            mask = dist_from_plane > 0
        else:
            mask = dist_from_plane > np.percentile(dist_from_plane, (1.0 - p_keep) * 100)
        return mask#original_points[mask, :]

    def normalize_size(self, mask):
        diff = np.sum(mask) - self.target_size
        if diff < 0:
            for i in range(mask.shape[0]):
                if not mask[i]:
                    mask[i] = 1
                    diff += 1
                    if diff == 0:
                        return
        elif diff > 0:
            for i in range(mask.shape[0]):
                if mask[i]:
                    mask[i] = 0
                    diff -= 1
                    if diff == 0:
                        return

    def __call__(self, sample):

        sample['crop_proportion'] = self.p_keep
        if np.all(self.p_keep == 1.0):
            return sample  # No need crop

        if 'deterministic' in sample and sample['deterministic']:
            np.random.seed(sample['idx'])

        if len(self.p_keep) == 1:
            sample['points_src'] = self.crop(sample['points_src'], self.p_keep[0])
        else:
            flow = sample['points_ref'] - sample['points_src']
            sample['points_flow'] = flow
            mask_src = self.crop(sample['points_src'], self.p_keep[0])
            mask_ref = self.crop(sample['points_ref'], self.p_keep[1])
            if self.target_size == 0:
               self.target_size = max(np.sum(mask_ref), np.sum(mask_src)) 
            
            sample['perm_mat'] = np.identity(sample['points_src'].shape[0])
            
            # Include only mask, this helps on shuffle, since we can shuffle the mask as well and apply over shuffled permat
            sample['mask_src'] = mask_src
            sample['mask_ref'] = mask_ref

        return sample

class RandomCropinv:
    """Randomly crops the *source* point cloud, approximately retaining half the points

    A direction is randomly sampled from S2, and we retain points which lie within the
    half-space oriented in this direction.
    If p_keep != 0.5, we shift the plane until approximately p_keep points are retained
    """
    def __init__(self, p_keep: List = None):
        if p_keep is None:
            p_keep = [0.7, 0.7]  # Crop both clouds to 70%
        self.p_keep = np.array(p_keep, dtype=np.float32)

    @staticmethod
    def crop(points, p_keep):
        rand_xyz = uniform_2_sphere()
        centroid = np.mean(points[:, :3], axis=0)
        points_centered = points[:, :3] - centroid

        dist_from_plane = np.dot(points_centered, rand_xyz)
        if p_keep == 0.5:
            mask = dist_from_plane < 0
        else:
            mask = dist_from_plane < np.percentile(dist_from_plane, (1.0 - p_keep) * 100)

        return points[mask, :]

    def __call__(self, sample):

        sample['crop_proportion'] = self.p_keep
        if np.all(self.p_keep == 1.0):
            return sample  # No need crop

        if 'deterministic' in sample and sample['deterministic']:
            np.random.seed(sample['idx'])

        if len(self.p_keep) == 1:
            sample['points_src'] = self.crop(sample['points_src'], self.p_keep[0])
        else:
            sample['points_src'] = self.crop(sample['points_src'], self.p_keep[0])
            sample['points_ref'] = self.crop(sample['points_ref'], self.p_keep[1])
        return sample


class RandomTransformSE3:
    def __init__(self, rot_mag: float = 180.0, trans_mag: float = 1.0, random_mag: bool = False, scale_mag: float = 1.0):
        """Applies a random rigid transformation to the source point cloud

        Args:
            rot_mag (float): Maximum rotation in degrees
            trans_mag (float): Maximum translation T. Random translation will
              be in the range [-X,X] in each axis
            random_mag (bool): If true, will randomize the maximum rotation, i.e. will bias towards small
                               perturbations
        """
        self._rot_mag = rot_mag
        self._trans_mag = trans_mag
        self._random_mag = random_mag
        self._scale_mag = scale_mag

    def generate_transform(self):
        """Generate a random SE3 transformation (3, 4) """

        if self._random_mag:
            attentuation = np.random.random()
            rot_mag, trans_mag = attentuation * self._rot_mag, attentuation * self._trans_mag
        else:
            rot_mag, trans_mag = self._rot_mag, self._trans_mag
        # Generate rotation
        rand_rot = special_ortho_group.rvs(3)
        axis_angle = Rotation.as_rotvec(Rotation.from_dcm(rand_rot))
        axis_angle *= rot_mag / 180.0
        rand_rot = Rotation.from_rotvec(axis_angle).as_dcm()

        # Generate scale
        side_one = self._scale_mag
        side_two = 1 / side_one
        rand_scale = np.random.uniform(side_one, side_two)
        scale_matrix = np.eye(3) * rand_scale

        # Generate translation
        #rand_rot = scale_matrix @ rand_rot

        rand_trans = np.random.uniform(-trans_mag, trans_mag, 3)
        rand_SE3 = np.concatenate((rand_rot, rand_trans[:, None]), axis=1).astype(np.float32)

        return rand_SE3

    def apply_transform(self, p0, transform_mat):
        p1 = se3.transform(transform_mat, p0[:, :3])
        if p0.shape[1] == 6:  # Need to rotate normals also
            n1 = so3.transform(transform_mat[:3, :3], p0[:, 3:6])
            p1 = np.concatenate((p1, n1), axis=-1)

        igt = transform_mat
        gt = se3.inverse(igt)

        return p1, gt, igt

    def transform(self, tensor):
        transform_mat = self.generate_transform()
        return self.apply_transform(tensor, transform_mat)

    def __call__(self, sample):
        sample['tranflag'] = True

        if 'deterministic' in sample and sample['deterministic']:
            np.random.seed(sample['idx'])

        if 'points' in sample:
            sample['points'], _, _ = self.transform(sample['points'])
        else:
            src_transformed, transform_r_s, transform_s_r = self.transform(sample['points_src'])
            sample['transform_gt'] = transform_r_s  # Apply to source to get reference
            sample['points_src'] = src_transformed

        return sample


class RandomTransformSE3Original:
    def __init__(self, rot_mag: float = 180.0, trans_mag: float = 1.0, random_mag: bool = False):
        """Applies a random rigid transformation to the source point cloud

        Args:
            rot_mag (float): Maximum rotation in degrees
            trans_mag (float): Maximum translation T. Random translation will
              be in the range [-X,X] in each axis
            random_mag (bool): If true, will randomize the maximum rotation, i.e. will bias towards small
                               perturbations
        """
        self._rot_mag = rot_mag
        self._trans_mag = trans_mag
        self._random_mag = random_mag

    def generate_transform(self):
        """Generate a random SE3 transformation (3, 4) """

        if self._random_mag:
            attentuation = np.random.random()
            rot_mag, trans_mag = attentuation * self._rot_mag, attentuation * self._trans_mag
        else:
            rot_mag, trans_mag = self._rot_mag, self._trans_mag

        # Generate rotation
        rand_rot = special_ortho_group.rvs(3)
        axis_angle = Rotation.as_rotvec(Rotation.from_dcm(rand_rot))
        axis_angle *= rot_mag / 180.0
        rand_rot = Rotation.from_rotvec(axis_angle).as_dcm()

        # Generate translation
        rand_trans = np.random.uniform(-trans_mag, trans_mag, 3)
        rand_SE3 = np.concatenate((rand_rot, rand_trans[:, None]), axis=1).astype(np.float32)

        return rand_SE3

    def apply_transform(self, p0, transform_mat):
        p1 = se3.transform(transform_mat, p0[:, :3])
        if p0.shape[1] == 6:  # Need to rotate normals also
            n1 = so3.transform(transform_mat[:3, :3], p0[:, 3:6])
            p1 = np.concatenate((p1, n1), axis=-1)

        igt = transform_mat
        gt = se3.inverse(igt)

        return p1, gt, igt

    def transform(self, tensor):
        transform_mat = self.generate_transform()
        return self.apply_transform(tensor, transform_mat)

    def __call__(self, sample):
        sample['tranflag'] = True

        if 'deterministic' in sample and sample['deterministic']:
            np.random.seed(sample['idx'])

        if 'points' in sample:
            sample['points'], _, _ = self.transform(sample['points'])
        else:
            src_transformed, transform_r_s, transform_s_r = self.transform(sample['points_src'])
            sample['transform_gt'] = transform_r_s  # Apply to source to get reference
            sample['points_src'] = src_transformed

        return sample

# noinspection PyPep8Naming
class RandomTransformSE3_euler(RandomTransformSE3):
    """Same as RandomTransformSE3, but rotates using euler angle rotations

    This transformation is consistent to Deep Closest Point but does not
    generate uniform rotations

    """
    def generate_transform(self):

        if self._random_mag:
            attentuation = np.random.random()
            rot_mag, trans_mag = attentuation * self._rot_mag, attentuation * self._trans_mag
        else:
            rot_mag, trans_mag = self._rot_mag, self._trans_mag

        # Generate rotation
        anglex = np.random.uniform() * np.pi * rot_mag / 180.0
        angley = np.random.uniform() * np.pi * rot_mag / 180.0
        anglez = np.random.uniform() * np.pi * rot_mag / 180.0

        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0],
                       [0, cosx, -sinx],
                       [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny],
                       [0, 1, 0],
                       [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0],
                       [sinz, cosz, 0],
                       [0, 0, 1]])

        R_ab = Rx @ Ry @ Rz

        # Generate scale
        side_one = self._scale_mag
        side_two = 1 / side_one
        rand_scale = 1.0 # np.random.uniform(side_one, side_two)
        scale_matrix = np.eye(3) * rand_scale
        # print('scale', rand_scale)

        # Generate translation
        # R_ab = scale_matrix @ R_ab

        t_ab = np.random.uniform(-trans_mag, trans_mag, 3)
        rand_SE3 = np.concatenate((R_ab, t_ab[:, None]), axis=1).astype(np.float32)
        return rand_SE3


class ShufflePoints:
    """Shuffles the order of the points for the 'two_clouds' case
    
    This makes sure there is a GT permutation matrix and flow estimation.
    """

    def __call__(self, sample):
        if 'points' in sample:
            sample['points'] = np.random.permutation(sample['points'])
        else:
            # sample['points_ref'] = np.random.permutation(sample['points_ref'])
            # sample['points_src'] = np.random.permutation(sample['points_src'])
            refperm = np.random.permutation(sample['points_ref'].shape[0])
            srcperm = np.random.permutation(sample['points_src'].shape[0])
            
            if 'points_flow' in sample:
                sample['points_flow'] = sample['points_flow'][srcperm, :]
            else:
                flow = sample['points_ref'] - sample['points_src']
                sample['points_flow'] = flow[srcperm, :]

            sample['points_ref'] = sample['points_ref'][refperm, :]
            sample['points_src'] = sample['points_src'][srcperm, :]
            
            if 'mask_src' in sample:
                sample['mask_src'] = sample['mask_src'][srcperm]
            if 'mask_ref' in sample:
                sample['mask_ref'] = sample['mask_ref'][refperm]
            
            if 'perm_mat' in sample:
                original_perm_mat = sample['perm_mat']
            else:
                min_dim = min(sample['points_src'].shape[0], sample['points_ref'].shape[0])
                original_perm_mat = np.zeros((sample['points_src'].shape[0], sample['points_ref'].shape[0]))
                original_perm_mat[:min_dim, :min_dim] = np.identity(min_dim) 

            perm_mat = np.zeros((sample['points_src'].shape[0], sample['points_ref'].shape[0]))
            srcpermsort = np.argsort(srcperm)
            refpermsort = np.argsort(refperm)
            
            original_index = 0
            for i,j in zip(srcpermsort,refpermsort):
                # In crop this is already wrong because there is no guarantee the match exists
                # it actually does not.
                perm_mat[i, j] = original_perm_mat[original_index, original_index]
                original_index += 1
            
            # Applying mask if they exist
            if 'mask_src' in sample:
                mask_src = sample['mask_src']
                sample['points_src'] = sample['points_src'][mask_src, :]
                sample['points_flow'] = sample['points_flow'][mask_src,:]
                perm_mat = perm_mat[mask_src,:]
            if 'mask_ref' in sample:
                mask_ref = sample['mask_ref']
                sample['points_ref'] = sample['points_ref'][mask_ref, :]
                perm_mat = perm_mat[:, mask_ref]
        
            sample['perm_mat'] = perm_mat
            sample['src_inlier'] = np.ones((sample['points_src'].shape[0], 1))
            sample['ref_inlier'] = np.ones((sample['points_ref'].shape[0], 1))

        return sample



class RandomCropOriginal:
    """Randomly crops the *source* point cloud, approximately retaining half the points

    A direction is randomly sampled from S2, and we retain points which lie within the
    half-space oriented in this direction.
    If p_keep != 0.5, we shift the plane until approximately p_keep points are retained
    """
    def __init__(self, p_keep: List = None):
        if p_keep is None:
            p_keep = [0.7, 0.7]  # Crop both clouds to 70%
        self.p_keep = np.array(p_keep, dtype=np.float32)

    @staticmethod
    def crop(points, p_keep):
        rand_xyz = uniform_2_sphere()
        centroid = np.mean(points[:, :3], axis=0)
        points_centered = points[:, :3] - centroid

        dist_from_plane = np.dot(points_centered, rand_xyz)
        if p_keep == 0.5:
            mask = dist_from_plane > 0
        else:
            mask = dist_from_plane > np.percentile(dist_from_plane, (1.0 - p_keep) * 100)

        return mask

    def __call__(self, sample):

        sample['crop_proportion'] = self.p_keep
        if np.all(self.p_keep == 1.0):
            return sample  # No need crop

        if 'deterministic' in sample and sample['deterministic']:
            np.random.seed(sample['idx'])

        if len(self.p_keep) == 1:
            sample['points_src'] = self.crop(sample['points_src'], self.p_keep[0])
        else:
            mask_src = self.crop(sample['points_src'], self.p_keep[0])
            mask_ref = self.crop(sample['points_ref'], self.p_keep[1])

            flow = sample['points_ref'] - sample['points_src']
            
            sample['points_flow'] = flow[mask_src, :]
            sample['points_src'] = sample['points_src'][mask_src, :]
            sample['points_ref'] = sample['points_ref'][mask_ref, :]
        return sample


class ShufflePointsOriginal:
    """Shuffles the order of the points"""
    def __call__(self, sample):
        if 'points' in sample:
            sample['points'] = np.random.permutation(sample['points'])
        else:
            # sample['points_ref'] = np.random.permutation(sample['points_ref'])
            # sample['points_src'] = np.random.permutation(sample['points_src'])
            refperm = np.random.permutation(sample['points_ref'].shape[0])
            srcperm = np.random.permutation(sample['points_src'].shape[0])
            sample['points_ref'] = sample['points_ref'][refperm, :]
            sample['points_src'] = sample['points_src'][srcperm, :]
            if 'points_flow' in sample:
                sample['points_flow'] = sample['points_flow'][srcperm, :]
            if 'jitterflag' in sample or 'corpflag' in sample:
                perm_mat = np.zeros((sample['points_src'].shape[0], sample['points_ref'].shape[0]))
                inlier_src = np.zeros((sample['points_src'].shape[0], 1))
                inlier_ref = np.zeros((sample['points_ref'].shape[0], 1))
                points_src_transform = se3.transform(sample['transform_gt'], sample['points_src'][:, :3])
                points_ref = sample['points_ref'][:, :3]
                dist_s2et, indx_s2et = nearest_neighbor(points_src_transform, points_ref)
                dist_t2es, indx_t2es = nearest_neighbor(points_ref, points_src_transform)
                padtype = 3 #双边对应填充， 完全填充，双边对应填充+部分对应填充，
                padth = 0.05
                if padtype==1:
                    for row_i in range(sample['points_src'].shape[0]):
                        if indx_t2es[indx_s2et[row_i]]==row_i and dist_s2et[row_i]<padth:
                            perm_mat[row_i, indx_s2et[row_i]] = 1
                elif padtype==2:
                    for row_i in range(sample['points_src'].shape[0]):
                        if dist_s2et[row_i]<padth:
                            perm_mat[row_i, indx_s2et[row_i]] = 1
                    for col_i in range(sample['points_ref'].shape[0]):
                        if dist_t2es[col_i]<padth:
                            perm_mat[indx_t2es[col_i], col_i] = 1
                elif padtype==3:
                    for row_i in range(sample['points_src'].shape[0]):
                        if indx_t2es[indx_s2et[row_i]]==row_i and dist_s2et[row_i]<padth:
                            perm_mat[row_i, indx_s2et[row_i]] = 1
                    for row_i in range(sample['points_src'].shape[0]):
                        if np.sum(perm_mat[row_i, :])==0 \
                                and np.sum(perm_mat[:, indx_s2et[row_i]])==0 \
                                and dist_s2et[row_i]<padth:
                            perm_mat[row_i, indx_s2et[row_i]] = 1
                    for col_i in range(sample['points_ref'].shape[0]):
                        if np.sum(perm_mat[:, col_i])==0 \
                                and np.sum(perm_mat[indx_t2es[col_i], :])==0 \
                                and dist_t2es[col_i]<padth:
                            perm_mat[indx_t2es[col_i], col_i] = 1
                    outlier_src_ind = np.where(np.sum(perm_mat, axis=1)==0)[0]
                    outlier_ref_ind = np.where(np.sum(perm_mat, axis=0)==0)[0]
                    points_src_transform_rest = points_src_transform[outlier_src_ind]
                    points_ref_rest = points_ref[outlier_ref_ind]
                    if points_src_transform_rest.shape[0]>0 and points_ref_rest.shape[0]>0:
                        dist_s2et, indx_s2et = nearest_neighbor(points_src_transform_rest, points_ref_rest)
                        dist_t2es, indx_t2es = nearest_neighbor(points_ref_rest, points_src_transform_rest)
                        for row_i in range(points_src_transform_rest.shape[0]):
                            if indx_t2es[indx_s2et[row_i]]==row_i and dist_s2et[row_i]<padth*2:
                                perm_mat[outlier_src_ind[row_i], outlier_ref_ind[indx_s2et[row_i]]] = 1
                inlier_src_ind = np.where(np.sum(perm_mat, axis=1))[0]
                inlier_ref_ind = np.where(np.sum(perm_mat, axis=0))[0]
                inlier_src[inlier_src_ind] = 1
                inlier_ref[inlier_ref_ind] = 1
                sample['perm_mat'] = perm_mat
                sample['src_inlier'] = inlier_src
                sample['ref_inlier'] = inlier_ref
            else:
                perm_mat = np.zeros((sample['points_src'].shape[0], sample['points_ref'].shape[0]))
                srcpermsort = np.argsort(srcperm)
                refpermsort = np.argsort(refperm)
                for i,j in zip(srcpermsort,refpermsort):
                    perm_mat[i, j] = 1
                # for i, src_i in enumerate(srcperm):
                #     for j, ref_i in enumerate(refperm):
                #         if src_i == ref_i:
                #             perm_mat1[i, j] = 1
                sample['perm_mat'] = perm_mat
                sample['src_inlier'] = np.ones((sample['points_src'].shape[0], 1))
                sample['ref_inlier'] = np.ones((sample['points_ref'].shape[0], 1))

        return sample


class SetDeterministic:
    """Adds a deterministic flag to the sample such that subsequent transforms
    use a fixed random seed where applicable. Used for test"""
    def __call__(self, sample):
        sample['deterministic'] = True
        return sample


class SetJitterFlag:
    """Adds a deterministic flag to the sample such that subsequent transforms
    use a fixed random seed where applicable. Used for test"""
    def __call__(self, sample):
        sample['jitterflag'] = True
        return sample

class SetOutlierFlag:
    """Adds a deterministic flag to the sample such that subsequent transforms
    use a fixed random seed where applicable. Used for test"""
    def __call__(self, sample):
        sample['outlierflag'] = True
        return sample

class SetCorpFlag:
    """Adds a deterministic flag to the sample such that subsequent transforms
    use a fixed random seed where applicable. Used for test"""
    def __call__(self, sample):
        sample['corpflag'] = True
        return sample

