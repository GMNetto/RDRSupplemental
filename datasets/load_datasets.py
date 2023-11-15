#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import glob
import h5py
import re
import torch
import numpy as np
from typing import List
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision

from .build_graphs import build_graphs
from . import data_transform_syndata as Transforms
from . import se3
# import utils.data_transform4 as Transforms

import open3d


# Part of the code is referred from: https://github.com/charlesq34/pointnet

def download():
    DATA_DIR = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www+' --no-check-certificate', zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_data(partition):
    download()
    DATA_DIR = os.path.dirname(os.path.abspath(__file__))
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
        f = h5py.File(h5_name, 'r')
        # data = f['data'][:].astype('float32')
        data = np.concatenate([f['data'][:], f['normal'][:]], axis=-1)
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def get_transforms(partition: str, num_points: int = 1024,
                   noise_type: str = 'clean', rot_mag: float = 45.0,
                   trans_mag: float = 0.5, scale_mag: float = 1.0,
                   partial_p_keep: List = None, two_clouds: bool = False,
                   nholes: int = 10, k: int = 10, ns: int = 50):
    """Get the list of transformation to be used for training or evaluating RegNet

    Args:
        noise_type: Either 'clean', 'jitter', 'crop', 'outlier', 'holes'.
          Depending on the option, some of the subsequent arguments may be ignored.
        rot_mag: Magnitude of rotation perturbation to apply to source, in degrees.
          Default: 45.0 (same as Deep Closest Point)
        trans_mag: Magnitude of translation perturbation to apply to source.
          Default: 0.5 (same as Deep Closest Point)
        num_points: Number of points to uniformly resample to.
          Note that this is with respect to the full point cloud. The number of
          points will be proportionally less if cropped
        partial_p_keep: Proportion to keep during cropping, [src_p, ref_p]
          Default: [0.7, 0.7], i.e. Crop both source and reference to ~70%
        two_clouds: Whether to consider a pair of clouds at first.
          It is assumed a one-to-one correspondence in this pair.
          Default: False
        nholes: Number of seeds whose surrounding will be removed at 'holes'.
          Default: [10]
        k: Number of neighbors of each nholes seeds which will be removed.
          Default: [10]

    Returns:
        train_transforms, test_transforms: Both contain list of transformations to be applied
    """
    partial_p_keep = partial_p_keep if partial_p_keep is not None else [0.7, 0.7]
    print('noise type', noise_type)
    if noise_type == "clean":
        # 1-1 correspondence for each point (resample first before splitting), no noise
        if partition == 'train':
            if two_clouds:
                transforms = [Transforms.Resampler2(num_points),
                            Transforms.SplitSourceRef2(),
                            Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag, scale_mag=scale_mag),
                            Transforms.ShufflePoints()
                            ]
            else:
                transforms = [Transforms.Resampler(num_points),
                            Transforms.SplitSourceRef(),
                            Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag, scale_mag=scale_mag),
                            Transforms.ShufflePoints()]

        else:
            if two_clouds:
                transforms = [Transforms.SetDeterministic(),
                                Transforms.FixedResampler2(num_points),
                                Transforms.SplitSourceRef2(),
                                Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag, scale_mag=scale_mag),
                                Transforms.ShufflePoints()]
            else:
                transforms = [Transforms.SetDeterministic(),
                                Transforms.FixedResampler(num_points),
                                Transforms.SplitSourceRef(),
                                Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag, scale_mag=scale_mag),
                                Transforms.ShufflePoints()]

    elif noise_type == "jitter":
        # Points randomly sampled (might not have perfect correspondence), gaussian noise to position
        if partition == 'train':
            if two_clouds:
                transforms = [Transforms.SetJitterFlag(),
                            Transforms.SplitSourceRef2(),
                            Transforms.Resampler2(num_points),
                            Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag, scale_mag=scale_mag),
                            Transforms.RandomJitter(),
                            Transforms.ShufflePoints()]
            else:
                transforms = [Transforms.SetJitterFlag(),
                            Transforms.SplitSourceRef(),
                            Transforms.Resampler(num_points),
                            Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag, scale_mag=scale_mag),
                            Transforms.RandomJitterOriginal(),
                            Transforms.ShufflePointsOriginal()]
        else:
            if two_clouds:
                transforms = [Transforms.SetJitterFlag(),
                            Transforms.SetDeterministic(),
                            Transforms.SplitSourceRef2(),
                            Transforms.Resampler2(num_points),
                            Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag, scale_mag=scale_mag),
                            Transforms.RandomJitter(),
                            Transforms.ShufflePoints()
                            ]
            else:
                transforms = [Transforms.SetJitterFlag(),
                            Transforms.SetDeterministic(),
                            Transforms.SplitSourceRef(),
                            Transforms.Resampler(num_points),
                            Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag, scale_mag=scale_mag),
                            Transforms.RandomJitterOriginal(),
                            Transforms.ShufflePointsOriginal()]

    elif noise_type == "crop":
        if partition == 'train':
            if two_clouds: # 1-1 correspondence for each point (resample first before splitting), no noise to preserve 1-1.
                transforms = [Transforms.SetCorpFlag(),
                            Transforms.SplitSourceRef2(),
                            Transforms.Resampler2(num_points),
                            Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag, scale_mag=scale_mag),
                            Transforms.RandomCrop(partial_p_keep),
                            Transforms.ShufflePoints()
                            ]
            else: # Both source and reference point clouds cropped, plus same noise in "jitter"
                transforms = [Transforms.SetCorpFlag(),
                            Transforms.SplitSourceRef(),
                            Transforms.Resampler(num_points),
                            Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag, scale_mag=scale_mag),
                            Transforms.RandomJitterOriginal(),
                            Transforms.RandomCropOriginal(partial_p_keep),
                            Transforms.ShufflePointsOriginal()]
        else:
            if two_clouds: # 1-1 correspondence for each point (resample first before splitting), no noise to preserve 1-1.
                transforms = [Transforms.SetCorpFlag(),
                            Transforms.SetDeterministic(),
                            Transforms.SplitSourceRef2(),
                            Transforms.Resampler2(num_points),
                            Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag, scale_mag=scale_mag),
                            Transforms.RandomCrop(partial_p_keep),
                            Transforms.ShufflePoints()]
            else: # Both source and reference point clouds cropped, plus same noise in "jitter"
                transforms = [Transforms.SetCorpFlag(),
                            Transforms.SetDeterministic(),
                            Transforms.SplitSourceRef(),
                            Transforms.Resampler(num_points),
                            Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag, scale_mag=scale_mag),
                            Transforms.RandomCropOriginal(partial_p_keep),
                            Transforms.RandomJitterOriginal(),
                            Transforms.ShufflePointsOriginal()]
    elif noise_type == "outlier":
        if not two_clouds:
            raise NotImplementedError
        # 20% of points on target are uniformly distributed outliers.
        if partition == 'train':
            transforms = [Transforms.SetOutlierFlag(),
                        Transforms.SplitSourceRef2(),
                        Transforms.Resampler2(num_points),
                        Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag, scale_mag=scale_mag),
                        Transforms.UniformOutlier(),
                        Transforms.ShufflePoints()]
        else:
            transforms = [Transforms.SetOutlierFlag(),
                        Transforms.SetDeterministic(),
                        Transforms.SplitSourceRef2(),
                        Transforms.Resampler2(num_points),
                        Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                        Transforms.UniformOutlier(),
                        Transforms.ShufflePoints()]
    elif noise_type == "holes":
        if not two_clouds:
            raise NotImplementedError
        # Neighbors of selected seeds are removed, theses seeds are selected by farthest sampling.
        if partition == 'train':
            transforms = [Transforms.SetOutlierFlag(),
                        Transforms.SplitSourceRef2(),
                        Transforms.Resampler2(num_points),
                        Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag, scale_mag=scale_mag),
                        Transforms.RandomHoles(nholes=nholes, k=k),
                        Transforms.ShufflePoints()]
        else:
            transforms = [Transforms.SetOutlierFlag(),
                        Transforms.SetDeterministic(),
                        Transforms.SplitSourceRef2(),
                        Transforms.Resampler2(num_points),
                        Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                        Transforms.RandomHoles(nholes=nholes, k=k),
                        Transforms.ShufflePoints()]
    elif noise_type == "cropinv":
        if two_clouds:
            raise NotImplementedError
        # Both source and reference point clouds cropped, plus same noise in "jitter"
        if partition == 'train':
            transforms = [Transforms.SetCorpFlag(),
                          Transforms.SplitSourceRef(),
                          Transforms.Resampler(num_points),
                          Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                          Transforms.RandomCropinv(partial_p_keep),
                          Transforms.RandomJitter(),
                          Transforms.ShufflePoints()]
        else:
            transforms = [Transforms.SetCorpFlag(),
                          Transforms.SetDeterministic(),
                          Transforms.SplitSourceRef(),
                          Transforms.Resampler(num_points),
                          Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                          Transforms.RandomCropinv(partial_p_keep),
                          Transforms.RandomJitter(),
                          Transforms.ShufflePoints()]
    else:
        raise NotImplementedError(f'{noise_type} not supported.')

    return transforms

class FingerPrints(Dataset):

    def sorted_nicely(self, l ): 
        """ Sort the given iterable in the way that humans expect.""" 
        convert = lambda text: int(text) if text.isdigit() else text 
        alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
        return sorted(l, key = alphanum_key)

    def __init__(self, partition='train', unseen=False, transform=None, crossval = False, train_part=False, proportion=0.8, fast_test = False):
        # data_shape:[B, N, 3]
        self.gallery = []
        self.gallery_sizes = []
        DATA_DIR = "/Users/I861962/Downloads/dataset/"
        for txt_name in self.sorted_nicely(glob.glob(os.path.join(DATA_DIR, 'gallery_points', '*.txt'))):
            gallery = np.loadtxt(txt_name)
            gallery = np.c_[gallery, np.zeros(gallery.shape[0])]
            self.gallery.append(gallery)
            # self.gallery_sizes.append(gallery.shape[0])

        tmp = self.gallery
        length = len(tmp)
        self.gallery = []
        for i in range(length):
            for _ in range(length):
                self.gallery.append(np.copy(tmp[i]))
                self.gallery_sizes.append(tmp[i].shape[0])

        self.query = []
        self.query_sizes = []
        for txt_name in self.sorted_nicely(glob.glob(os.path.join(DATA_DIR, 'query_points', '*.txt'))):
            query = np.loadtxt(txt_name)
            query = np.c_[query, np.zeros(query.shape[0])]
            self.query.append(query)
            # self.query = np.concatenate(all_query, axis=0)
            self.query_sizes.append(query.shape[0])


        self.query = len(self.query)*self.query
        self.query_sizes = len(self.query_sizes)*self.query_sizes


    def __getitem__(self, item):
        
        print("loading item", item, self.gallery[item].shape, self.query[item].shape)

        # Normals currently not being used
        src_o3 = open3d.geometry.PointCloud()
        ref_o3 = open3d.geometry.PointCloud()
        src_o3.points = open3d.utility.Vector3dVector(self.gallery[item])
        ref_o3.points = open3d.utility.Vector3dVector(self.query[item])
        src_o3.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30))
        ref_o3.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30))


        n1_gt = self.gallery_sizes[item]
        n2_gt = self.query_sizes[item]

        ret_dict = {'Ps': [torch.Tensor(x) for x in [self.query[item], self.gallery[item]]],
                    'Fs': [torch.Tensor(x) for x in [np.asarray(ref_o3.normals), np.asarray(src_o3.normals)]],
                    'ns': [torch.tensor(x) for x in [n2_gt, n1_gt]],
                    }
        return ret_dict
        
    def __len__(self):
        return len(self.gallery)

class ModelNet40(Dataset):
    def __init__(self, partition='train', unseen=False, transform=None, crossval = False, train_part=False, proportion=0.8, fast_test = False):
        # data_shape:[B, N, 3]
        if partition == 'eval':
            self.data, self.label = load_data('test')
        else:    
            self.data, self.label = load_data('train')
        if partition == 'test':
            self.data = self.data[-1000:,...]
            self.label = self.label[-1000:,...]
        elif partition == 'train':
            self.data = self.data[:-1000,...]
            self.label = self.label[:-1000,...]

        self.partition = partition
        self.unseen = unseen
        self.label = self.label.squeeze()
        self.transform = transform
        self.crossval = crossval
        self.train_part = train_part
        self.fast_test = fast_test
        if self.unseen:
            ######## simulate testing on first 20 categories while training on last 20 categories
            if self.partition == 'eval':
                self.data = self.data[self.label>=20]
                self.label = self.label[self.label>=20]
            elif self.partition == 'train':
                self.data = self.data[self.label<20]
                self.label = self.label[self.label<20]
            else:
                self.data = self.data[self.label<20]
                self.label = self.label[self.label<20]
        else:
            if self.crossval:
                if self.train_part:
                    self.data = self.data[0:int(self.label.shape[0]*proportion)]
                    self.label = self.label[0:int(self.label.shape[0]*proportion)]
                else:
                    self.data = self.data[int(self.label.shape[0]*proportion):-1]
                    self.label = self.label[int(self.label.shape[0]*proportion):-1]

    def __getitem__(self, item):
        sample = {'points': self.data[item, :, :3], 'label': self.label[item], 'idx': np.array(item, dtype=np.int32)}

        if self.transform:
            sample = self.transform(sample)

        T_ab = sample['transform_gt']
        T_ba = np.concatenate((T_ab[:,:3].T, np.expand_dims(-(T_ab[:,:3].T).dot(T_ab[:,3]), axis=1)), axis=-1)

        n1_gt, n2_gt = sample['perm_mat'].shape
        A1_gt, e1_gt = build_graphs(sample['points_src'], sample['src_inlier'], n1_gt, stg='fc')
        if 'fc' == 'same':
            A2_gt = A1_gt.transpose().contiguous()
            e2_gt= e1_gt
        else:
            A2_gt, e2_gt = build_graphs(sample['points_ref'], sample['ref_inlier'], n2_gt, stg='fc')

        # Normals currently not being used
        src_o3 = open3d.geometry.PointCloud()
        ref_o3 = open3d.geometry.PointCloud()
        src_o3.points = open3d.utility.Vector3dVector(sample['points_src'][:, :3])
        ref_o3.points = open3d.utility.Vector3dVector(sample['points_ref'][:, :3])
        src_o3.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        ref_o3.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        ret_dict = {'Ps': [torch.Tensor(x) for x in [sample['points_src'], sample['points_ref']]],
                    'Fs': [torch.Tensor(x) for x in [np.asarray(src_o3.normals), np.asarray(ref_o3.normals)]],
                    'ns': [torch.tensor(x) for x in [n1_gt, n2_gt]],
                    'es': [torch.tensor(x) for x in [e1_gt, e2_gt]],
                    'gt_perm_mat': torch.tensor(sample['perm_mat'].astype('float32')),
                    'As': [torch.Tensor(x) for x in [A1_gt, A2_gt]],
                    'Ts': [torch.Tensor(x) for x in [T_ab.astype('float32'), T_ba.astype('float32')]],
                    'Ins': [torch.Tensor(x) for x in [sample['src_inlier'], sample['ref_inlier']]],
                    'label': torch.tensor(sample['label']),
                    'raw': torch.Tensor(sample['points_raw']),
                    }
        if 'points_flow' in sample:
            ret_dict['Ds'] = torch.tensor(sample['points_flow'].astype('float32'))
        return ret_dict
        
    def __len__(self):
        if self.fast_test:
            return min(self.data.shape[0],1000)
        return self.data.shape[0]

class RMANet(Dataset):

    def __init__(self, partition='train', train_part=False, transform=None, normalfeat = False, normalize = False, fast_test = False):

        if 'TRAINING_SET' in os.environ:
            self.train_root = os.getenv('TRAINING_SET')
        else:
            self.train_root = './datasets/'

        if 'EVALUATION_SET' in os.environ:
            self.eval_root = os.getenv('EVALUATION_SET')
        else:
            self.eval_root = './datasets/'

        self.num_points = 2048
        self.train = True
        if partition == 'train':
            self.train_points_pair=np.fromfile(os.path.join(self.train_root, 'train_rma.bin'), dtype = np.float32).reshape(-1,4096,3)
        elif partition == 'eval':
            self.train = False
            self.test_points_pair=np.fromfile(os.path.join(self.eval_root, 'all_test_pairs.bin'), dtype = np.float32).reshape(-1,4096,3)
        else:
            self.train = False
            self.test_points_pair=np.fromfile(os.path.join(self.train_root, 'train_rma.bin'), dtype = np.float32).reshape(-1,4096,3)[-1000:,...]

        self.transform = transform
        self.samples = self.make_dataset(train_part)
        self.fast_test = fast_test

    def __len__(self):
        if self.fast_test:
            return min(len(self.samples),1000)
        else:
            print(len(self.samples))
            return len(self.samples)

    def __getitem__(self, index):
        pair = self.samples[index]
        pair_half = int(pair.shape[0]/2)

        pc1_loaded = pair[:pair_half,:]
        pc2_loaded = pair[pair_half:,:]

        sample = {'points1': pc1_loaded, 'points2': pc2_loaded, 'idx': 0}

        if self.transform:
            sample = self.transform(sample)

        T_ab = sample['transform_gt']
        T_ba = np.concatenate((T_ab[:,:3].T, np.expand_dims(-(T_ab[:,:3].T).dot(T_ab[:,3]), axis=1)), axis=-1)

        n1_gt, n2_gt = sample['perm_mat'].shape
        A1_gt, e1_gt = build_graphs(sample['points_src'], sample['src_inlier'], n1_gt, stg='fc')
        A2_gt, e2_gt = build_graphs(sample['points_ref'], sample['ref_inlier'], n2_gt, stg='fc')

        src_o3 = open3d.geometry.PointCloud()
        ref_o3 = open3d.geometry.PointCloud()
        src_o3.points = open3d.utility.Vector3dVector(sample['points_src'][:, :3])
        ref_o3.points = open3d.utility.Vector3dVector(sample['points_ref'][:, :3])
        src_o3.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        ref_o3.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        
        ret_dict = {'Ps': [torch.Tensor(x) for x in [sample['points_src'], sample['points_ref']]],
                    'Fs': [torch.Tensor(x) for x in [np.asarray(src_o3.normals), np.asarray(ref_o3.normals)]],
                    'ns': [torch.tensor(x) for x in [n1_gt, n2_gt]],
                    'As': [torch.Tensor(x) for x in [A1_gt, A2_gt]],
                    'gt_perm_mat': torch.tensor(sample['perm_mat'].astype('float32')),
                    'Ts': [torch.Tensor(x) for x in [T_ab.astype('float32'), T_ba.astype('float32')]],
                    'Ins': [torch.Tensor(x) for x in [sample['src_inlier'], sample['ref_inlier']]],
                    'label':torch.tensor(1)
                    }
        if 'points_flow' in sample:
            ret_dict['Ds'] = torch.tensor(sample['points_flow'].astype('float32'))
        return ret_dict

    def make_dataset(self, full):
        if self.train:
            if full:
                return self.train_points_pair[:-1000,...]
            return self.train_points_pair[-1000:,...]
        return self.test_points_pair

class RMANetSelf(RMANet):
    ''' We do not deform deform each cloud during runtime because of time constraints.

    A script is available to create such self-learning deformation dataset.
    '''

    def __init__(self, partition='train', train_part=False, transform=None, normalfeat = False, normalize = False, fast_test = False):

        if 'TRAINING_SET' in os.environ:
            self.train_root = os.getenv('TRAINING_SET')
        else:
            self.train_root = './datasets/'

        if 'EVALUATION_SET' in os.environ:
            self.eval_root = os.getenv('EVALUATION_SET')
        else:
            self.eval_root = './datasets/'

        self.num_points = 2048

        self.train = True
        if partition == 'train':
            self.train_points_pair=np.fromfile(os.path.join(self.train_root, 'train_rmanet_def.bin'), dtype = np.float32).reshape(-1,4096,3)
        elif partition == 'eval':
            self.train = False
            self.test_points_pair=np.fromfile(os.path.join(self.eval_root, 'all_test_pairs.bin'), dtype = np.float32).reshape(-1,4096,3)
        else:
            self.train = False
            self.test_points_pair=np.fromfile(os.path.join(self.train_root, 'train_rmanet_def.bin'), dtype = np.float32).reshape(-1,4096,3)[-1000:,...]
        
        self.transform = transform
        self.samples = self.make_dataset(train_part)

        self.fast_test = fast_test


class ModelNet40TOSCAMITPartition(Dataset):

    def __init__(self, partition='train', train_part=False, transform=None, normalfeat = False, normalize = False, fast_test = False):
        if 'TRAINING_SET' in os.environ:
            self.train_root = os.getenv('TRAINING_SET')
        else:
            self.train_root = './datasets/'

        if 'VALIDATION_SET' in os.environ:
            self.test_root = os.getenv('VALIDATION_SET')
        else:
            self.test_root = './datasets/'

        if 'EVALUATION_SET' in os.environ:
            self.eval_root = os.getenv('EVALUATION_SET')
        else:
            self.eval_root = './datasets/'

        self.num_points = 2048

        self.train = False
        if partition == 'train':
            self.train = True
            self.train_points_pair = np.fromfile(os.path.join(self.train_root, 'custom_train.bin'), dtype = np.float32).reshape(-1,4096,3)
        elif partition == 'eval':
            self.test_points_pair = np.fromfile(os.path.join(self.eval_root, 'custom_eval.bin'), dtype = np.float32).reshape(-1,4096,3)
        else:
            self.test_points_pair=np.fromfile(os.path.join(self.test_root, 'custom_test.bin'), dtype = np.float32).reshape(-1,4096,3)

        self.transform = transform
        self.samples = self.make_dataset(train_part)

        self.fast_test = fast_test

    def make_dataset(self, full):
        def shuffle(clouds):
            indices = np.arange(clouds.shape[0])
            np.random.shuffle(indices)
            return clouds[indices, :, :]

        if self.train:
            return shuffle(self.train_points_pair)

        return shuffle(self.test_points_pair)

    def __len__(self):
        if self.fast_test:
            return min(len(self.samples),1000)
        else:
            print(len(self.samples))
            return len(self.samples)

    def generate_transform(self):
        ''' On top of the random rigid transformations used for training, a random rotation is applied.
        The objective is to have the network dealing with any possible orientation of input clouds.
        '''

        rot_mag = 360

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
        t_ab = np.zeros(3)
        rand_SE3 = np.concatenate((R_ab, t_ab[:, None]), axis=1).astype(np.float32)
        return rand_SE3

    def apply_transform(self, p0, transform_mat):
            p1 = se3.transform(transform_mat, p0[:, :3])
            if p0.shape[1] == 6:  # Need to rotate normals also
                n1 = so3.transform(transform_mat[:3, :3], p0[:, 3:6])
                p1 = np.concatenate((p1, n1), axis=-1)

            igt = transform_mat
            gt = se3.inverse(igt)

            return p1, gt, igt

    def __getitem__(self, index):
        pair = self.samples[index]
        pair_half = int(pair.shape[0]/2)

        pc1_loaded = pair[:pair_half,:]
        pc2_loaded = pair[pair_half:,:]

        transform = self.generate_transform()
        pc1_loaded, _, _ = self.apply_transform(pc1_loaded, transform)
        pc2_loaded, _, _ = self.apply_transform(pc2_loaded, transform)
   
        sample = {'points1': pc1_loaded, 'points2': pc2_loaded, 'idx': 0}

        if self.transform:
            sample = self.transform(sample)

        T_ab = sample['transform_gt']
        T_ba = np.concatenate((T_ab[:,:3].T, np.expand_dims(-(T_ab[:,:3].T).dot(T_ab[:,3]), axis=1)), axis=-1)

        n1_gt, n2_gt = sample['perm_mat'].shape
        A1_gt, e1_gt = build_graphs(sample['points_src'], sample['src_inlier'], n1_gt, stg='fc')
    
        A2_gt, e2_gt = build_graphs(sample['points_ref'], sample['ref_inlier'], n2_gt, stg='fc')

        src_o3 = open3d.geometry.PointCloud()
        ref_o3 = open3d.geometry.PointCloud()
        src_o3.points = open3d.utility.Vector3dVector(sample['points_src'][:, :3])
        ref_o3.points = open3d.utility.Vector3dVector(sample['points_ref'][:, :3])
        src_o3.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        ref_o3.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        
        ret_dict = {'Ps': [torch.Tensor(x) for x in [sample['points_src'], sample['points_ref']]],
                    'Fs': [torch.Tensor(x) for x in [np.asarray(src_o3.normals), np.asarray(ref_o3.normals)]],
                    'ns': [torch.tensor(x) for x in [n1_gt, n2_gt]],
                    'As': [torch.Tensor(x) for x in [A1_gt, A2_gt]],
                    'gt_perm_mat': torch.tensor(sample['perm_mat'].astype('float32')),
                    'Ts': [torch.Tensor(x) for x in [T_ab.astype('float32'), T_ba.astype('float32')]],
                    'Ins': [torch.Tensor(x) for x in [sample['src_inlier'], sample['ref_inlier']]],
                    'label':torch.tensor(1)
                    }
        if 'points_flow' in sample:
            ret_dict['Ds'] = torch.tensor(sample['points_flow'].astype('float32'))
        return ret_dict



    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Number of points per point cloud: {}\n'.format(self.num_points)
        fmt_str += '    is training: {}\n'.format(self.train)
        fmt_str += '    Root Locations: {} {} {}\n'.format(self.train_root, self.eval_root, self.test_root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class ModelNet40TOSCAMITPartitionSelf(ModelNet40TOSCAMITPartition):

    def __init__(self, partition='train', train_part=False, transform=None, normalfeat = False, normalize = False, fast_test = False):
        ''' We do not deform deform each cloud during runtime because of time constraints.

        A script is available to create such self-learning deformation dataset.
        '''

        if 'TRAINING_SET' in os.environ:
            self.train_root = os.getenv('TRAINING_SET')
        else:
            self.train_root = './datasets/'

        if 'VALIDATION_SET' in os.environ:
            self.test_root = os.getenv('VALIDATION_SET')
        else:
            self.test_root = './datasets/'

        if 'EVALUATION_SET' in os.environ:
            self.eval_root = os.getenv('EVALUATION_SET')
        else:
            self.eval_root = './datasets/'

        self.num_points = 2048

        self.train = False
        # TODO: adapt the size to the config parameter
        if partition == 'train':
            self.train = True
            self.train_points_pair = np.fromfile(os.path.join(self.train_root, 'custom_self_train.bin'), dtype = np.float32).reshape(-1,4096,3)
        elif partition == 'eval':
            self.test_points_pair = np.fromfile(os.path.join(self.eval_root, 'custom_self_eval.bin'), dtype = np.float32).reshape(-1,4096,3)
        else:
            self.test_points_pair=np.fromfile(os.path.join(self.test_root, 'custom_self_test.bin'), dtype = np.float32).reshape(-1,4096,3)

        self.transform = transform
        self.samples = self.make_dataset(train_part)

        self.fast_test = fast_test


def get_datasets(dataset_name='TOSCAMIT', partition='train', num_points=1024, unseen=False,
                 noise_type="clean" , rot_mag = 45.0, trans_mag = 0.5, scale_mag = 1.0,
                 partial_p_keep = [0.7, 0.7], crossval = False, train_part=False,
                 normalfeat = False, normalize = False, fast_test = False,
                 number_holes = 12, kholes = 20):
    print('dataset name', dataset_name)
    if dataset_name=='ModelNet40':
        transforms = get_transforms(partition=partition, num_points=num_points , noise_type=noise_type,
                                    rot_mag = rot_mag, trans_mag = trans_mag, scale_mag = scale_mag, partial_p_keep = partial_p_keep)
        transforms = torchvision.transforms.Compose(transforms)
        datasets = ModelNet40(partition, unseen, transforms, crossval=False, train_part=train_part, fast_test=fast_test)

    if dataset_name=='Fingerprints':
        transforms = get_transforms(partition=partition, num_points=num_points , noise_type=noise_type,
                                    rot_mag = rot_mag, trans_mag = trans_mag, scale_mag = scale_mag, partial_p_keep = partial_p_keep)
        transforms = torchvision.transforms.Compose(transforms)
        datasets = FingerPrints(partition, unseen, transforms, crossval=False, train_part=train_part, fast_test=fast_test)

    elif dataset_name=='ModelNet40TOSCAMITPartition':
        transforms = get_transforms(partition=partition, num_points=num_points , noise_type=noise_type,
                                    rot_mag = rot_mag, trans_mag = trans_mag, scale_mag = scale_mag, partial_p_keep = partial_p_keep, two_clouds=True,
                                    nholes=number_holes, k=kholes)
        transforms = torchvision.transforms.Compose(transforms)
        datasets = ModelNet40TOSCAMITPartition(partition=partition, train_part=train_part, transform=transforms,
                            normalfeat = normalfeat, normalize = normalize, fast_test = fast_test)
    elif dataset_name=='ModelNet40TOSCAMITPartitionSelf':
        transforms = get_transforms(partition=partition, num_points=num_points , noise_type=noise_type,
                                    rot_mag = rot_mag, trans_mag = trans_mag, scale_mag = scale_mag, partial_p_keep = partial_p_keep, two_clouds=True,
                                    nholes=number_holes, k=kholes)
        transforms = torchvision.transforms.Compose(transforms)
        datasets = ModelNet40TOSCAMITPartitionSelf(partition=partition, train_part=train_part, transform=transforms,
                            normalfeat = normalfeat, normalize = normalize, fast_test = fast_test)
    elif dataset_name=='RMANet':
        transforms = get_transforms(partition=partition, num_points=num_points , noise_type=noise_type,
                                    rot_mag = rot_mag, trans_mag = trans_mag, scale_mag = scale_mag, partial_p_keep = partial_p_keep, two_clouds=True,
                                    nholes=number_holes, k=kholes)
        transforms = torchvision.transforms.Compose(transforms)
        datasets = RMANet(partition=partition, train_part=train_part, transform=transforms,
                            normalfeat = normalfeat, normalize = normalize, fast_test = fast_test)

    elif dataset_name=='RMANetSelf':
        transforms = get_transforms(partition=partition, num_points=num_points , noise_type=noise_type,
                                    rot_mag = rot_mag, trans_mag = trans_mag, scale_mag = scale_mag, partial_p_keep = partial_p_keep, two_clouds=True,
                                    nholes=number_holes, k=kholes)
        transforms = torchvision.transforms.Compose(transforms)
        datasets = RMANetSelf(partition=partition, train_part=train_part, transform=transforms,
                            normalfeat = normalfeat, normalize = normalize, fast_test = fast_test)

    else:
        print('please input RMANet, RMANetSelf, ModelNet40TOSCAMITPartition, ModelNet40TOSCAMITPartitionSelf or ModelNet40')

    return datasets


def collate_fn(data: list):
    """
    Create mini-batch data2d for training.
    :param data: data2d dict
    :return: mini-batch
    """
    def pad_tensor(inp):
        assert type(inp[0]) == torch.Tensor
        it = iter(inp)
        t = next(it)
        max_shape = list(t.shape)
        while True:
            try:
                t = next(it)
                for i in range(len(max_shape)):
                    max_shape[i] = int(max(max_shape[i], t.shape[i]))
            except StopIteration:
                break
        max_shape = np.array(max_shape)

        padded_ts = []
        for t in inp:
            pad_pattern = np.zeros(2 * len(max_shape), dtype=np.int64)
            pad_pattern[::-2] = max_shape - np.array(t.shape)
            # if max_shape.shape[0] > 0 and max_shape[0] == 718:
            #     print(pad_pattern, max_shape, t.shape)

            pad_pattern = tuple(pad_pattern.tolist())
            padded_ts.append(F.pad(t, pad_pattern, 'constant'))

        return padded_ts

    def stack(inp):
        if type(inp[0]) == list:
            ret = []
            for vs in zip(*inp):
                ret.append(stack(vs))
        elif type(inp[0]) == dict:
            ret = {}
            for kvs in zip(*[x.items() for x in inp]):
                ks, vs = zip(*kvs)
                for k in ks:
                    assert k == ks[0], "Key value mismatch."
                ret[k] = stack(vs)
        elif type(inp[0]) == torch.Tensor:
            new_t = pad_tensor(inp)
            ret = torch.stack(new_t, 0)
        elif type(inp[0]) == np.ndarray:
            new_t = pad_tensor([torch.from_numpy(x) for x in inp])
            ret = torch.stack(new_t, 0)
        elif type(inp[0]) == str:
            ret = inp
        else:
            raise ValueError('Cannot handle type {}'.format(type(inp[0])))
        return ret

    ret = stack(data)

    # compute CPU-intensive matrix K1, K2 here to leverage multi-processing nature of dataloader
    # if 'Gs' in ret and 'Hs' in ret and :
    #     try:
    #         G1_gt, G2_gt = ret['Gs']
    #         H1_gt, H2_gt = ret['Hs']
    #         sparse_dtype = np.float32
    #         K1G = [kronecker_sparse(x, y).astype(sparse_dtype) for x, y in zip(G2_gt, G1_gt)]  # 1 as source graph, 2 as target graph
    #         K1H = [kronecker_sparse(x, y).astype(sparse_dtype) for x, y in zip(H2_gt, H1_gt)]
    #         K1G = CSRMatrix3d(K1G)
    #         K1H = CSRMatrix3d(K1H).transpose()
    #
    #         ret['Ks'] = K1G, K1H #, K1G.transpose(keep_type=True), K1H.transpose(keep_type=True)
    #     except ValueError:
    #         pass

    return ret


def get_dataloader(dataset, batch_size = 4, workers = 16, shuffle=False):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                       shuffle=shuffle, num_workers=workers,
                                       collate_fn=collate_fn, pin_memory=False)


if __name__ == '__main__':
    train = ModelNet40(1024)
    test = ModelNet40(1024, 'test')
    for data in train:
        print(len(data))
        break

