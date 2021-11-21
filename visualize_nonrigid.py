import argparse
import sys 
import os 

import torch, numpy as np, glob, math, torch.utils.data, scipy.ndimage, multiprocessing as mp
import torch.nn.functional as F
import time
import torch.nn as nn
import pickle 
import datetime
import logging
import open3d as o3d

from tqdm import tqdm 


from models.models import SuperGlueModel
import torchvision

from pathlib import Path

from datasets import load_datasets
import cmd_args 

from models.bcpd import bcpd_refine, bcpd_loop

def visualize_cloud(source, pred_sf, target):
    ''' Displays deformed points in orange, target points are in blue.
    '''
    source_c = o3d.geometry.PointCloud()
    source_c.points = o3d.utility.Vector3dVector(source)
    source_c.paint_uniform_color([0.2, 0.7, 0])

    pred_c = o3d.geometry.PointCloud()
    pred_c.points = o3d.utility.Vector3dVector(source + pred_sf)
    pred_c.paint_uniform_color([0.9, 0.6, 0])

    target_c = o3d.geometry.PointCloud()
    target_c.points = o3d.utility.Vector3dVector(target)
    target_c.paint_uniform_color([0, 0, 1])

    o3d.visualization.draw_geometries([target_c, pred_c])

def visualize_no_match(pred, target, no_match):
    ''' Display deformed points in orange if it has a match,
    or magenta, if does not. Target points are in blue.
    '''
    pred_c = o3d.geometry.PointCloud()
    pred_c.points = o3d.utility.Vector3dVector(pred)
    pred_colors = np.zeros_like(pred)
    pred_colors[:, :2] = 0.6
    pred_colors[:, 0] = 0.9
    no_match_np = no_match[0].cpu().numpy()
    pred_colors[no_match_np, 2] = 0.75
    pred_colors[no_match_np, 1] = 0 
    pred_c.colors = o3d.utility.Vector3dVector(pred_colors)

    target_c = o3d.geometry.PointCloud()
    target_c.points = o3d.utility.Vector3dVector(target)
    target_c.paint_uniform_color([0, 0, 1])

    o3d.visualization.draw_geometries([pred_c, target_c])


def main():
    global args 
    args = cmd_args.parse_args_from_yaml(sys.argv[1])
    args = cmd_args.parse_args_from_cmd(args)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu if args.multi_gpu is None else '0,1,2,3'

    blue = lambda x: '\033[94m' + x + '\033[0m'
    print("Model Name", args.model_name)
    if args.model_name == "SuperGlue":
        model = SuperGlueModel(args)
    else:
        raise NotImplementedError(f'Model: {args.model_name} not implemented')
    
    #load pretrained model
    pretrain = args.ckpt_dir + args.pretrain
    model.load_state_dict(torch.load(pretrain))
    print('load model %s'%pretrain)

    cuda_enabled = True

    if cuda_enabled:
        model.cuda()

    # path to source and target cloud should be passed at the config file.
    if args.pc1[-3:] == 'ply' or args.pc1[-3:] == 'off':
        src_mesh = o3d.io.read_triangle_mesh(args.pc1)
        src = np.asarray(src_mesh.vertices)
        tgt_mesh = o3d.io.read_triangle_mesh(args.pc2)
        tgt = np.asarray(tgt_mesh.vertices)
        minpoints = min(src.shape[0], tgt.shape[0])
        src = src[:minpoints,:]
        tgt = tgt[:minpoints,:]
        print('mesh load', src.shape, tgt.shape)
    else:
        src = np.loadtxt(args.pc1)
        tgt = np.loadtxt(args.pc2)

    transforms = load_datasets.get_transforms(partition='eval',
                                              num_points=args.num_points, 
                                              noise_type=args.DATASET.NOISE_TYPE,
                                              rot_mag = args.DATASET.ROT_MAG,
                                              trans_mag = args.DATASET.TRANS_MAG,
                                              partial_p_keep = args.DATASET.PARTIAL_P_KEEP, 
                                              two_clouds=True, 
                                              nholes=args.DATASET.NHOLES, k=args.DATASET.KHOLES)                       
    
    transforms = torchvision.transforms.Compose(transforms)

    sample = {'points1': src, 'points2': tgt, 'idx': 0}
    sample = transforms(sample)

    src_o3 = o3d.geometry.PointCloud()
    ref_o3 = o3d.geometry.PointCloud()
    src_o3.points = o3d.utility.Vector3dVector(sample['points_src'][:, :3])
    ref_o3.points = o3d.utility.Vector3dVector(sample['points_ref'][:, :3])
    src_o3.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    ref_o3.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    pos1 = torch.Tensor(sample['points_src']).unsqueeze(0)
    pos2 = torch.Tensor(sample['points_ref']).unsqueeze(0)
    norm1 = torch.Tensor(np.asarray(src_o3.normals)).unsqueeze(0)
    norm2 = torch.Tensor(np.asarray(ref_o3.normals)).unsqueeze(0)
    perm_mat = torch.tensor(sample['perm_mat'].astype('float32')).unsqueeze(0)

    if 'points_flow' in sample:
        flow = torch.tensor(sample['points_flow'].astype('float32')).unsqueeze(0)
    else:
        flow = torch.bmm(perm_mat, pos2) - pos1

    if cuda_enabled:
        pos1 = pos1.cuda()
        pos2 = pos2.cuda()
        norm1 = norm1.cuda()
        norm2 = norm2.cuda()
        perm_mat = perm_mat.cuda()
        flow = flow.cuda()
    
    beta = args.BCPD.BETA
    gamma = args.BCPD.GAMMA
    lmbd = args.BCPD.LMBD
    if hasattr(args.BCPD, 'TOLERANCE'):
        tolerance = args.BCPD.TOLERANCE
    else:
        tolerance = 1e-3
    if hasattr(args.BCPD, 'K'):
        K = args.BCPD.K
    else:
        K = 100
            
    model = model.eval()
    with torch.no_grad():

        if args.EVAL.LOOP:
            refined_cloud, s_pred = bcpd_loop(model, pos1, pos2, beta=beta, gamma=gamma, lmbd=lmbd, tolerance=tolerance, K=K)
            pred_flow = refined_cloud - pos1
            Inlier_src_pre = torch.sum(s_pred, dim=-1, keepdim=True)
            Inlier_ref_pre = torch.sum(s_pred, dim=-2)[:, :, None]
            
            no_match = torch.sum(s_pred, dim=-1) < 0.5
        else:
            preds = model(pos1, pos2, norm1, norm2)
            s_pred = preds["P"]

            if hasattr(args, 'EVAL') and hasattr(args.EVAL, 'CYCLE') and args.EVAL.CYCLE:
                preds_inv  = model(pos2, pos1, norm2, norm1)
                s_pred_inv = preds_inv["P"]
                s_pred = s_pred * s_pred_inv.permute(0, 2, 1)

            Inlier_src_pre = preds["SrcInlier"]
            Inlier_ref_pre  = preds["RefInlier"]

            no_match = torch.sum(s_pred, dim=-1) < 0.5
            for batch in range(no_match.shape[0]):
                s_pred[batch, no_match[batch], :] = 1e-9
            refined_cloud = bcpd_refine(s_pred, pos1, pos2, beta=beta, gamma=gamma, lmbd=lmbd, K=K, tolerance=tolerance)
            pred_flow = refined_cloud - pos1

        if args.DATASET.NOISE_TYPE != 'crop':
            pred_flow_back_proj =  torch.bmm(s_pred, pos2) - pos1
            epe3d_back_proj = torch.norm(pred_flow_back_proj - flow, dim = 2).mean()

        matched_points = torch.sum(perm_mat, dim=-1) > 0
        epe3d = torch.norm(pred_flow - flow, dim = 2)[matched_points].mean()        

    pred_sf = pred_flow.cpu().numpy()

    print(f'EPE3D {epe3d:.4f}')
    
    if args.DATASET.NOISE_TYPE != 'crop':
        print(f'EPE3D BP {epe3d_back_proj:.4f}')
        visualize_cloud(sample['points_src'], pred_sf[0], sample['points_ref'])
    else:
        visualize_no_match(refined_cloud[0].cpu().numpy(), sample['points_ref'], no_match)


if __name__ == '__main__':
    main()




