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

from tqdm import tqdm 

from models.loss import PermLossOcclusion
from utils.hungarian import hungarian
from models.models import SuperGlueModel
from metrics import stats_meter, matching_accuracy, compute_metrics_nonrigid, compute_rigid_metrics

from models.bcpd import bcpd_refine

from pathlib import Path

from datasets import load_datasets
import cmd_args 

import open3d as o3d

def transform(g: np.ndarray, pts: np.ndarray):
    """ Applies the SE3 transform

    Args:
        g: SE3 transformation matrix of size ([B,] 3/4, 4)
        pts: Points to be transformed ([B,] N, 3)

    Returns:
        transformed points of size (N, 3)
    """
    rot = g[..., :3, :3]  # (3, 3)
    trans = g[..., :3, 3]  # (3)

    transformed = pts[..., :3] @ np.swapaxes(rot, -1, -2) + trans[..., None, :]
    return transformed

def square_distance(src, dst):
    return torch.sum((src[:, :, None, :] - dst[:, None, :, :]) ** 2, dim=-1)

def visualize_cloud(source, pred_sf, target):
    ''' Displays deformed points in orange, target points are in blue.
    '''
    print(source.shape, pred_sf.shape, target.shape)
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

def main():
    global args 
    args = cmd_args.parse_args_from_yaml(sys.argv[1])

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu if args.multi_gpu is None else '0,1,2,3'

    pmodel = args.PMODEL if 'PMODEL' in args else ''

    '''CREATE DIR'''
    experiment_dir = Path('./Evaluate_experiment/')
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(str(experiment_dir) + f'/Rigid_{args.model_name}_{args.dataset}-' + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
    file_dir.mkdir(exist_ok=True)
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = file_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    os.system('cp %s %s' % ('models/models.py', log_dir))
    os.system('cp %s %s' % (__file__, log_dir))
    os.system('cp %s %s' % (sys.argv[1], log_dir))

    '''LOG'''
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + 'eval_%s_deform_serial.txt'%args.model_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('----------------------------------------EVALUATING----------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    blue = lambda x: '\033[94m' + x + '\033[0m'
    print("Model Name", args.model_name)
    if args.model_name == "SuperGlue":
        model = SuperGlueModel(args)
    elif args.model_name == "SuperGlueDownUp":
        model = SuperGlueModelDownUp(args)

    val_dataset = load_datasets.get_datasets(dataset_name = args.dataset, 
                                  partition = 'eval',
                                  num_points = args.num_points,
                                  unseen = args.DATASET.UNSEEN,
                                  noise_type = args.DATASET.NOISE_TYPE,
                                  rot_mag = args.DATASET.ROT_MAG,
                                  trans_mag = args.DATASET.TRANS_MAG,
                                  scale_mag = args.DATASET.SCALE_MAG,
                                  partial_p_keep = args.DATASET.PARTIAL_P_KEEP,
                                  crossval = False,
                                  train_part = False,
                                  fast_test = args.fast_test)

    logger.info('val_dataset: ' + str(val_dataset))

    val_loader = load_datasets.get_dataloader(val_dataset, batch_size=args.batch_size, workers=args.workers, shuffle=False)

    #load pretrained model
    pretrain = args.ckpt_dir + args.pretrain
    model.load_state_dict(torch.load(pretrain, map_location='cpu'))
    print('load model %s'%pretrain)
    logger.info('load model %s'%pretrain)

    # model.cuda()

    print('using Kernel based covariance')
    logger.info('using Kernel based covariance')
    
    perm_loss = PermLossOcclusion()

    metrics = stats_meter()

    beta = torch.ones(1).to(device) * args.BCPD.BETA
    gamma = torch.ones(1).to(device) * args.BCPD.GAMMA
    lmbd = torch.ones(1).to(device) * args.BCPD.LMBD
    if hasattr(args.BCPD, 'TOLERANCE'):
        tolerance = args.BCPD.TOLERANCE
    else:
        tolerance = 1e-3
    if hasattr(args.BCPD, 'K'):
        K = args.BCPD.K
    else:
        K = 100

    avg_chamfer = 0
    dim = int(len(val_loader)**(0.5))
    results_chamfer = np.zeros((dim, dim))
    
    for i, data in tqdm(enumerate(val_loader, 0), total=len(val_loader), smoothing=0.9):
        pos1, pos2 = [_ for _ in data['Ps']]
        norm1, norm2 = [_ for _ in data['Fs']]     #featpoints values
        n1_gt, n2_gt = [_ for _ in data['ns']]     #keypoints number

       
        print("eval model", pos1.shape, pos2.shape)
        model = model.eval()

        if args.EVAL.ITERATION: # Taken from RGM
            P1_gt_copy = pos1.clone()
            P2_gt_copy = pos2.clone()
            P1_gt_copy_inv = pos1.clone()
            P2_gt_copy_inv = pos2.clone()
            with torch.no_grad(): 
                s_perm_mat, s_pred = caliters_perm(model, P1_gt_copy, P2_gt_copy, norm1, norm2, n1_gt, n2_gt, args.EVAL.ITERATION_NUM)
                if args.EVAL.CYCLE:
                    s_perm_mat_inv, _ = caliters_perm(model, P2_gt_copy_inv, P1_gt_copy_inv, norm2, norm1, n2_gt, n1_gt, args.EVAL.ITERATION_NUM)
                    s_perm_mat = s_perm_mat * s_perm_mat_inv.permute(0, 2, 1)

            no_match = torch.sum(s_pred, dim=-1) < 0.5
            for batch in range(no_match.shape[0]):
                s_pred[batch, no_match[batch], :] += 1e-9

        else:
            with torch.no_grad(): 
                preds = model(pos1, pos2, norm1, norm2)

            s_pred = preds["P"]
            Inlier_src_pre = preds["SrcInlier"]
            Inlier_ref_pre  = preds["RefInlier"]
           
            no_match = torch.sum(s_pred, dim=-1) < 0.5
            for batch in range(no_match.shape[0]):
                s_pred[batch, no_match[batch], :] = 1e-9

            deformed_pos1 = bcpd_refine(s_pred, pos1, pos2, beta=beta, gamma=gamma, lmbd=lmbd, K=K, tolerance=tolerance)
            pred_flow = deformed_pos1  - pos1
            permloss = perm_loss(s_pred, perm_mat, n1_gt, n2_gt)

            matched_points = torch.sum(perm_mat, dim=-1) > 0.5
            epe3d = torch.norm(pred_flow - flow, dim = 2)[matched_points].mean()
            s_perm_mat = hungarian(s_pred, n1_gt, n2_gt, Inlier_src_pre, Inlier_ref_pre)
            
        
        # compute r,t
        P1_gt = pos1[:,:,:3].clone()
        P2_gt = pos2[:,:,:3].clone()
        pre_P2_gt, _ = calcorrespondpc(s_perm_mat, pos2[:,:,:3])
        R_pre, T_pre = SVDslover(P1_gt, pre_P2_gt, s_perm_mat)


        # Chamfer distance
        # src_transformed = transform(pred_transforms, points_src)
        P1_transformed = torch.from_numpy(transform(torch.cat((R_pre, T_pre[:, :, None]), dim=2).detach().cpu().numpy(),
                                                    P1_gt.detach().cpu().numpy())).to(P1_gt)

        
        P1_transformed = P1_transformed
        P2_gt = P2_gt

        dist_src = torch.min(square_distance(P1_transformed, P2_gt), dim=-1)[0]
        dist_ref = torch.min(square_distance(P2_gt, P1_transformed), dim=-1)[0]
        chamfer_dist = torch.mean(dist_src, dim=1) + torch.mean(dist_ref, dim=1)
        # print('chamfer distance:', chamfer_dist)
        avg_chamfer += chamfer_dist[0]
        row = i//dim
        col = i%dim
        print(row, col)
        results_chamfer[row, col] = chamfer_dist[0]

    print('avg chamfer', avg_chamfer/len(val_loader))
    np.savetxt("chamfer_distances.csv", results_chamfer, delimiter=",", fmt='%-4d')
        # visualize_cloud(P1_gt.detach().cpu().numpy()[0], P1_transformed.detach().cpu().numpy()[0], pos2[:,:,:3].detach().cpu().numpy()[0])

from metrics import SVDslover, calcorrespondpc

def caliters_perm(model, P1_gt_copy, P2_gt_copy, norm1, norm2, n1_gt, n2_gt, estimate_iters):
    lap_solver1 = hungarian
    # s_perm_indexs = []
    s_pred = None
    for estimate_iter in range(estimate_iters):
        preds = model(P1_gt_copy, P2_gt_copy, norm1, norm2)
        s_prem_i = preds["P"]
        s_pred = s_prem_i
        Inlier_src_pre = preds["SrcInlier"]
        Inlier_ref_pre  = preds["RefInlier"]

        s_perm_i_mat = lap_solver1(s_prem_i, n1_gt, n2_gt, Inlier_src_pre, Inlier_ref_pre)
        P2_gt_copy1, s_perm_i_mat_index = calcorrespondpc(s_perm_i_mat, P2_gt_copy)
        # P2_gt_copy1 = P2_gt_copy.clone()
        # s_perm_indexs.append(s_perm_i_mat_index)
        R_pre, T_pre = SVDslover(P1_gt_copy[:,:,:3], P2_gt_copy1[:,:,:3], s_perm_i_mat)
        P1_gt_copy[:,:,:3] = torch.bmm(P1_gt_copy[:,:,:3], R_pre.transpose(2, 1).contiguous()) + T_pre[:, None, :]
        
        norm1[:,:,:3] = norm1[:,:,:3] @ R_pre.transpose(-1, -2)
    return s_perm_i_mat, s_pred

if __name__ == '__main__':
    main()




