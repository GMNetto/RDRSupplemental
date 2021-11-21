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

    val_loader = load_datasets.get_dataloader(val_dataset, batch_size=args.batch_size, workers=args.workers, shuffle=True)

    #load pretrained model
    pretrain = args.ckpt_dir + args.pretrain
    model.load_state_dict(torch.load(pretrain))
    print('load model %s'%pretrain)
    logger.info('load model %s'%pretrain)

    model.cuda()

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

    for i, data in tqdm(enumerate(val_loader, 0), total=len(val_loader), smoothing=0.9):
        pos1, pos2 = [_.cuda() for _ in data['Ps']]     #keypoints coordinate
        norm1, norm2 = [_.cuda() for _ in data['Fs']]     #featpoints values
        n1_gt, n2_gt = [_.cuda() for _ in data['ns']]     #keypoints number
        A1_gt, A2_gt = [_.cuda() for _ in data['As']]     #edge connect matrix
        perm_mat = data['gt_perm_mat'].cuda()             #permute matrix
        T1_gt, T2_gt = [_.cuda() for _ in data['Ts']]
        Inlier_src_gt, Inlier_ref_gt = [_.cuda() for _ in data['Ins']]

        if args.DATASET.NOISE_TYPE != 'crop':
            # This works because all points on src have a match on tgt, so there 
            # is no risk of a point going to zero.
            flow = torch.bmm(perm_mat, pos2) - pos1
        elif 'Ds' in data: # in case of crop, flow directly from src is used.
            flow = data['Ds'].cuda()

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

            deformed_pos1 = bcpd_refine(s_pred, pos1, pos2, beta=beta, gamma=gamma, lmbd=lmbd, K=K, tolerance=tolerance)
            pred_flow = deformed_pos1  - pos1

            matched_points = torch.sum(perm_mat, dim=-1) > 0.5
            epe3d = torch.norm(pred_flow - flow, dim = 2)[matched_points].mean()
            permloss = perm_loss(s_pred, perm_mat, n1_gt, n2_gt)
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
            
        
        match_metrics = matching_accuracy(s_perm_mat, perm_mat, n1_gt)
        perform_metrics = compute_metrics_nonrigid(pred_flow + pos1, s_pred, pos1, pos2)

        rigid_metrics = compute_rigid_metrics(s_perm_mat, pos1[:,:,:3], pos2[:,:,:3], T1_gt[:,:3,:3], T1_gt[:,:3,3])

        metrics['epe3d'].update(epe3d.cpu().numpy(), n=args.batch_size)
        metrics['loss'].update(permloss.cpu().numpy(), n=args.batch_size)
        metrics['chamfer'].update(perform_metrics['chamfer_dist'], n=args.batch_size)

        for i in range(len(match_metrics['acc_gt'])):
            metrics['match_acc_gt'].update(match_metrics['acc_gt'][i])
            metrics['match_acc_pred'].update(match_metrics['acc_pred'][i])
            metrics['match_acc_gt_no_match'].update(match_metrics['acc_gt_no_match'][i])
            metrics['match_acc_pred_no_match'].update(match_metrics['acc_pred_no_match'][i])
            metrics['match_acc_pred_as_match'].update(match_metrics['acc_pred_as_match'][i])

            metrics['r_mse'].update(rigid_metrics['r_mse'][i])
            metrics['r_mae'].update(rigid_metrics['r_mae'][i])
            metrics['t_mse'].update(rigid_metrics['t_mse'][i])
            metrics['t_mae'].update(rigid_metrics['t_mae'][i])
            metrics['err_r_deg'].update(rigid_metrics['err_r_deg'][i])
            metrics['err_t'].update(rigid_metrics['err_t'][i])

    str_out = '%s mean loss: %f mean epe: %f'%(blue('Evaluate'), metrics['loss'].avg, metrics['epe3d'].avg)
    print(str_out)
    logger.info(str_out)

    print('metric info', metrics['r_mse'].count)

    res_str = (' * EPE3D {epe3d_:.4f}\t'
               'CHAMFER {chamfer_:.4f}\t'
               'MATCHGT {matchgt_:.4f}\t'
               'MATCHPRED {matchpred_:.4f}\t'
               'MATCHGTNOMATCH {matchgtnomatch_:.4f}\t'
               'MATCHPREDNOMATCH {matchprednomatch_:.4f}\t'
               'MATCHPREDASMATCH {matchpredasmatch_:.4f}\t'
               '\n'
               'RMSE {r_mse_:.4f}\t'
               'RMAE(MAE) {r_mae_:.4f}\t'
               'TMSE {t_mse_:.4f}\t'
               'TMAE(MAE) {t_mae_:.4f}\t'
               'ERRROT(MIE) {err_r_deg:.4f}\t'
               'ERRTRAN(MIE) {err_t:.4f}\t'
               .format(
                       epe3d_=metrics['epe3d'].avg,
                       chamfer_=metrics['chamfer'].avg,
                       matchgt_=metrics['match_acc_gt'].avg,
                       matchpred_=metrics['match_acc_pred'].avg,
                       matchgtnomatch_=metrics['match_acc_gt_no_match'].avg,
                       matchprednomatch_=metrics['match_acc_pred_no_match'].avg,
                       matchpredasmatch_=metrics['match_acc_pred_as_match'].avg,
                       r_mse_=np.sqrt(metrics['r_mse'].avg),
                       r_mae_=metrics['r_mae'].avg,
                       t_mse_=np.sqrt(metrics['t_mse'].avg),
                       t_mae_=metrics['t_mae'].avg,
                       err_r_deg=metrics['err_r_deg'].avg,
                       err_t=metrics['err_t'].avg
                    )
            )

    print(res_str)
    logger.info(res_str)


from metrics import SVDslover, calcorrespondpc

def caliters_perm(model, P1_gt_copy, P2_gt_copy, norm1, norm2, n1_gt, n2_gt, estimate_iters):
    lap_solver1 = hungarian
    s_perm_indexs = []
    s_pred = None
    for estimate_iter in range(estimate_iters):
        preds = model(P1_gt_copy, P2_gt_copy, norm1, norm2)
        s_prem_i = preds["P"]
        s_pred = s_prem_i
        Inlier_src_pre = preds["SrcInlier"]
        Inlier_ref_pre  = preds["RefInlier"]

        s_perm_i_mat = lap_solver1(s_prem_i, n1_gt, n2_gt, Inlier_src_pre, Inlier_ref_pre)
        P2_gt_copy1, s_perm_i_mat_index = calcorrespondpc(s_perm_i_mat, P2_gt_copy)
        s_perm_indexs.append(s_perm_i_mat_index)
        R_pre, T_pre = SVDslover(P1_gt_copy[:,:,:3], P2_gt_copy1[:,:,:3], s_perm_i_mat)
        P1_gt_copy[:,:,:3] = torch.bmm(P1_gt_copy[:,:,:3], R_pre.transpose(2, 1).contiguous()) + T_pre[:, None, :]
        
        norm1[:,:,:3] = norm1[:,:,:3] @ R_pre.transpose(-1, -2)
    return s_perm_i_mat, s_pred

if __name__ == '__main__':
    main()




