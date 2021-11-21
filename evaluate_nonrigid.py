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
from pathlib import Path

import cmd_args 

from models.loss import PermLossOcclusion
from utils.hungarian import hungarian
from models.models import SuperGlueModel
# SuperGlueModelDownUp, SuperGlueModelOT, SuperGlueModel0Layers, SuperGlueModel6Layers 
from metrics import stats_meter, matching_accuracy, compute_metrics_nonrigid, AverageMeter

from models.bcpd import bcpd_refine, bcpd_loop
from datasets import load_datasets

def main():

    global args 
    args = cmd_args.parse_args_from_yaml(sys.argv[1])

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu if args.multi_gpu is None else '0,1,2,3'

    '''CREATE DIR'''
    experiment_dir = Path('./Evaluate_experiment/')
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(str(experiment_dir) + f'/NonRigid_{args.model_name}_{args.dataset}-' + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
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
    file_handler = logging.FileHandler(str(log_dir) + 'eval_%s.txt'%args.model_name)
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
    # elif args.model_name == "SuperGlue2Layers":
    #     model = SuperGlueModel2Layers(args)
    # elif args.model_name == "SuperGlueOT":
    #     model = SuperGlueModelOT(args)
    # elif args.model_name == "SuperGlue0Layers":
    #     model = SuperGlueModel0Layers(args)
    # elif args.model_name == "SuperGlue6Layers":
    #     model = SuperGlueModel6Layers(args)
    # elif args.model_name == "SuperGlueDownUp":
    #     model = SuperGlueModelDownUp(args)

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
                                  fast_test = args.fast_test,
                                  number_holes=args.DATASET.NHOLES, kholes=args.DATASET.KHOLES)

    logger.info('val_dataset: ' + str(val_dataset))

    val_loader = load_datasets.get_dataloader(val_dataset, batch_size=args.batch_size, workers=args.workers, shuffle=False)

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
    time_meter = AverageMeter()

    # Moving this part later
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
    
    for i, data in tqdm(enumerate(val_loader, 0), total=len(val_loader), smoothing=0.9):
        pos1, pos2 = [_.cuda() for _ in data['Ps']]     #keypoints coordinate
        norm1, norm2 = [_.cuda() for _ in data['Fs']]     #featpoints values
        n1_gt, n2_gt = [_.cuda() for _ in data['ns']]     #keypoints number
        A1_gt, A2_gt = [_.cuda() for _ in data['As']]     #edge connect matrix
        perm_mat = data['gt_perm_mat'].cuda()             #permute matrix
        T1_gt, T2_gt = [_.cuda() for _ in data['Ts']]
        Inlier_src_gt, Inlier_ref_gt = [_.cuda() for _ in data['Ins']]
       
        if args.DATASET.NOISE_TYPE != 'crop' and args.DATASET.NOISE_TYPE != 'holes' and args.DATASET.NOISE_TYPE != 'sample':
            # This works because all points on src have a match on tgt, so there 
            # is no risk of a point going to zero.
            flow = torch.bmm(perm_mat, pos2) - pos1
        elif 'Ds' in data: # in case of not clean, flow directly from src is used.
            flow = data['Ds'].cuda()

        model = model.eval()
        with torch.no_grad(): 
            if time_meter is not None:
                torch.cuda.synchronize()
                start = time.time()

            if args.EVAL.LOOP:
                refined_cloud, s_pred = bcpd_loop(model, pos1, pos2, beta=beta, gamma=gamma, lmbd=lmbd, tolerance=tolerance, K=K)
                pred_flow = refined_cloud - pos1
                Inlier_src_pre = torch.sum(s_pred, dim=-1, keepdim=True)
                Inlier_ref_pre = torch.sum(s_pred, dim=-2)[:, :, None]

                # If clean, it is possible to back proj, this gives a flow without bcpd
                if args.DATASET.NOISE_TYPE == 'clean' or args.DATASET.NOISE_TYPE == 'outlier':
                    pred_flow_back_proj =  torch.bmm(s_pred, pos2) - pos1
                    epe3d_back_proj = torch.norm(pred_flow_back_proj - flow, dim = 2).mean()
            else:
                preds = model(pos1, pos2, norm1, norm2)
                s_pred = preds["P"]

                if hasattr(args, 'EVAL') and hasattr(args.EVAL, 'CYCLE') and args.EVAL.CYCLE:
                    preds_inv  = model(pos2, pos1, norm2, norm1)
                    s_pred_inv = preds_inv["P"]
                    s_pred = s_pred * s_pred_inv.permute(0, 2, 1)

                Inlier_src_pre = preds["SrcInlier"]
                Inlier_ref_pre  = preds["RefInlier"]

                # If clean, it is possible to back proj, this gives a flow without bcpd
                if args.DATASET.NOISE_TYPE == 'clean' or args.DATASET.NOISE_TYPE == 'outlier':
                    pred_flow_back_proj =  torch.bmm(s_pred, pos2) - pos1
                    epe3d_back_proj = torch.norm(pred_flow_back_proj - flow, dim = 2).mean()

                no_match = torch.sum(s_pred, dim=-1) < 0.5
                for batch in range(no_match.shape[0]):
                    s_pred[batch, no_match[batch], :] = 1e-9
                pred_flow = bcpd_refine(s_pred, pos1, pos2, beta=beta, gamma=gamma, lmbd=lmbd, K=K, tolerance=tolerance) - pos1
            
            if args.DATASET.NOISE_TYPE == 'holes':
                epe3d_back_proj = torch.norm(pred_flow - flow, dim = 2).mean()

            if time_meter is not None:
                torch.cuda.synchronize()
                diff = time.time() - start
                time_meter.update(diff, pred_flow.shape[0])

            permloss = perm_loss(s_pred, perm_mat, n1_gt, n2_gt)

            matched_points = torch.sum(perm_mat, dim=-1) > 0
            epe3d = torch.norm(pred_flow - flow, dim = 2)[matched_points].mean()

        s_perm_mat = hungarian(s_pred, n1_gt, n2_gt, Inlier_src_pre, Inlier_ref_pre)
        match_metrics = matching_accuracy(s_perm_mat, perm_mat, n1_gt)
        perform_metrics = compute_metrics_nonrigid(pred_flow + pos1, s_pred, pos1, pos2)

        metrics['epe3d'].update(epe3d.cpu().numpy(), n=args.batch_size)
        metrics['loss'].update(permloss.cpu().numpy(), n=args.batch_size)
        metrics['chamfer'].update(perform_metrics['chamfer_dist'], n=args.batch_size)

        if args.DATASET.NOISE_TYPE != 'crop': # Only exists if no crop
            metrics['epe3dbp'].update(epe3d_back_proj.cpu().numpy(), n=args.batch_size)

        for i in range(len(match_metrics['acc_gt'])):
            metrics['match_acc_gt'].update(match_metrics['acc_gt'][i])
            metrics['match_acc_pred'].update(match_metrics['acc_pred'][i])
            metrics['match_acc_gt_no_match'].update(match_metrics['acc_gt_no_match'][i])
            metrics['match_acc_pred_no_match'].update(match_metrics['acc_pred_no_match'][i])
            metrics['match_acc_pred_as_match'].update(match_metrics['acc_pred_as_match'][i])

    if time_meter is not None:
        str_time = f'Time avg {time_meter.avg}, {time_meter.count}'
        print(str_time)
        logger.info(str_time)

    str_out = '%s mean loss: %f mean epe: %f'%(blue('Evaluate'), metrics['loss'].avg, metrics['epe3d'].avg)
    print(str_out)
    logger.info(str_out)

    res_str = (' * EPE3D {epe3d_:.4f}\t'
               'EPE3DBP {epe3dbp_:.4f}\t'
               'CHAMFER {chamfer_:.4f}\t'
               'MATCHGT {matchgt_:.4f}\t'
               'MATCHPRED {matchpred_:.4f}\t'
               'MATCHGTNOMATCH {matchgtnomatch_:.4f}\t'
               'MATCHPREDNOMATCH {matchprednomatch_:.4f}\t'
               'MATCHPREDASMATCH {matchpredasmatch_:.4f}\t'
               .format(
                       epe3d_=metrics['epe3d'].avg,
                       epe3dbp_=metrics['epe3dbp'].avg,
                       chamfer_=metrics['chamfer'].avg,
                       matchgt_=metrics['match_acc_gt'].avg,
                       matchpred_=metrics['match_acc_pred'].avg,
                       matchgtnomatch_=metrics['match_acc_gt_no_match'].avg,
                       matchprednomatch_=metrics['match_acc_pred_no_match'].avg,
                       matchpredasmatch_=metrics['match_acc_pred_as_match'].avg
                       ))

    print(res_str)
    logger.info(res_str)


if __name__ == '__main__':
    main()




