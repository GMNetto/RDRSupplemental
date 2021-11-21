
import argparse
import sys 
import os 

import torch, numpy as np, glob, math, torch.utils.data, scipy.ndimage, multiprocessing as mp
import torch.nn.functional as F
import time
import pickle 
import datetime
import logging

from tqdm import tqdm 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import open3d as o3d

from models.loss import PermLoss, PermLossOcclusion
from models.bcpd import bcpd_refine
from models.models import SuperGlueModel
from utils.hungarian import hungarian

from metrics import stats_meter, matching_accuracy, compute_metrics_nonrigid

from pathlib import Path
from datasets import load_datasets
import cmd_args 

# From RMANet
def update_test_cache(metrics, path, current_iter, batch_size, val=True):
    if val:
        test_cache_file=path+'/result_cache.txt'
    else:
        test_cache_file=path+'/result_cache_train.txt'
    cf=open(test_cache_file,'a+')
    # The first number is the iteration times.
    cf.write(str(current_iter)+' ')
    cf.write(str(metrics['loss'].avg)+' ')
    cf.write(str(metrics['match_acc_gt'].avg)+' ')
    cf.write(str(metrics['match_acc_pred'].avg)+' ')
    cf.write(str(metrics['match_acc_gt_no_match'].avg)+' ')
    cf.write(str(metrics['match_acc_pred_no_match'].avg)+' ')
    if val:
        cf.write(str(metrics['epe3d'].avg)+' ')
        cf.write(str(metrics['chamfer'].avg)+' ')

    cf.write('\n')
    cf.close()

    update_pics(test_cache_file, path, val)
    

def update_pics(path, image_dir, val=True):
    
    cf=open(path,'r')
    lines=cf.readlines()
    x=[]
    y_loss=[]
    y_match_acc_gt=[]
    y_match_acc_pred=[]
    y_match_acc_gt_no_match=[]
    y_match_acc_pred_no_match=[]
    y_epe3d=[]
    y_chamfer=[]
    
    for i in range(len(lines)):
        if i%1==0:
            index = int(lines[i].split(' ')[0])
            loss = float(lines[i].split(' ')[1])
            match_acc_gt = float(lines[i].split(' ')[2])
            match_acc_pred = float(lines[i].split(' ')[3])
            match_acc_gt_no_match = float(lines[i].split(' ')[4])
            match_acc_pred_no_match = float(lines[i].split(' ')[5])
            if val:
                epe3d = float(lines[i].split(' ')[6])
                chamfer = float(lines[i].split(' ')[7])
            
            x.append(index)
            y_loss.append(loss)
            y_match_acc_gt.append(match_acc_gt)
            y_match_acc_pred.append(match_acc_pred)
            y_match_acc_gt_no_match.append(match_acc_gt_no_match)
            y_match_acc_pred_no_match.append(match_acc_pred_no_match)
            if val:
                y_epe3d.append(epe3d)
                y_chamfer.append(chamfer)

    fig = plt.figure(0)
    fig.clear()
    plt.title('loss')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.plot(x, y_loss, c='r', ls='-')
    f = 'loss'
    if not val:
        f += '_train'
    plt.savefig(image_dir+f'/{f}.png')
    
    fig = plt.figure(0)
    fig.clear()
    plt.title('match_acc_gt')
    plt.xlabel('iteration')
    plt.ylabel('match_acc_gt')
    plt.plot(x, y_match_acc_gt, c='#526922', ls='-')
    f = 'match_acc_gt'
    if not val:
        f += '_train'
    plt.savefig(image_dir+f'/{f}.png')
    
    fig = plt.figure(0)
    fig.clear()
    plt.title('match_acc_pred')
    plt.xlabel('iteration')
    plt.ylabel('match_acc_pred')
    plt.plot(x, y_match_acc_pred, c='#526922', ls='-')
    f = 'match_acc_pred'
    if not val:
        f += '_train'
    plt.savefig(image_dir+f'/{f}.png')

    fig = plt.figure(0)
    fig.clear()
    plt.title('match_acc_gt_no_match')
    plt.xlabel('iteration')
    plt.ylabel('match_acc_gt_no_match')
    plt.plot(x, y_match_acc_gt_no_match, c='#526922', ls='-')
    f = 'match_acc_gt_no_match'
    if not val:
        f += '_train'
    plt.savefig(image_dir+f'/{f}.png')
    
    fig = plt.figure(0)
    fig.clear()
    plt.title('match_acc_pred_no_match')
    plt.xlabel('iteration')
    plt.ylabel('match_acc_pred_no_match')
    plt.plot(x, y_match_acc_pred_no_match, c='#526922', ls='-')
    f = 'match_acc_pred_no_match'
    if not val:
        f += '_train'
    plt.savefig(image_dir+f'/{f}.png')
    
    if val:
        fig = plt.figure(0)
        fig.clear()
        plt.title('epe3d')
        plt.xlabel('iteration')
        plt.ylabel('epe3d')
        plt.plot(x, y_epe3d, c='#526922', ls='-')
        plt.savefig(image_dir+'/epe3d.png')
        
        fig = plt.figure(0)
        fig.clear()
        plt.title('chamfer')
        plt.xlabel('iteration')
        plt.ylabel('chamfer')
        plt.plot(x, y_chamfer, c='#526922', ls='-')
        plt.savefig(image_dir+'/chamfer.png')

def main():
    global args 
    args = cmd_args.parse_args_from_yaml(sys.argv[1])

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu if args.multi_gpu is None else '0,1'

    '''CREATE DIR'''
    experiment_dir = Path('./experiment/')
    experiment_dir.mkdir(exist_ok=True)
    
    file_dir = Path(str(experiment_dir) + f'/Serial_{args.model_name}_{args.dataset}-' + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
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
    file_handler = logging.FileHandler(str(log_dir) + '/train_%s.txt'%args.model_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('----------------------------------------TRAINING----------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)

    blue = lambda x: '\033[94m' + x + '\033[0m'

    print("Model Name", args.model_name)
    if args.model_name == "SuperGlue":
        model = SuperGlueModel(args)
    else:
        raise NotImplementedError(f'Model: {args.model_name} not implemented')
    
    # TODO: include normal features
    train_dataset = load_datasets.get_datasets(dataset_name = args.dataset,
                                  partition = 'train',
                                  num_points = args.num_points,
                                  unseen = args.DATASET.UNSEEN,
                                  noise_type = args.DATASET.NOISE_TYPE,
                                  rot_mag = args.DATASET.ROT_MAG,
                                  trans_mag = args.DATASET.TRANS_MAG,
                                  scale_mag = args.DATASET.SCALE_MAG,
                                  partial_p_keep = args.DATASET.PARTIAL_P_KEEP,
                                  crossval = False,
                                  train_part = True,
                                  fast_test = args.fast_test)

    logger.info('train_dataset: ' + str(train_dataset))

    train_loader = load_datasets.get_dataloader(train_dataset, batch_size=args.batch_size, workers=args.workers, shuffle=True)

    val_dataset = load_datasets.get_datasets(dataset_name = args.dataset, 
                                  partition = 'test',
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

    val_loader = load_datasets.get_dataloader(val_dataset, batch_size=args.batch_size, workers=args.workers, shuffle=True)

    model.cuda()

    perm_loss = PermLossOcclusion()

    if args.pretrain is not None:
        model.load_state_dict(torch.load(args.pretrain))
        print('load model %s'%args.pretrain)
        logger.info('load model %s'%args.pretrain)
    else:
        print('Training from scratch')
        logger.info('Training from scratch')

    pretrain = args.pretrain 
    try:
        init_epoch = int(pretrain[-14:-11]) if args.pretrain is not None else 0 
    except ValueError:
        init_epoch = 0

    # TODO: Update optimizer
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), 
                                     eps=1e-08, weight_decay=args.weight_decay)
                
    optimizer.param_groups[0]['initial_lr'] = args.learning_rate 
    # Temporary
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma, last_epoch = init_epoch - 1)
    LEARNING_RATE_CLIP = 1e-5 

    best_loss = 1000.0
    for epoch in range(init_epoch, args.epochs):
        lr = max(optimizer.param_groups[0]['lr'], LEARNING_RATE_CLIP)
        print('Learning rate:%f'%lr)
        logger.info('Learning rate:%f'%lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Accumulates all metrics
        metrics = stats_meter()
        statistics_step = 100

        running_since = time.time()

        optimizer.zero_grad()
        for i, data in tqdm(enumerate(train_loader, 0), total=len(train_loader), smoothing=0.9):
            pos1, pos2 = [_.cuda() for _ in data['Ps']]     #keypoints coordinate
            norm1, norm2 = [_.cuda() for _ in data['Fs']]     #featpoints values
            n1_gt, n2_gt = [_.cuda() for _ in data['ns']]     #keypoints number
            A1_gt, A2_gt = [_.cuda() for _ in data['As']]     #edge connect matrix
            perm_mat = data['gt_perm_mat'].cuda()             #permute matrix
            T1_gt, T2_gt = [_.cuda() for _ in data['Ts']]
            Inlier_src_gt, Inlier_ref_gt = [_.cuda() for _ in data['Ins']]

            model = model.train()
            preds = model(pos1, pos2, norm1, norm2)
            s_pred = preds["P"]
            Inlier_src_pre = preds["SrcInlier"]
            Inlier_ref_pre  = preds["RefInlier"]

            permloss = perm_loss(s_pred, perm_mat, n1_gt, n2_gt)
            loss = permloss

            s_perm_mat = hungarian(s_pred, n1_gt, n2_gt, Inlier_src_pre, Inlier_ref_pre)

            loss.backward()
            
            optimizer.step() 
            optimizer.zero_grad()

            match_metrics = matching_accuracy(s_perm_mat, perm_mat, n1_gt)
            
            metrics['loss'].update(loss.cpu().data.numpy(), n=args.batch_size)
            for batch_idx in range(len(match_metrics['acc_gt'])):
                metrics['match_acc_gt'].update(match_metrics['acc_gt'][batch_idx])
                metrics['match_acc_pred'].update(match_metrics['acc_pred'][batch_idx])
                metrics['match_acc_gt_no_match'].update(match_metrics['acc_gt_no_match'][batch_idx])
                metrics['match_acc_pred_no_match'].update(match_metrics['acc_pred_no_match'][batch_idx])
            if i % statistics_step == 0:
                running_speed = statistics_step * args.batch_size / (time.time() - running_since)
                step_stats_str = 'EPOCH {:<4} Iteration {:<4} {:>4.2f}sample/s Loss={:<8.4f} GT-Acc:{:.4f} Pred-Acc:{:.4f} GT-Acc-No-Match:{:.4f} Pred-Acc-No-Match:{:.4f}'.format(epoch, i, running_speed,
                                    metrics['loss'].avg,
                                    metrics['match_acc_gt'].avg,
                                    metrics['match_acc_pred'].avg,
                                    metrics['match_acc_gt_no_match'].avg,
                                    metrics['match_acc_pred_no_match'].avg)
                print(step_stats_str)
                logger.info(step_stats_str)    
                running_since = time.time()

        scheduler.step()

        str_out = 'EPOCH %d %s mean loss: %f'%(epoch, blue('train'), metrics['loss'].avg)
        print(str_out)
        logger.info(str_out)
        str_out1 = 'EPOCH %d %s mean match accuracy GT: %f'%(epoch, blue('train'), metrics['match_acc_gt'].avg)
        str_out2 = 'EPOCH %d %s mean match accuracy pred: %f'%(epoch, blue('train'), metrics['match_acc_pred'].avg)
        str_out3 = 'EPOCH %d %s mean no-match accuracy GT: %f'%(epoch, blue('train'), metrics['match_acc_gt_no_match'].avg)
        str_out4 = 'EPOCH %d %s mean no-match accuracy pred: %f'%(epoch, blue('train'), metrics['match_acc_pred_no_match'].avg)

        print(str_out1)
        print(str_out2)
        print(str_out3)
        print(str_out4)
        logger.info(str_out1)
        logger.info(str_out2)
        logger.info(str_out3)
        logger.info(str_out4)

        update_test_cache(metrics, str(log_dir), epoch, args.batch_size, val=False)

        eval_metrics = eval_sceneflow(model.eval(), val_loader, perm_loss)
        eval_epe3d = eval_metrics['epe3d'].avg
        eval_loss = eval_metrics['loss'].avg

        str_out = 'EPOCH %d %s mean epe3d: %f  mean eval loss: %f'%(epoch, blue('eval'), eval_epe3d, eval_loss)
        print(str_out)
        logger.info(str_out)
        str_out1 = 'EPOCH %d %s mean match accuracy GT: %f'%(epoch, blue('eval'), eval_metrics['match_acc_gt'].avg)
        str_out2 = 'EPOCH %d %s mean match accuracy pred: %f'%(epoch, blue('eval'), eval_metrics['match_acc_pred'].avg)
        str_out3 = 'EPOCH %d %s mean chamfer distance: %f'%(epoch, blue('eval'), eval_metrics['chamfer'].avg)
        str_out4 = 'EPOCH %d %s mean no-match accuracy GT: %f'%(epoch, blue('eval'), eval_metrics['match_acc_gt_no_match'].avg)
        str_out5 = 'EPOCH %d %s mean no-match accuracy pred: %f'%(epoch, blue('eval'), eval_metrics['match_acc_pred_no_match'].avg)
        print(str_out1)
        print(str_out2)
        print(str_out3)
        print(str_out4)
        print(str_out5)
        logger.info(str_out1)
        logger.info(str_out2)
        logger.info(str_out3)
        logger.info(str_out4)
        logger.info(str_out5)

        if eval_loss < best_loss or epoch == args.epochs - 1:
            best_loss = eval_loss
            torch.save(optimizer.state_dict(), '%s/optimizer.pth'%(checkpoints_dir))
            if args.multi_gpu is not None:
                torch.save(model.module.state_dict(), '%s/%s_%.3d_%.4f.pth'%(checkpoints_dir, args.model_name, epoch, best_loss))
            else:
                torch.save(model.state_dict(), '%s/%s_%.3d_%.4f.pth'%(checkpoints_dir, args.model_name, epoch, best_loss))
            logger.info('Save model ...')
            print('Save model ...')
        print('Best loss is: %.5f'%(best_loss))
        logger.info('Best loss is: %.5f'%(best_loss))

        update_test_cache(eval_metrics, str(log_dir), epoch, args.batch_size, val=True)


def eval_sceneflow(model, loader, perm_loss):
    metrics = stats_meter()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # BCPD refine parameters
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
        K = 200

    for batch_id, data in tqdm(enumerate(loader), total=len(loader), smoothing=0.9):
        
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
        else:
            print('using null flow')
            flow = torch.zeros_like(pos1)

        with torch.no_grad():
            preds = model(pos1, pos2, norm1, norm2)
            s_pred = preds["P"]
            Inlier_src_pre = preds["SrcInlier"]
            Inlier_ref_pre  = preds["RefInlier"]

            permloss = perm_loss(s_pred, perm_mat, n1_gt, n2_gt)

            no_match = torch.sum(s_pred, dim=-1) < 0.5
            for batch in range(no_match.shape[0]):
                s_pred[batch, no_match[batch], :] = 1e-9
            pred_flow = bcpd_refine(s_pred, pos1, pos2, beta=beta, gamma=gamma, lmbd=lmbd, tolerance=tolerance, K=K) - pos1

            matched_points = torch.sum(perm_mat, dim=-1) > 0
            epe3d = torch.norm(pred_flow - flow, dim = 2)[matched_points].mean()

        s_perm_mat = hungarian(s_pred, n1_gt-1, n2_gt-1, Inlier_src_pre, Inlier_ref_pre)
        match_metrics = matching_accuracy(s_perm_mat, perm_mat, n1_gt)
        perform_metrics = compute_metrics_nonrigid(pred_flow + pos1, s_pred, pos1, pos2)

        metrics['epe3d'].update(epe3d.cpu().numpy(), n=args.batch_size)
        metrics['loss'].update(permloss.cpu().numpy(), n=args.batch_size)
        metrics['chamfer'].update(perform_metrics['chamfer_dist'], n=args.batch_size)
        for i in range(len(match_metrics['acc_gt'])):
            metrics['match_acc_gt'].update(match_metrics['acc_gt'][i])
            metrics['match_acc_pred'].update(match_metrics['acc_pred'][i])
            metrics['match_acc_gt_no_match'].update(match_metrics['acc_gt_no_match'][i])
            metrics['match_acc_pred_no_match'].update(match_metrics['acc_pred_no_match'][i])

    return metrics

if __name__ == '__main__':
    main()




