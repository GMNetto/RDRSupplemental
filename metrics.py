import torch
import open3d
import numpy as np
from scipy.spatial.transform import Rotation
from datasets.se3 import transform

def stats_dict():
    stats=dict()
    stats['loss']=0.
    stats['dataloss']=0.
    stats['smoothloss']=0.
    stats['curvatureloss']=0.
    stats['flowloss']=0.
    stats['match_acc_gt']=0.
    stats['match_acc_pred']=0.
    stats['match_acc_gt_no_match']=0.
    stats['match_acc_pred_no_match']=0.
    stats['match_acc_pred_as_match']=0.
    stats['epe3d'] = 0.
    stats['chamfer'] = 0.
    stats['epe3dbp']=0.
    
    stats['r_mse']=0.
    stats['r_mae']=0.
    stats['t_mse']=0.
    stats['t_mae']=0.
    stats['err_r_deg']=0.
    stats['err_t']=0.

    return stats

def stats_meter():
    meters=dict()
    stats=stats_dict()
    for key,_ in stats.items():
        meters[key]=AverageMeter()
    return meters

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0.0
        self.sq_sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.sq_sum += val ** 2 * n
        self.var = self.sq_sum / self.count - self.avg ** 2


def to_numpy(tensor):
    """Wrapper around .detach().cpu().numpy() """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    else:
        raise NotImplementedError

def batch_matching_accuracy(pmat_pred, pmat_gt, ns, batch):
    match_num = torch.sum(pmat_pred[batch, :ns[batch]] * pmat_gt[batch, :ns[batch]]) + 1e-8
    gt_num = torch.sum(pmat_gt[batch, :ns[batch]]) + 1e-8
    pred_num = torch.sum(pmat_pred[batch, :ns[batch]]) + 1e-8
    return match_num, gt_num, pred_num

def batch_no_matching_accuracy(pmat_pred, pmat_gt, ns, batch):
    no_match_gt = torch.logical_not(torch.sum(pmat_gt[batch, :ns[batch]], dim=-1))
    no_match_pred = torch.logical_not(torch.sum(pmat_pred[batch, :ns[batch]], dim=-1))

    no_match_num = torch.sum(no_match_gt * no_match_pred)
    
    gt_num = torch.sum(no_match_gt) + 1e-8
    pred_num = torch.sum(no_match_pred) + 1e-8

    return no_match_num, gt_num, pred_num


def matching_accuracy(pmat_pred, pmat_gt, ns):
    """
    Matching Accuracy between predicted permutation matrix and ground truth permutation matrix.
    :param pmat_pred: predicted permutation matrix
    :param pmat_gt: ground truth permutation matrix
    :param ns: number of exact pairs
    :return: matching accuracy, matched num of pairs, total num of pairs
    """
    device = pmat_pred.device
    batch_num = pmat_pred.shape[0]

    pmat_gt = pmat_gt.to(device)

    assert torch.all((pmat_pred == 0) + (pmat_pred == 1)), 'pmat_pred can noly contain 0/1 elements.'
    assert torch.all((pmat_gt == 0) + (pmat_gt == 1)), 'pmat_gt should noly contain 0/1 elements.'
    assert torch.all(torch.sum(pmat_pred, dim=-1) <= 1) and torch.all(torch.sum(pmat_pred, dim=-2) <= 1)
    assert torch.all(torch.sum(pmat_gt, dim=-1) <= 1) and torch.all(torch.sum(pmat_gt, dim=-2) <= 1)

    match_num_list = []
    gt_num_list = []
    pred_num_list = []
    acc_gt = []
    acc_pred = []
    acc_gt_no_match = []
    acc_pred_no_match = []
    acc_pred_as_match = []
    for b in range(batch_num):
        match_num, gt_num, pred_num = batch_matching_accuracy(pmat_pred, pmat_gt, ns, b)
        no_match_num, no_gt_num, no_pred_num = batch_no_matching_accuracy(pmat_pred, pmat_gt, ns, b)

        match_num_list.append(match_num.cpu().numpy())
        gt_num_list.append(gt_num.cpu().numpy())
        pred_num_list.append(pred_num.cpu().numpy())

        acc_gt.append((match_num/gt_num).cpu().numpy())
        acc_pred.append((match_num/pred_num).cpu().numpy())

        acc_gt_no_match.append((no_match_num/no_gt_num).cpu().numpy())
        acc_pred_no_match.append((no_match_num/no_pred_num).cpu().numpy())
        
        acc_pred_as_match.append((pred_num/pmat_pred.shape[-2]))

    return {'acc_gt':    acc_gt,
            'acc_pred':  acc_pred,
            'match_num': match_num_list,
            'gt_num':    gt_num_list,
            'pred_num':  pred_num_list,
            'acc_gt_no_match': acc_gt_no_match,
            'acc_pred_no_match': acc_pred_no_match,
            'acc_pred_as_match': acc_pred_as_match}

def calcorrespondpc(pmat_pred, pc2_gt):
    # print('pmat', torch.max(pmat_pred))
    # print('calculate corr', pmat_pred.shape, pc2_gt.shape)
    pc2 = torch.zeros((pc2_gt.shape[0],pmat_pred.shape[1],pc2_gt.shape[2])).to(pc2_gt)
    # print('calculate corr', pmat_pred.shape, pc2_gt.shape, pc2.shape)
    pmat_pred_index = np.zeros((pc2_gt.shape[0], pc2_gt.shape[1]), dtype=int)
    
    for i in range(pmat_pred.shape[0]):
        pmat_predi_index1 = torch.where(pmat_pred[i])
        pmat_predi_index00 = torch.where(torch.sum(pmat_pred[i], dim=0) == 0)[0]  #n row sum->1，1024
        pmat_predi_index01 = torch.where(torch.sum(pmat_pred[i], dim=1) == 0)[0]  #n col sum->1024,1
        
        
        # print(torch.sum(pmat_pred[i], dim=0), torch.sum(torch.sum(pmat_pred[i], dim=0)))

        # print(torch.sum(pmat_pred[i], dim=1), torch.sum(torch.sum(pmat_pred[i], dim=1)))
        
        # positions in cloud 1 where there are matches followed by where it is 0
        # pc2[i, torch.cat((pmat_predi_index1[0], pmat_predi_index01))] = \
        #     pc2_gt[i, torch.cat((pmat_predi_index1[1], pmat_predi_index00))]
        den = torch.matmul(pmat_pred[i,...], pc2_gt[i,...]) 
        div = 1 / (torch.sum(pmat_pred[i], dim=1) + 1e-10)
        div = torch.diag_embed(1 / div)
        pc2[i,...] = torch.matmul(div, den)
        # print(den.shape, div.shape)
        # pc2[i,...] = den / div
        
        # pmat_pred_index[i, pmat_predi_index1[0].cpu().numpy()] = pmat_predi_index1[1].cpu().numpy()
        # pmat_pred_index[i, pmat_predi_index01.cpu().numpy()] = pmat_predi_index00.cpu().numpy()
    return pc2, pmat_pred_index

def square_distance(src, dst):
    return torch.sum((src[:, :, None, :] - dst[:, None, :, :]) ** 2, dim=-1)

def compute_metrics_nonrigid(P1_transformed, s_pred, P1_gt, P2_gt):
    # Chamfer distance
    dist_src = torch.min(square_distance(P1_transformed, P2_gt), dim=-1)[0]
    dist_ref = torch.min(square_distance(P2_gt, P1_transformed), dim=-1)[0]
    chamfer_dist = torch.mean(dist_src, dim=1) + torch.mean(dist_ref, dim=1)

    metrics = {'chamfer_dist': to_numpy(torch.mean(chamfer_dist))}

    return metrics

def compute_transform(s_perm_mat, P1_gt, P2_gt, R_gt, T_gt, viz=None, usepgm=True, userefine=False):
    if usepgm:
        pre_P2_gt, _ = calcorrespondpc(s_perm_mat, P2_gt)
        R_pre, T_pre = SVDslover(P1_gt.clone(), pre_P2_gt, s_perm_mat)
    else:
        pre_P2_gt = P2_gt
        R_pre = torch.eye(3,3).repeat(P1_gt.shape[0], 1, 1).to(P2_gt)
        T_pre = torch.zeros_like(T_gt).to(P2_gt)
    if userefine:
        R_pre, T_pre = refine_reg(R_pre, T_pre, P1_gt, pre_P2_gt)
    return R_pre, T_pre

def compute_rigid_metrics(s_perm_mat, P1_gt, P2_gt, R_gt, T_gt, viz=None, usepgm=True, userefine=False, R_pre=None, T_pre=None):
    # compute r,t
    if R_pre is None or T_pre is None:
        R_pre, T_pre = compute_transform(s_perm_mat, P1_gt, P2_gt, R_gt, T_gt, viz=viz, usepgm=usepgm, userefine=userefine)

    r_pre_euler_deg = npmat2euler(R_pre.detach().cpu().numpy(), seq='xyz')
    r_gt_euler_deg = npmat2euler(R_gt.detach().cpu().numpy(), seq='xyz')
    r_mse = np.mean((r_gt_euler_deg - r_pre_euler_deg) ** 2, axis=1)
    r_mae = np.mean(np.abs(r_gt_euler_deg - r_pre_euler_deg), axis=1)
    t_mse = torch.mean((T_gt - T_pre) ** 2, dim=1)
    t_mae = torch.mean(torch.abs(T_gt - T_pre), dim=1)

    # Rotation, translation errors (isotropic, i.e. doesn't depend on error
    # direction, which is more representative of the actual error)
    concatenated = concatenate(inverse(R_gt.cpu().numpy(), T_gt.cpu().numpy()),
                                       np.concatenate([R_pre.cpu().numpy(), T_pre.unsqueeze(-1).cpu().numpy()],
                                                      axis=-1))
    rot_trace = concatenated[:, 0, 0] + concatenated[:, 1, 1] + concatenated[:, 2, 2]
    residual_rotdeg = torch.acos(torch.clamp(0.5 * (rot_trace - 1), min=-1.0, max=1.0)) * 180.0 / np.pi
    residual_transmag = concatenated[:, :, 3].norm(dim=-1)

    # Chamfer distance
    # src_transformed = transform(pred_transforms, points_src)
    P1_transformed = torch.from_numpy(transform(torch.cat((R_pre, T_pre[:, :, None]), dim=2).detach().cpu().numpy(),
                                                P1_gt.detach().cpu().numpy())).to(P1_gt)
    
    dist_src = torch.min(square_distance(P1_transformed, P2_gt), dim=-1)[0]
    dist_ref = torch.min(square_distance(P2_gt, P1_transformed), dim=-1)[0]
    chamfer_dist = torch.mean(dist_src, dim=1) + torch.mean(dist_ref, dim=1)

    # Source distance
    P1_pre_trans = torch.from_numpy(transform(torch.cat((R_pre, T_pre[:, :, None]), dim=2).detach().cpu().numpy(),
                                              P1_gt.detach().cpu().numpy())).to(P1_gt)
    P1_gt_trans = torch.from_numpy(transform(torch.cat((R_gt, T_gt[:, :, None]), dim=2).detach().cpu().numpy(),
                                             P1_gt.detach().cpu().numpy())).to(P1_gt)
    dist_src = torch.min(square_distance(P1_pre_trans, P1_gt_trans), dim=-1)[0]
    presrc_dist = torch.mean(dist_src, dim=1)

    # Clip Chamfer distance
    clip_val = torch.Tensor([0.1]).cuda()
    P1_transformed = torch.from_numpy(transform(torch.cat((R_pre, T_pre[:, :, None]), dim=2).detach().cpu().numpy(),
                                                P1_gt.detach().cpu().numpy())).to(P1_gt)
    dist_src = torch.min(torch.min(torch.sqrt(square_distance(P1_transformed, P2_gt)), dim=-1)[0], clip_val)
    dist_ref = torch.min(torch.min(torch.sqrt(square_distance(P2_gt, P1_transformed)), dim=-1)[0], clip_val)
    clip_chamfer_dist = torch.mean(dist_src, dim=1) + torch.mean(dist_ref, dim=1)

    # correspondence distance
    P2_gt_copy, _ = calcorrespondpc(s_perm_mat, P2_gt.detach())
    inlier_src = torch.sum(s_perm_mat, axis=-1)[:, :, None]
    # inlier_ref = torch.sum(s_perm_mat, axis=-2)[:, :, None]
    P1_gt_trans_corr = P1_gt_trans.mul(inlier_src)
    P2_gt_copy_coor = P2_gt_copy.mul(inlier_src)
    correspond_dis=torch.sqrt(torch.sum((P1_gt_trans_corr-P2_gt_copy_coor)**2, dim=-1, keepdim=True))
    correspond_dis[inlier_src == 0] = np.nan

    metrics = {'r_mse': r_mse,
               'r_mae': r_mae,
               't_mse': to_numpy(t_mse),
               't_mae': to_numpy(t_mae),
               'err_r_deg': to_numpy(residual_rotdeg),
               'err_t': to_numpy(residual_transmag),
               'chamfer_dist': to_numpy(chamfer_dist),
               'pcab_dist': to_numpy(presrc_dist),
               'clip_chamfer_dist': to_numpy(clip_chamfer_dist),
               'pre_transform':np.concatenate((to_numpy(R_pre),to_numpy(T_pre)[:,:,None]),axis=2),
               'gt_transform':np.concatenate((to_numpy(R_gt),to_numpy(T_gt)[:,:,None]),axis=2),
               'cpd_dis_nomean':to_numpy(correspond_dis)}

    return metrics


def npmat2euler(mats, seq='zyx'):
    eulers = []
    for i in range(mats.shape[0]):
        r = Rotation.from_dcm(mats[i])
        eulers.append(r.as_euler(seq, degrees=True))
    return np.asarray(eulers, dtype='float32')


def inverse(rot: np.ndarray, trans: np.ndarray):
    """Returns the inverse of the SE3 transform

    Args:
        g: ([B,] 3/4, 4) transform

    Returns:
        ([B,] 3/4, 4) matrix containing the inverse

    """
    # rot = g[..., :3, :3]  # (3, 3)
    # trans = g[..., :3, 3]  # (3)

    inv_rot = np.swapaxes(rot, -1, -2)
    inverse_transform = np.concatenate([inv_rot, inv_rot @ -trans[..., None]], axis=-1)
    # if g.shape[-2] == 4:
    #     inverse_transform = np.concatenate([inverse_transform, [[0.0, 0.0, 0.0, 1.0]]], axis=-2)

    return inverse_transform

def concatenate(a: np.ndarray, b: np.ndarray):
    """ Concatenate two SE3 transforms

    Args:
        a: First transform ([B,] 3/4, 4)
        b: Second transform ([B,] 3/4, 4)

    Returns:
        a*b ([B, ] 3/4, 4)

    """

    r_a, t_a = a[..., :3, :3], a[..., :3, 3]
    r_b, t_b = b[..., :3, :3], b[..., :3, 3]

    r_ab = r_a @ r_b
    t_ab = r_a @ t_b[..., None] + t_a[..., None]

    concatenated = np.concatenate([r_ab, t_ab], axis=-1)

    # if a.shape[-2] == 4:
    #     concatenated = np.concatenate([concatenated, [[0.0, 0.0, 0.0, 1.0]]], axis=-2)

    return torch.tensor(concatenated)

def SVDslover(src_o, tgt_o, s_perm_mat):
    """Compute rigid transforms between two point sets

    Args:
        src_o (torch.Tensor): (B, M, 3) points
        tgt_o (torch.Tensor): (B, N, 3) points
        s_perm_mat (torch.Tensor): (B, M, N)

    Returns:
        Transform R (B, 3, 3) t(B, 3) to get from src_o to tgt_o, i.e. T*src = tgt
    """
    weights = torch.sum(s_perm_mat, dim=2)
    weights_normalized = weights[..., None] / (torch.sum(weights[..., None], dim=1, keepdim=True) + 1e-5)
    # print('weights_normalized', weights_normalized.shape)
    centroid_src_o = torch.sum(src_o * weights_normalized, dim=1)
    centroid_tgt_o = torch.sum(tgt_o * weights_normalized, dim=1)
    src_o_centered = src_o - centroid_src_o[:, None, :]
    tgt_o_centered = tgt_o - centroid_tgt_o[:, None, :]
    cov = src_o_centered.transpose(-2, -1) @ (tgt_o_centered * weights_normalized)

    # Compute rotation using Kabsch algorithm. Will compute two copies with +/-V[:,:3]
    # and choose based on determinant to avoid flips
    u, s, v = torch.svd(cov, some=False, compute_uv=True)
    rot_mat_pos = v @ u.transpose(-1, -2)
    v_neg = v.clone()
    v_neg[:, :, 2] *= -1
    rot_mat_neg = v_neg @ u.transpose(-1, -2)
    R = torch.where(torch.det(rot_mat_pos)[:, None, None] > 0, rot_mat_pos, rot_mat_neg)
    assert torch.all(torch.det(R) > 0)

    # Compute translation (uncenter centroid)
    t = -R @ centroid_src_o[:, :, None] + centroid_tgt_o[:, :, None]

    return R, t.view(s_perm_mat.shape[0], 3)


