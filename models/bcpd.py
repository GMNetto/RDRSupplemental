import torch
import subprocess
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch import Tensor
import numpy as np
import math
import time

def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, dim=-1)[:, :, None]
    dist += torch.sum(dst ** 2, dim=-1)[:, None, :]
    return dist

def calculate_sigma2(Y, X, gamma):
    distances = square_distance(Y, X)
    return torch.sum(distances, dim=[-2, -1]) / (Y.shape[1] * X.shape[1] * X.shape[2])

def gramdec(Y, k=300, var=2.0**2):
    if k == Y.shape[1]:
        permutation = torch.arange(Y.shape[-2])
    else:
        permutation = torch.randperm(Y.shape[-2])[:k]
    permutated_Y = Y[:,permutation, :]
    sampled_gram_matrix = square_distance(Y, permutated_Y)
    sampled_gram_matrix = torch.exp(- 0.5 * sampled_gram_matrix / (var ** 2))
    square_sampled_gram_matrix = sampled_gram_matrix[:, permutation, :]
        
    eigen_values, eigen_vectors = torch.symeig(square_sampled_gram_matrix, eigenvectors=True)
    factor = k/float(Y.shape[-2])
    sampled_eigen_values = eigen_values/factor
    sum_considered_kernel = torch.bmm(sampled_gram_matrix, eigen_vectors)
    for i in range(sum_considered_kernel.shape[0]):
        sum_considered_kernel[i, ...] = sum_considered_kernel[i, ...] / eigen_values[i,...]
    
    sampled_eigen_vectors = math.sqrt(factor)*sum_considered_kernel
    
    return sampled_eigen_vectors, sampled_eigen_values

def calculate_G(Y, beta):
    factor = -0.5 / (beta**2)
    distances = square_distance(Y, Y)
    return torch.exp(factor * distances)

def new_sigma2(X, P, Y_hat, nu, nu_line, N_hat):
    p0 = torch.bmm(X.transpose(1,2), torch.bmm(nu_line, X))
    p0 = torch.diagonal(p0, dim1=-2, dim2=-1).sum(-1) 
    
    p1 = torch.bmm(X.transpose(1,2), torch.bmm(P.transpose(1,2), Y_hat))
    p1 = torch.diagonal(p1, dim1=-2, dim2=-1).sum(-1)
    
    p2 = torch.bmm(Y_hat.transpose(1,2), torch.bmm(nu, Y_hat))
    
    p2 = torch.diagonal(p2, dim1=-2, dim2=-1).sum(-1)
                
    p3 = p0 - 2*p1 + p2

    new_sigma2 = (1/(N_hat*X.shape[-1])) * p3 

#     return torch.max(new_sigma2, torch.ones_like(new_sigma2) * 0.0001)
    return new_sigma2
    
def rigid(u_hat, x_hat, d_nu, Nu_hat):
    nu_normalized = d_nu / (Nu_hat[:, None, None] + 1e-6)
    u_line = torch.sum(torch.bmm(nu_normalized, u_hat), dim=1)
    x_line = torch.sum(torch.bmm(nu_normalized, x_hat), dim=1)
    
    u_centered = u_hat - u_line[:, None, :]
    x_centered = x_hat - x_line[:, None, :]
        
    cov = torch.bmm(u_centered.transpose(-2, -1), torch.bmm(nu_normalized, x_centered ))
    cov_uu = u_centered.transpose(-2, -1) @ torch.bmm(nu_normalized, u_centered)
 
    # Compute rotation using Kabsch algorithm. Will compute two copies with +/-V[:,:3]
    # and choose based on determinant to avoid flips
    u, s, v = torch.svd(cov, some=False, compute_uv=True)

    rot_mat_pos = v @ u.transpose(-1, -2)
    v_neg = v.clone()
    v_neg[:, :, 2] *= -1
    rot_mat_neg = v_neg @ u.transpose(-1, -2)
    R = torch.where(torch.det(rot_mat_pos)[:, None, None] > 0, rot_mat_pos, rot_mat_neg)
    assert torch.all(torch.det(R) > 0)

    scale = torch.bmm(R, cov).diagonal(dim1=-1,dim2=-2,offset=0).sum(dim=-1) / cov_uu.diagonal(dim1=-1,dim2=-2,offset=0).sum(dim=-1)
    # Compute translation (uncenter centroid)
    t = scale[:, None, None] * -R @ u_line[:, :, None] + x_line[:, :, None]

    return scale, R, t.view(d_nu.shape[0], 3)

def calculate_pout(X):
    maxes, _ = torch.max(X, dim=1)
    mins, _ = torch.min(X, dim=1)
    volume = torch.prod(maxes - mins, dim=-1)
    return 1/volume

def calculate_P(am, Phi, pout, w):
    summed_Phi = (1-w) * am * torch.sum(Phi, dim=-2, keepdim=True)
    num = (1 - w) * Phi * am
    den = w*pout[:, None, None] + summed_Phi
    return num/den

def calculate_Phi(X, Y_hat, sigma2):
    norm = 1 / (2 * math.pi * sigma2) ** (X.shape[2]*0.5)
    var_ = -0.5/sigma2

    distances = square_distance(Y_hat, X)
    distances = var_[:, None, None] * distances

    # Include eps for numerial stability
    return torch.exp(distances) * norm[:, None, None] + 1e-9

def apply_rigid(Y, R, s, t):
    Y = s[:, None, None] * torch.bmm(R, Y.transpose(1,2))
    Y += t.unsqueeze(2)
    return Y.transpose(1,2)

def bcpd(Y, X, beta, gamma, lmbd, w, K=0, max_iter=50, tolerance=1e-3):
    if K <= 0:
        G = calculate_G(Y, beta)
    else:
        Q, L = gramdec(Y, K, beta)
        d_L_inv = torch.diag_embed(1/L)
        L = torch.diag_embed(L)
    sigma2 = (gamma * torch.sqrt(calculate_sigma2(Y, X, gamma)))**2
    pout = calculate_pout(X)
    am = 1 / Y.shape[1]
    Y_hat = Y
    
    R = torch.eye(3)
    R = R.reshape((1, 3, 3))
    R = R.repeat(Y.shape[0], 1, 1).to(Y.device)
    t = torch.zeros(3)[None]
    t = t.repeat(Y.shape[0], 1).to(Y.device)

    s = torch.ones(Y.shape[0]).to(Y.device)
    
    for i in range(max_iter):
        Phi = calculate_Phi(X, Y_hat, sigma2)

        nu_phi = torch.sum(Phi, dim=-1)
        P = calculate_P(am, Phi, pout, w)
        nu = torch.sum(P, dim=-1)
        nu_line = torch.sum(P, dim=-2)
        N_hat = torch.sum(nu, dim=-1)
        d_nu = torch.diag_embed(nu)
        d_nu_line = torch.diag_embed(nu_line)
        d_nu_inv = torch.diag_embed(1 / nu)
        X_hat = torch.bmm(d_nu_inv, torch.bmm(P, X))

        X_hat_b = X_hat.transpose(1,2) - t.unsqueeze(2)
        residual = (1/s[:, None, None]) * torch.bmm(R.transpose(1,2), X_hat_b).transpose(1,2) - Y
        
        cc = (lmbd / (s * s)[:, None, None]) * sigma2[:, None, None]
        if K <= 0:
            G1 = cc * d_nu_inv  + G
            result, _ = torch.solve(residual, G1)
            nuhat = torch.bmm(G, result)
        else:
            C = torch.bmm(d_nu, Q)
            B = torch.bmm(C.transpose(1,2), residual)
            A = cc * d_L_inv + torch.bmm(Q.transpose(1,2), C)
            B, _ = torch.solve(B, A)
            W = 1/cc * (torch.bmm(d_nu, residual) - torch.bmm(C, B))
            nuhat = torch.bmm(Q, torch.bmm(L, torch.bmm(Q.transpose(1,2), W)))
            
        Y_hat = Y + nuhat
        
        s, R, t = rigid(Y_hat, X_hat, d_nu, N_hat)
        Y_hat = apply_rigid(Y_hat, R, s, t)
        
        n_sigma2 = new_sigma2(X, P, Y_hat, d_nu, d_nu_line, N_hat)

        if torch.all(torch.abs(n_sigma2 - sigma2) < tolerance):
            break
        sigma2 = n_sigma2
        # print('rigid\n', s, R, t)
    return sigma2, d_nu_inv, R, X_hat, t, s, Y_hat

def calculate_GG(Y_, Y, beta):
    factor = -0.5 / (beta**2)
    distances = square_distance(Y_, Y)
    return torch.exp(factor * distances)

def down_up_bcpd(Y, X, beta, gamma, lmbd, w, K=0, L=0, sampled_size=0, max_iter=50, tolerance=1e-4):
    # Used for sampling/downsample
    from models.lib import pointnet2_utils as pointutils
    
    # Use random sampling, in the project use furthest point sampling
    if sampled_size == 0:
        down_Y = Y
        down_X = X
    else:
        Y_T = Y.transpose(1,2).contiguous()
        fps_idx = pointutils.furthest_point_sample(Y, sampled_size)  # [B, N]
        down_Y = pointutils.gather_operation(Y_T, fps_idx).transpose(1,2)  # [B, C, N]

        X_T = X.transpose(1,2).contiguous()
        fps_idx = pointutils.furthest_point_sample(X, sampled_size)  # [B, N]
        down_X = pointutils.gather_operation(X_T, fps_idx).transpose(1,2)  # [B, C, N]

        if L == 0:
            GYZ = calculate_GG(Y, down_Y, beta) 
        else:
            fps_idx = pointutils.furthest_point_sample(Y, L)  # [B, N]
            down_L = pointutils.gather_operation(Y_T, fps_idx).transpose(1,2)

            GUU = calculate_G(down_L, beta)
            GYU = calculate_GG(Y, down_L, beta) 
            GUZ = calculate_GG(down_L, down_Y, beta) 
    
    sigma2, d_nu_inv, R, X_hat, t, s, Y_hat = bcpd(down_Y, down_X, beta, gamma, lmbd, w, K, max_iter, tolerance)
    if sampled_size == 0:
        return Y_hat
    
    X_hat_b = X_hat.transpose(1,2) - t.unsqueeze(2)
    E = (1/s[:, None, None]) * torch.bmm(R.transpose(1,2), X_hat_b).transpose(1,2) - down_Y
    cc = (lmbd / s[:, None, None] ** 2) * sigma2[:, None, None]
    psi = cc * d_nu_inv
    if L == 0:
        GZZ = calculate_G(down_Y, beta)
        V_hat = torch.bmm(torch.bmm(GYZ, torch.inverse(GZZ + psi)), E)
        Y_hat = Y + V_hat
        return apply_rigid(Y_hat, R, s, t)
    
    Psi_inv = torch.diag_embed((1/torch.diagonal(d_nu_inv, dim1=-2, dim2=-1)))
    Q, L = gramdec(down_Y, L, beta)
    A = cc * torch.diag_embed(1/L) + torch.bmm(Q.transpose(1,2), torch.bmm(Psi_inv, Q))
    B = torch.bmm(Q.transpose(1,2), torch.bmm(Psi_inv, E))
    B, _ = torch.solve(B, A)
    Q = torch.bmm(Psi_inv, Q)
    W = torch.bmm(Psi_inv, E) - torch.bmm(Q, B)
    W = (1/cc) * W
    
    result, _ = torch.solve(torch.bmm(GUZ, W), GUU)
    Y_hat = Y + torch.bmm(GYU, result)
    return apply_rigid(Y_hat, R, s, t)


def bcpd_refine(Phi, Y, X, beta, gamma, lmbd, K=200, max_iter=50, tolerance=1e-3):
    low_rank = False
    if K<=0:
        G = calculate_G(Y, beta)
    else:
        low_rank = True
        Q, L = gramdec(Y, K, beta)
        d_L_inv = torch.diag_embed(1/L)
        L = torch.diag_embed(L)

    sigma2 = (gamma * torch.sqrt(calculate_sigma2(Y, X, gamma))**2)

    Y_hat = Y.clone()
    P = Phi
    nu = torch.sum(P, dim=-1)
    nu_line = torch.sum(P, dim=-2)
    N_hat = torch.sum(nu, dim=-1)
    d_nu = torch.diag_embed(nu)
    d_nu_line = torch.diag_embed(nu_line)
    d_nu_inv = torch.diag_embed(1 / nu)
    X_hat = torch.bmm(d_nu_inv, torch.bmm(P, X))
    
    R = torch.eye(3)
    R = R.reshape((1, 3, 3))
    R = R.repeat(Phi.shape[0], 1, 1).to(Phi.device)
    t = torch.zeros(3)[None]
    t = t.repeat(Phi.shape[0], 1).to(Phi.device)

    s = torch.ones(Y.shape[0]).to(Y.device)

    converged = torch.zeros(Y.shape[0], dtype=torch.bool).to(Y.device)
    
    for i in range(max_iter):
        # print('BCPD iter ',i, sigma2, s)
        X_hat_b = X_hat.transpose(1,2) - t.unsqueeze(2)
        residual = (1/s[:, None, None]) * torch.bmm(R.transpose(1,2), X_hat_b).transpose(1,2) - Y

        cc = (lmbd / (s * s)[:, None, None]) * sigma2[:, None, None]
        if not low_rank:
            G1 = cc * d_nu_inv  + G
            result, _ = torch.solve(residual, G1)
            nuhat = torch.bmm(G, result)
        else:
            C = torch.bmm(d_nu, Q)
            B = torch.bmm(C.transpose(1,2), residual)
            A = cc * d_L_inv + torch.bmm(Q.transpose(1,2), C)
            B, _ = torch.solve(B, A)
            W = 1/cc * (torch.bmm(d_nu, residual) - torch.bmm(C, B))
            nuhat = torch.bmm(Q, torch.bmm(L, torch.bmm(Q.transpose(1,2), W)))
        Y_hat_tmp = Y + nuhat
        
        s, R, t = rigid(Y_hat_tmp, X_hat, d_nu, N_hat)
        Y_hat_tmp = apply_rigid(Y_hat_tmp, R, s, t)

        n_sigma2 = new_sigma2(X, P, Y_hat_tmp, d_nu, d_nu_line, N_hat)

        n_converged = torch.abs(n_sigma2 - sigma2) < tolerance
        to_update = torch.logical_not(converged)
        converged = torch.logical_or(n_converged, converged)
        
        Y_hat[to_update] = Y_hat_tmp[to_update]
        # Y_hat = Y_hat_tmp
        # if torch.all(n_converged):
        if torch.all(converged):
            break
        
        sigma2 = n_sigma2
    
    return Y_hat


def bcpd_loop(model, Y, X, beta, gamma, lmbd, K=200, max_iter=10, tolerance=1e-3, initial_rigid=False):
    if K <= 0:
        G = calculate_G(Y, beta)
    else:
        Q, L = gramdec(Y, K, beta)
        d_L_inv = torch.diag_embed(1/L)
        L = torch.diag_embed(L)

    Y_hat = Y
    Y_base = Y

    R = torch.eye(3)
    R = R.reshape((1, 3, 3))
    R = R.repeat(Y.shape[0], 1, 1).to(Y.device)
    t = torch.zeros(3)[None]
    t = t.repeat(Y.shape[0], 1).to(Y.device)
    s = torch.ones(Y.shape[0]).to(Y.device)

    n_iter = 0
    # last_sigma2 = torch.tensor(0)
    sigma2 = (gamma * torch.sqrt(calculate_sigma2(Y_base, X, gamma))**2)
    last_sigma2 = torch.zeros_like(sigma2)
    P = None 
    outer_iter = 0

    outer_converged = torch.zeros(Y.shape[0], dtype=torch.bool).to(Y.device)
    while True:
        with torch.no_grad():
            preds = model(Y_hat, X, Y_hat, X)
        P = preds["P"]

        no_match = torch.sum(P, dim=-1) < 0.5
        for batch in range(no_match.shape[0]):
            P[batch, no_match[batch], :] = 1e-9

        nu = torch.sum(P, dim=-1)
        nu_line = torch.sum(P, dim=-2)
        N_hat = torch.sum(nu, dim=-1)
        d_nu = torch.diag_embed(nu)
        d_nu_line = torch.diag_embed(nu_line)
        d_nu_inv = torch.diag_embed(1 / nu)
        X_hat = torch.bmm(d_nu_inv, torch.bmm(P, X))

        converged = torch.zeros(Y.shape[0], dtype=torch.bool).to(Y.device)
        
        while True:

            X_hat_b = X_hat.transpose(1,2) - t.unsqueeze(2)
            residual = (1/s[:, None, None]) * torch.bmm(R.transpose(1,2), X_hat_b).transpose(1,2) - Y_base

            cc = (lmbd / (s * s)[:, None, None]) * sigma2[:, None, None]
            if K <= 0:
                G1 = cc * d_nu_inv  + G
                result, _ = torch.solve(residual, G1)
                nuhat = torch.bmm(G, result)
            else:    
                C = torch.bmm(d_nu, Q)
                B = torch.bmm(C.transpose(1,2), residual)
                A = cc * d_L_inv + torch.bmm(Q.transpose(1,2), C)  
                B, _ = torch.solve(B, A)
                W = 1/cc * (torch.bmm(d_nu, residual) - torch.bmm(C, B))
                nuhat = torch.bmm(Q, torch.bmm(L, torch.bmm(Q.transpose(1,2), W)))
            Y_hat_tmp = Y_base + nuhat
            
            s, R, t = rigid(Y_hat_tmp, X_hat, d_nu, N_hat)
            Y_hat_tmp = apply_rigid(Y_hat_tmp, R, s, t)

            n_sigma2 = new_sigma2(X, P, Y_hat_tmp, d_nu, d_nu_line, N_hat)
            
            # Leaving loop for iter
            n_iter += 1
            if n_iter >= max_iter:
                return Y_hat, P


            n_converged = torch.abs(n_sigma2 - sigma2) < tolerance
            to_update = torch.logical_and(torch.logical_not(converged), torch.logical_not(outer_converged))
            converged = torch.logical_or(n_converged, converged)
            
            Y_hat[to_update] = Y_hat_tmp[to_update]
            # Y_hat = Y_hat_tmp
            # if torch.all(n_converged):
            if torch.all(converged):
                break


            # if torch.all(torch.abs(n_sigma2 - sigma2) < (tolerance)):
            #     sigma2 = n_sigma2
            #     Y_hat = Y_hat_tmp
            #     break


            # if torch.all(n_sigma2 - sigma2 > (tolerance)):
            #     break

            sigma2 = n_sigma2
            # Y_hat = Y_hat_tmp

        n_converged = torch.abs(sigma2 - last_sigma2) < tolerance
        outer_converged = torch.logical_or(n_converged, outer_converged)
        last_sigma2 = sigma2

        if torch.all(outer_converged):
            return Y_hat, P
        else:
            outer_iter += 1

        # if torch.all(torch.abs(last_sigma2 - sigma2) < tolerance):
        #     return Y_hat, P
        # else:
        #     outer_iter += 1
        #     last_sigma2 = sigma2

    return Y_hat, P
