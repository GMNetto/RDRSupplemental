'''
Deformation based on the work of BCPD - https://github.com/ohirose/bcpd
'''

import os
import argparse
import random
import igl
import glob
import h5py
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--in', default='out', type=str)
    parser.add_argument('--out', default='out', type=str)
    parser.add_argument('--outfile', default='outfile', type=str)

    return parser.parse_args()

def gram_matrix(Y, Sampled_Y, var, dim=3):
    G = np.stack([np.sum(np.square(Y - ts), axis=1) for ts in Sampled_Y])
    G = np.exp(-G / (2.0 * var))
    G = G.T
    return G

def nystroem(Y, k=300, var=2.0**2, lmd=50.0):
    ''' Approximates Eigen values and Vectors of the RBF Kernel matrix.

    Args:
        k: number of samples
        var: variance of the Gaussian kernels
        lmbd: deformation intensity, inversily proportional.
    '''

    if k == Y.shape[0]:
        permutation = np.arange(Y.shape[0])
    else:
        permutation = np.random.permutation(Y.shape[0])[:k]
    permutated_Y = Y[permutation, :]
    sampled_gram_matrix = gram_matrix(Y, permutated_Y, var)
    sampled_gram_matrix /= lmd
    
    square_sampled_gram_matrix = sampled_gram_matrix[permutation, :]
    
    eigen_values, eigen_vectors = np.linalg.eig(square_sampled_gram_matrix)
    eigen_values[eigen_values < 0] *= -1
    factor = k/Y.shape[0]
    sampled_eigen_values = eigen_values/factor
    
    sum_considered_kernel = np.dot(sampled_gram_matrix, eigen_vectors)
    sampled_eigen_vectors = np.sqrt(factor)*sum_considered_kernel/eigen_values
    
    return sampled_eigen_values, sampled_eigen_vectors

def sample_cloud(eigen_values, eigen_vectors, cloud, k):
    sampled_normal = np.random.standard_normal(size=(k, cloud.shape[1]))
    samples = np.dot(eigen_vectors, np.dot(np.diag(np.sqrt(eigen_values)), sampled_normal))
    return samples

def generate(clouds, out, out_file):
    n = clouds.shape[0]
    n_points = 2048
    t_array = np.zeros((n, 2*n_points, 3))

    number_samples = 50 # Number of Samples used during Nystrom approximation of the Eigen values, vectors

    for i in range(n):
        source_points = clouds[i,:n_points,...]

        sampled_eigen_values, sampled_eigen_vectors = nystroem(source_points, k = number_samples, var=1, lmd=np.random.randint(5, 30))
        target_points = source_points + sample_cloud(sampled_eigen_values, sampled_eigen_vectors, source_points, number_samples)

        while np.max(target_points) > 5: # This avoids degenerated deformations.
            sampled_eigen_values, sampled_eigen_vectors = nystroem(source_points, k = number_samples, var=1, lmd=np.random.randint(5, 30))
            target_points = source_points + sample_cloud(sampled_eigen_values, sampled_eigen_vectors, source_points, number_samples)

        sampled_eigen_values, sampled_eigen_vectors = nystroem(source_points, k = number_samples, var=1, lmd=np.random.randint(5, 30))
        new_source_points = source_points + sample_cloud(sampled_eigen_values, sampled_eigen_vectors, source_points, number_samples)

        while np.max(new_source_points) > 5: # This avoids degenerated deformations.
            sampled_eigen_values, sampled_eigen_vectors = nystroem(source_points, k = number_samples, var=1, lmd=np.random.randint(5, 30))
            new_source_points = source_points + sample_cloud(sampled_eigen_values, sampled_eigen_vectors, source_points, number_samples)

        mean = np.mean(new_source_points, axis=0)
        new_source_points -= mean

        mean = np.mean(target_points, axis=0)
        target_points -= mean

        t_array[i, :n_points, :] = new_source_points
        t_array[i, n_points:, :] = target_points

        if i % 250 == 0:
            print(i)
    
    print('dataset shape', t_array.shape)
    t_array = t_array.astype(np.float32).flatten()
    try:
        os.mkdir(out)
    except FileExistsError:
        pass

    t_array.tofile(os.path.join(out, out_file))


if __name__=='__main__':
    random.seed()
    args = parse_args()
    
    train_points_pair=np.fromfile(args.in, dtype = np.float32).reshape(-1,4096,3)
    
    generate(train_points_pair, args.out, args.outfile)
       


