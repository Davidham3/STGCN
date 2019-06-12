# -*- coding:utf-8 -*-

import csv

import numpy as np

from scipy.sparse.linalg import eigs


def scaled_laplacian(W):
    '''
    Normalized graph Laplacian

    Parameters
    ----------
    W: np.ndarray, adjacency matrix,
       shape is (num_of_vertices, num_of_vertices)

    Returns
    ----------
    np.ndarray, shape is (num_of_vertices, num_of_vertices)

    '''

    num_of_vertices = W.shape[0]
    d = np.sum(W, axis=1)
    L = np.diag(d) - W
    for i in range(num_of_vertices):
        for j in range(num_of_vertices):
            if (d[i] > 0) and (d[j] > 0):
                L[i, j] = L[i, j] / np.sqrt(d[i] * d[j])
    # lambda_max \approx 2.0, the largest eigenvalues of L.
    lambda_max = eigs(L, k=1, which='LR')[0][0].real
    return 2 * L / lambda_max - np.identity(num_of_vertices)


def cheb_poly_approx(L, order_of_cheb):
    '''
    Chebyshev polynomials approximation

    Parameters
    ----------
    L: np.ndarray, scaled graph Laplacian,
       shape is (num_of_vertices, num_of_vertices)

    Returns
    ----------
    np.ndarray, shape is (num_of_vertices, order_of_cheb * num_of_vertices)

    '''

    if order_of_cheb == 1:
        return np.identity(L.shape[0])

    cheb_polys = [np.identity(L.shape[0]), L]

    for i in range(2, order_of_cheb):
        cheb_polys.append(2 * L * cheb_polys[i - 1] - cheb_polys[i - 2])

    return np.concatenate(cheb_polys, axis=-1)


def first_approx(adj):
    '''
    1st-order approximation

    Parameters
    ----------
    adj: np.ndarray, adjacency matrix,
         shape is (num_of_vertices, num_of_vertices)

    Returns
    ----------
    np.ndarray, shape is (num_of_vertices, num_of_vertices)

    '''
    A = adj + np.identity(adj.shape[0])
    sinvD = np.sqrt(np.diag(np.sum(A, axis=1)).I)
    # refer to Eq.5
    return np.identity(adj.shape[0]) + sinvD * A * sinvD


def weight_matrix(file_path, sigma2=0.1, epsilon=0.5, scaling=True):
    '''
    Load weight matrix

    Parameters
    ----------
    file_path: str, path of adjacency matrix file

    sigma2: float, default 0.1, scalar of matrix adj

    epsilon: float, default 0.5,
             thresholds to control the sparsity of matrix adj

    scaling: bool, default True, whether applies numerical scaling on adj

    Returns
    ----------
    np.ndarray, shape is (num_of_vertices, num_of_vertices)

    '''
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        adj = np.array([list(map(float, i)) for i in reader if i])

    # check whether adj is a 0/1 matrix.
    if set(np.unique(adj)) == {0, 1}:
        print('The input graph is a 0/1 matrix; set "scaling" to False.')
        scaling = False

    if scaling:
        adj = adj / 10
        mask = np.ones_like(adj) - np.identity(adj.shape[0])
        # refer to Eq.10
        exp = np.exp(- adj ** 2 / sigma2)
        return exp * (exp >= epsilon) * mask
    return adj
