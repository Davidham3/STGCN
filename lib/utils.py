# -*- coding:utf-8 -*-
import numpy as np
from scipy.sparse.linalg import eigs

def scaled_Laplacian(W):
    '''
    compute \tilde{L}
    
    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices
    
    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)
    
    '''
    
    assert W.shape[0] == W.shape[1]
    
    D = np.diag(np.sum(W, axis = 1))
    
    L = D - W
    
    lambda_max = eigs(L, k = 1, which = 'LR')[0].real
    
    return (2 * L) / lambda_max - np.identity(W.shape[0])

def cheb_polynomial(L_tilde, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}
    
    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)
    
    K: the maximum order of chebyshev polynomials
    
    Returns
    ----------
    cheb_polynomials: list(np.ndarray), length: K, from T_0 to T_{K-1}
    
    '''
    
    N = L_tilde.shape[0]
    
    cheb_polynomials = [np.identity(N), L_tilde.copy()]
    
    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])
        
    return cheb_polynomials

def get_adjacency_matrix(distance_df, num_of_vertices, normalized_k=0.1):
    """
    Parameters
    ----------
    distance_df: pd.DataFrame, contains distance between vertices, three columns [from, to, distance]

    num_of_vertices: int, number of vertices

    normalized_k: parameter of gaussian kernel
    
    Returns
    ----------
    A: np.ndarray, adjacency matrix

    """
    A = np.zeros((num_of_vertices, num_of_vertices), dtype = np.float32)

    # Fills cells in the matrix with distances.
    for row in distance_df.values:
        i, j = int(row[0]), int(row[1])
        A[i, j] = row[2]
        A[j, i] = row[2]

    # Calculates the standard deviation as theta.
    # compute the variance of the all distances which does not equal zero
    mask = (A == 0)

    tmp = A.flatten()
    var = np.var(tmp[tmp!=0])

    # normalization
    A = np.exp(- (A ** 2) / var)

    # drop the value less than threshold
    A[A < normalized_k] = 0
    A[mask] = 0
    
    return A
