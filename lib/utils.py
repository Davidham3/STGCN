# -*- coding:utf-8 -*-
# pylint: disable=no-member
import time
import csv
import json
import numpy as np
from scipy.sparse.linalg import eigs
from mxnet import autograd
from mxnet import nd
from mxboard import SummaryWriter
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


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

    D = np.diag(np.sum(W, axis=1))

    L = D - W

    lambda_max = eigs(L, k=1, which='LR')[0].real

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
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] -
                                cheb_polynomials[i - 2])

    return cheb_polynomials


def get_adj_matrix(adj_filename, num_of_vertices, sigma2=0.1, epsilon=0.5):
    '''
    Parameters
    ----------
    adj_filename: str, path of adj file

    num_of_vertices, int, number of vertices

    sigma2, epsilon: float

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    indices_dict: dict, map station_id into integers from 0 to num_of_vertices

    '''
    A = np.zeros(shape=(num_of_vertices, num_of_vertices))
    indices_dict = {}

    def transform_indices(id_):
        if id_ not in indices_dict:
            indices_dict[id_] = len(indices_dict)

    with open(adj_filename, 'r') as f:
        csv_reader = csv.reader(f)
        _ = csv_reader.__next__()
        for start, end, distance in csv_reader:
            start, end = int(start), int(end)
            transform_indices(start)
            transform_indices(end)
            A[indices_dict[start], indices_dict[end]] = float(distance)

    assert len(indices_dict) == num_of_vertices

    mask = (np.ones((num_of_vertices, num_of_vertices)) -
            np.identity(num_of_vertices))

    # normalization
    A = np.exp(- (A ** 2) / sigma2)
    A = A * (A >= epsilon) * mask

    return A, indices_dict


def data_preprocess(indices_dict, graph_signal_filename):
    '''
    Parameters
    ----------
    indices_dict: dict

    graph_signal_filename: str

    Returns
    ----------
    X: Graph signal matrix, np.ndarray, shape is
        (num_of_vertices, num_of_features, num_of_samples)
    '''

    # preprocessing graph signal data
    with open(graph_signal_filename, 'r') as f:
        data = json.loads(f.read().strip())

    random_key = str(list(indices_dict)[0])
    random_feature_name = list(data[random_key].keys())[0]
    num_of_vertices = len(indices_dict)

    # initialize the graph signal matrix
    # shape is (num_of_vertices, num_of_features, num_of_samples)
    X = np.empty(shape=(num_of_vertices,
                        len(data[random_key].keys()),
                        len(data[random_key][random_feature_name])))

    # idx is the index of the vertice
    for vertice, idx in indices_dict.items():
        vertice = str(vertice)
        X[idx, 0, :] = np.array(data[vertice]['flow'])
        X[idx, 1, :] = np.array(data[vertice]['occupy'])
        X[idx, 2, :] = np.array(data[vertice]['speed'])

    return X[:, 0: 1, :]


def build_dataset(graph_signal_matrix, num_points_for_train,
                  num_points_for_predict):
    '''
    Parameters
    ----------
    graph_signal_matrix: np.ndarray, graph signal matrix, shape is
        (num_of_vertices, num_of_features, num_of_samples)

    Returns
    ----------
    features: np.ndarray, shape is
        (num_of_samples, num_of_features,
         num_of_vertices, num_points_for_training)

    target: np.ndarray, shape is
        (num_of_samples, num_of_vertices, num_points_for_prediction)
    '''

    # generate the beginning index and the ending index of a sample,
    # which bounds (num_points_for_training + num_points_for_predicting) points
    indices = [(i, i + (num_points_for_train + num_points_for_predict))
               for i in range(
                   graph_signal_matrix.shape[2] -
                   (num_points_for_train + num_points_for_predict) + 1)]

    # save samples
    features, target = [], []
    for i, j in indices:
        features.append(
            graph_signal_matrix[:, :, i: i + num_points_for_train]
                .transpose((1, 0, 2)))
        target.append(graph_signal_matrix[:, 0, i + num_points_for_train: j])

    features = np.concatenate([np.expand_dims(i, 0) for i in features], axis=0)
    target = np.concatenate([np.expand_dims(i, 0) for i in target], axis=0)

    return features, target


def predict(net, x, num_points_for_train, num_points_for_predict):
    '''
    net

    x: mx.ndarray, shape is
        (batch_size, num_of_features, num_of_vertices, num_of_training_point)

    num_points_for_train: int

    num_points_for_predict: int

    Returns
    ----------
    mx.ndarray, shape is
        (batch_size, num_of_vertices, num_of_training_point)
    '''
    predictions = []
    for _ in range(num_points_for_predict):
        # pylint: disable=invalid-sequence-index
        p = nd.concat(x, *predictions, dim=-1)[:, :, :, -num_points_for_train:]
        predictions.append(net(p).expand_dims(1))
    return nd.concat(*predictions, dim=-1).squeeze(axis=1)


def train_model(net, sw, train_count, val_count,
                training_dataloader, validation_dataloader,
                testing_dataloader, epochs, loss_function,
                trainer, decay_interval=None, decay_rate=None):
    '''
    train the model

    Parameters
    ----------
    net: model which has been initialized

    sw: mxboard.SummarayWriter

    training_dataloader: gluon.data.dataloader.DataLoader
    validation_dataloader: gluon.data.dataloader.DataLoader
    testing_dataloader: gluon.data.dataloader.DataLoader

    Returns
    ----------
    train_loss_list: list[float], which contains loss of training

    val_loss_list: list[float], which contains loss of validation

    test_loss_list: list[float], which contains loss of testing

    '''
    for epoch in range(epochs):

        for x, y in training_dataloader:
            with autograd.record():
                l = loss_function(net(x), y)
            l.backward()
            trainer.step(x.shape[0])
            sw.add_scalar('training_loss',
                          value=l.mean().asscalar(),
                          global_step=train_count)
            train_count += 1

        val_loss_list = [loss_function(net(x), y).mean().asscalar()
                         for x, y in validation_dataloader]
        sw.add_scalar('val_loss',
                      value=sum(val_loss_list) / len(val_loss_list),
                      global_step=val_count)

        predictions, ground_truth = zip(*[(predict(net, x, 12, 12).asnumpy(),
                                          y.asnumpy())
                                          for x, y in testing_dataloader])
        predictions = np.concatenate(predictions, axis=0)
        ground_truth = np.concatenate(ground_truth, axis=0)
        predictions = predictions.reshape(predictions.shape[0], -1)
        ground_truth = ground_truth.reshape(ground_truth.shape[0], -1)

        sw.add_scalar('mae',
                      value=mean_absolute_error(ground_truth, predictions),
                      global_step=val_count)
        sw.add_scalar('rmse',
                      value=mean_squared_error(ground_truth, predictions)**0.5,
                      global_step=val_count)
        val_count += 1

        if (epoch + 1) % 5 == 0:
            filename = 'stgcn_params/stgcn.params_{}'.format(epoch)
            net.save_parameters(filename)

        if decay_interval:
            if (epoch + 1) % decay_interval == 0:
                trainer.set_learning_rate(trainer.learning_rate * decay_rate)

    return train_count, val_count

if __name__ == "__main__":
    adj_filename = 'data/test_data1/distance.csv'
    graph_signal_filename = 'data/test_data1/graph_signal_data_small.txt'
    A, indices_dict = get_adj_matrix(adj_filename, 307)
    X = data_preprocess(indices_dict, graph_signal_filename)
    print(A.sum(), X.sum())
