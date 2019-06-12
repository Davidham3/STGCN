# -*- coding:utf-8 -*-

import csv

import numpy as np

from utils.math_utils import z_score


class Dataset(object):
    def __init__(self, data, stats):
        self.__data = data
        self.mean = stats['mean']
        self.std = stats['std']

    def __getitem__(self, key):
        return self.__data[key]

    def get_stats(self):
        return {'mean': self.mean, 'std': self.std}

    def z_inverse(self, type_):
        return self.__data[type_] * self.std + self.mean


def seq_gen(data_seq, n_frame):
    '''
    Generate data in the form of standard sequence unit.

    Parameters
    ----------
    data_seq: np.ndarray, time-series, shape is (length, num_of_vertices)


    n_frame: int, n_his + n_pred

    Returns
    ----------
    np.ndarray, shape is (length - n_frame + 1, n_frame, num_of_vertices, 1)

    '''

    data = np.zeros(shape=(data_seq.shape[0] - n_frame + 1,
                           n_frame, data_seq.shape[1], 1))
    for i in range(data_seq.shape[0] - n_frame + 1):
        data[i, :, :, 0] = data_seq[i: i + n_frame, :]
    return data


def data_gen(file_path, n_frame=24):
    '''
    Source file load and dataset generation.

    Parameters
    ----------
    file_path: str, path of time series data

    n_frame: int, n_his + n_pred

    Returns
    ----------
    Dataset, dataset that contains training, validation and test with stats.

    '''

    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        data_seq = np.array([list(map(float, i)) for i in reader if i])

    num_of_samples = data_seq.shape[0]
    splitting_line1 = int(num_of_samples * 0.6)
    splitting_line2 = int(num_of_samples * 0.8)

    seq_train = seq_gen(data_seq[: splitting_line1], n_frame)
    seq_val = seq_gen(data_seq[splitting_line1: splitting_line2], n_frame)
    seq_test = seq_gen(data_seq[splitting_line2:], n_frame)

    mean = np.mean(seq_train)
    std = np.std(seq_train)
    x_stats = {'mean': mean, 'std': std}

    x_train = z_score(seq_train, mean, std)
    x_val = z_score(seq_val, mean, std)
    x_test = z_score(seq_test, mean, std)

    x_data = {'train': x_train, 'val': x_val, 'test': x_test}
    dataset = Dataset(x_data, x_stats)
    return dataset
