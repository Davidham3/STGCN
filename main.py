# -*- coding:utf-8 -*-
import numpy as np
import random
import json
import time
import os
import pandas as pd

import mxnet as mx
from mxnet import nd
from mxnet import autograd
from mxnet import gluon
from mxnet.gluon import Trainer
from mxnet.gluon import Parameter
from mxnet.gluon import ParameterDict
from mxnet.gluon import nn

##########
# configuration part

# choose which device you want to use, if you want to use CPU, set ctx = mx.cpu()
ctx = mx.gpu(0)

# number of vertices in your graph
num_of_vertices = 307

# we use a Gaussian kernel to normalize the adjacency matrix, epsilon is the threshold
epsilon = 0.5

# number of filters in time convolution, kernel_size is the size of these filters
Co, kernel_size = 64, 3

# number of features or the embedding size of your input
embedding_size = 3

# number of spatial filters
num_spatial_kernels = 16

# how many points you want to train the model and how many you want to predict
num_points_for_train, num_points_for_predict = 24, 12

# learning rate
learning_rate = 1e-3

# optimizer
optimizer = 'RMSprop'

# decay rate
decay_rate = 0.7

# decay interval
decay_interval = 5

# training epochs
epochs = 10

# batch_size
batch_size = 50
##########

def get_D(A):
    '''
    get degree matrix of A, A is the adjacency matrix
    '''
    D = nd.zeros_like(A, ctx = ctx)
    for i in range(D.shape[0]):
        D[i, i] = nd.sum(A[:, i])
    return D

def get_D_(D):
    '''
    get D^(-\frac{1}{2})
    '''
    D_ = nd.zeros_like(D, dtype = 'np.float32', ctx = ctx)
    for i in range(D_.shape[0]):
        if D[i, i] == 0:
            D_[i, i] = 1e-9 ** (-0.5)
        else:
            D_[i, i] = D[i, i] ** (-0.5)
    return D_

def get_L(A):
    '''
    get Laplacian, L = I_n−D^{−1/2}AD^{−1/2}
    '''
    D = get_D(A)
    D_ = get_D_(D)
    L = nd.array(np.diag(np.ones(A.shape[0])), ctx = ctx) - nd.dot(nd.dot(D_, A), D_)
    return L

def get_D_wave(W_wave):
    '''
    get normalized D
    '''
    t = nd.zeros_like(W_wave, ctx = ctx)
    for i in range(W_wave.shape[0]):
        t[i, i] = W_wave.sum(axis = 0)[i]
    return t

def data_preprocess():
    '''
    Returns
    ----------
    A: Adjacency matrix
    
    X: Graph signal matrix
    '''
    
    # distance information between vertices in the graph
    distance = pd.read_csv('data/distance.csv')

    # initialize an adjacency matrix of the graph with 0
    A = np.zeros((num_of_vertices, num_of_vertices))
    
    # put distances into the corresponding position
    for x, y, dis in distance.values:
        x_index = int(x)
        y_index = int(y)
        A[x_index, y_index] = dis
        A[y_index, x_index] = dis

    # compute the variance of the all distances which does not equal zero
    tmp = A.flatten()
    var = np.var(tmp[tmp!=0])

    # normalization
    A = np.exp(- (A ** 2) / var)

    # drop the value less than threshold
    A[A < epsilon] = 0
    
    # copy the value to mxnet ndarray
    A = nd.array(A, ctx = ctx)
    
    # preprocessing graph signal data
    with open('data/graph_signal_data_small.txt', 'r') as f:
        data = json.loads(f.read().strip())

    # initialize the graph signal matrix, shape is (num_of_vertices, num_of_features, num_of_samples)
    X = nd.empty(shape = (num_of_vertices,  # num_of_vertices
                          len(data[list(data.keys())[0]].keys()),  # num_of_features
                          len(list(data[list(data.keys())[0]].values())[0])),  # num_of_samples
                 ctx = ctx)
    
    # i is the index of the vertice
    for i in range(num_of_vertices):
        X[i, 0, :] = nd.array(data[str(i)]['flow'], ctx = ctx)
        X[i, 1, :] = nd.array(data[str(i)]['occupy'], ctx = ctx)
        X[i, 2, :] = nd.array(data[str(i)]['speed'], ctx = ctx)
    return A, X

def loss(output, target):
    '''
    loss function: MSE
    
    Parameters
    ----------
    output: mx.ndarray, output of the network, shape is (batch_size, num_of_vertices, num_points_for_predicting)
    
    target: mx.ndarray, target value of the prediction, shape is (batch_size, num_of_vertices, num_points_for_predicting)
    '''
    return nd.sum((output - target) ** 2) / np.prod(output.shape)

class time_conv_block(nn.Block):
    def __init__(self, **kwargs):
        super(time_conv_block, self).__init__(**kwargs)
        self.conv1 = nn.Conv2D(Co, (1, kernel_size), activation = 'relu', layout = 'NHWC')
        self.conv2 = nn.Conv2D(Co, (1, kernel_size), activation = 'relu', layout = 'NHWC')
        self.conv3 = nn.Conv2D(Co, kernel_size = (1, 3), layout = 'NHWC')
    
    def forward(self, x):
        t = self.conv1(x) + nd.sigmoid(self.conv2(x))
        return nd.relu(t + self.conv3(x))

class stgcn_block(nn.Block):
    def __init__(self, name_, **kwargs):
        super(stgcn_block, self).__init__(**kwargs)
        self.temporal1 = time_conv_block()
        self.temporal2 = time_conv_block()
        with self.name_scope():
            self.Theta1 = self.params.get('%s-Theta1'%(name_), shape = (Co, num_spatial_kernels))
        
        self.batch_norm = nn.BatchNorm()

    def forward(self, x):
        t = self.temporal1(x)
        lfs = nd.dot(A_hat, t.transpose((1,0,2,3))).transpose((1,0,2,3))
        t2 = nd.relu(nd.dot(lfs, self.Theta1.data()))
        t3 = self.temporal2(t2)
        return self.batch_norm(t3)

class STGCN_GLU(nn.Block):
    def __init__(self, **kwargs):
        super(STGCN_GLU, self).__init__(**kwargs)
        with self.name_scope():
            self.block1 = stgcn_block('block1')
            self.block2 = stgcn_block('block2')
            self.last_temporal = time_conv_block()
            self.fully = nn.Dense(num_points_for_predict, flatten = False)
    
    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.block2(out1)
        out3 = self.last_temporal(out2)
        return self.fully(out3.reshape((out3.shape[0], out3.shape[1], -1)))

def make_dataset(graph_signal_matrix):
    '''
    Parameters
    ----------
    graph_signal_matrix: graph signal matrix, shape is (num_of_vertices, num_of_features, num_of_samples)
    
    Returns
    ----------
    features: list[graph_signal_matrix], shape of each element is (num_of_vertices, num_of_features, num_points_for_training)
    
    target: list[graph_signal_matrix], shape of each element is (num_of_vertices, num_points_for_predicting)
    '''
    
    # generate the beginning index and the ending index of a sample, which contains (num_points_for_training + num_points_for_predicting) points
    indices = [(i, i + (num_points_for_train + num_points_for_predict)) for i in range(graph_signal_matrix.shape[2] - (num_points_for_train + num_points_for_predict) + 1)]
    
    # save samples
    features, target = [], []
    for i, j in indices:
        features.append(graph_signal_matrix[:, :, i: i + num_points_for_train].transpose((0,2,1)))
        target.append(graph_signal_matrix[:, 0, i + num_points_for_train: j])
    
    return features, target

def train_model(net, training_dataloader, validation_dataloader, testing_dataloader):
    '''
    train the model
    
    Parameters
    ----------
    net: model which has been initialized
    
    training_dataloader, validation_dataloader, testing_dataloader: gluon.data.dataloader.DataLoader
    
    Returns
    ----------
    train_loss_list: list(float), which contains loss values of training process
    
    val_loss_list: list(float), which contains loss values of validation process
    
    test_loss_list: list(float), which contains loss values of testing process
    
    '''
    train_loss_list = []
    val_loss_list = []
    test_loss_list = []
    for epoch in range(epochs):
        t = time.time()

        train_loss_list_tmp = []
        for x, y in training_dataloader:
            with autograd.record():
                output = net(x)
                l = loss(output, y)
            l.backward()
            train_loss_list_tmp.append(l.asscalar())
            trainer.step(batch_size)

        train_loss_list.append( sum(train_loss_list_tmp) / len(train_loss_list_tmp) )

        val_loss_list_tmp = []
        for x, y in validation_dataloader:
            output = net(x)
            val_loss_list_tmp.append(loss(output, y).asscalar())

        val_loss_list.append( sum(val_loss_list_tmp) / len(val_loss_list_tmp) )

        test_loss_list_tmp = []
        for x, y in testing_dataloader:
            output = net(x)
            test_loss_list_tmp.append(loss(output, y).asscalar())

        test_loss_list.append( sum(test_loss_list_tmp) / len(test_loss_list_tmp) )

        print('current epoch is %s'%(epoch + 1))
        print('training loss(MSE):', train_loss_list[-1])
        print('validation loss(MSE):', val_loss_list[-1])
        print('testing loss(MSE):', test_loss_list[-1])
        print('time:', time.time() - t)
        print()

        with open('results.log', 'a') as f:
            f.write('training loss(MSE): %s'%(train_loss_list[-1]))
            f.write('\n')
            f.write('validation loss(MSE): %s'%(val_loss_list[-1]))
            f.write('\n')
            f.write('testing loss(MSE): %s'%(test_loss_list[-1]))
            f.write('\n\n')

        if (epoch + 1) % 5 == 0:
            filename = 'stgcn_params/stgcn.params_%s'%(epoch)
            net.save_parameters(filename)

        if (epoch + 1) % decay_interval == 0:
            trainer.set_learning_rate(trainer.learning_rate * decay_rate)
    
    return train_loss_list, val_loss_list, test_loss_list

if __name__ == "__main__":
    A, X = data_preprocess()

    # training: validation: testing = 6: 2: 2
    split_line1 = int(X.shape[2] * 0.6)
    split_line2 = int(X.shape[2] * 0.8)

    train_original_data = X[:, :, :split_line1]
    val_original_data = X[:, :, split_line1: split_line2]
    test_original_data = X[:, :, split_line2:]

    training_data, training_target = make_dataset(train_original_data)
    val_data, val_target = make_dataset(val_original_data)
    testing_data, testing_target = make_dataset(test_original_data)

    # pre-computing
    A_wave = A + nd.array(np.diag(np.ones(A.shape[0])), ctx = ctx)
    D_wave = get_D_wave(A_wave)
    D_wave_ = get_D_(D_wave)
    A_hat = nd.dot(nd.dot(D_wave_, A_wave), D_wave_)

    # model initialization
    net = STGCN_GLU()
    net.initialize(ctx = ctx)

    trainer = Trainer(net.collect_params(), optimizer, {'learning_rate': learning_rate})
    training_dataloader = gluon.data.DataLoader(gluon.data.ArrayDataset(training_data, training_target), batch_size = batch_size, shuffle = True)
    validation_dataloader = gluon.data.DataLoader(gluon.data.ArrayDataset(val_data, val_target), batch_size = batch_size, shuffle = False)
    testing_dataloader = gluon.data.DataLoader(gluon.data.ArrayDataset(testing_data, testing_target), batch_size = batch_size, shuffle = False)

    if not os.path.exists('stgcn_params'):
        os.mkdir('stgcn_params')

    train_loss_list, val_loss_list, test_loss_list = train_model(net, training_dataloader, validation_dataloader, testing_dataloader)