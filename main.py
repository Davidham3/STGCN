# -*- coding:utf-8 -*-
import numpy as np
import random
import json
import time
import os

import mxnet as mx
from mxnet import nd
from mxnet import autograd
from mxnet import gluon
from mxnet.gluon import Trainer
from mxnet.gluon import Parameter
from mxnet.gluon import ParameterDict
from mxnet.gluon import nn

ctx = mx.gpu(0)
Co, kernel_size = 64, 3
embedding_size = 3
num_kernels1 = 16
num_points_for_train, num_points_for_predict = 24, 12
training_set_ratio = 0.7
epochs = 100
optimizer = 'adam'
learning_rate = 1e-3

def get_D(A):
    '''
    get degree matrix of A
    '''
    D = nd.zeros_like(A, ctx = ctx)
    for i in range(D.shape[0]):
        D[i, i] = nd.sum(A[:, i])
    return D

def get_D_(D):
    '''
    get D^(-frac{1}{2})
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
    A is the adjacency matrix of the road graph
    X is the feature matrix
    '''
    with open('remainID5min.json', 'r') as f:
        data = json.loads(f.read().strip())
        
    indices = sorted([int(i[:i.find(',')]) for i in data.keys()])
    indices_dict = {i: index for index, i in enumerate(indices)}
    
    new_data = {indices_dict[int(key[:key.find(',')])]: value for key, value in data.items()}
    
    with open('thre3.5twoPointIndex.txt', 'r') as f:
        t = (list(map(int, i.strip().split(','))) for i in f.readlines())
    
    A = nd.zeros(shape = (len(indices), len(indices)), ctx = ctx)
    for x, y in t:
        A[x, y] = 1.0
        
    X = nd.empty(shape = (len(indices), 3, 16992), ctx = ctx)
    for i in range(len(indices)):
        data = new_data[i]
        X[i, 0, :] = nd.array(data['flow'], ctx = ctx)
        X[i, 1, :] = nd.array(data['occupy'], ctx = ctx)
        X[i, 2, :] = nd.array(data['speed'], ctx = ctx)
    return A, X

def loss(output, target):
    return nd.sum((output - target) ** 2) / np.prod(output.shape)

class time_conv_block(nn.Block):
    def __init__(self, **kwargs):
        super(time_conv_block, self).__init__(**kwargs)
        self.conv1 = nn.Conv2D(Co, kernel_size, padding = 1, activation = 'relu', layout = 'NHWC')
        self.conv2 = nn.Conv2D(Co, kernel_size, padding = 1, activation = 'relu', layout = 'NHWC')
        self.conv3 = nn.Conv2D(Co, kernel_size = 1, layout = 'NHWC')
    
    def forward(self, x):
        t = self.conv1(x) + nd.sigmoid(self.conv2(x))
        return nd.relu(t + self.conv3(x))

class stgcn_block(nn.Block):
    def __init__(self, name_, **kwargs):
        super(stgcn_block, self).__init__(**kwargs)
        self.temporal1 = time_conv_block()
        self.temporal2 = time_conv_block()
        with self.name_scope():
            self.Theta1 = self.params.get('%s-Theta1'%(name_), shape = (Co, num_kernels1))
        self.batch_norm = nn.BatchNorm()

    def forward(self, x):
        t = self.temporal1(x)
        lfs = nd.dot(A_hat, t.transpose((1,0,2,3))).transpose((1,0,2,3))
        t2 = nd.relu(nd.dot(lfs, self.Theta1.data()))
        t3 = self.temporal2(t2)
        return self.batch_norm(t3)

class st_gcn_ijcai(nn.Block):
    def __init__(self, **kwargs):
        super(st_gcn_ijcai, self).__init__(**kwargs)
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

if __name__ == '__main__':
    A, X = data_preprocess()

    indices = [(i, i + (num_points_for_train + num_points_for_predict)) for i in range(X.shape[2] - (num_points_for_train + num_points_for_predict))]
    split_line = int(X.shape[2] * training_set_ratio)

    training_data, training_target = [], []
    for i, j in indices[:split_line]:
        training_data.append(X[:, :, i: i + num_points_for_train].transpose((0,2,1)))
        training_target.append(X[:, 0, i + num_points_for_train: j])

    test_data, test_target = [], []
    for i, j in indices[split_line:]:
        test_data.append(X[:, :, i: i + num_points_for_train].transpose((0,2,1)))
        test_target.append(X[:, 0, i + num_points_for_train: j])

    A_wave = A + nd.array(np.diag(np.ones(A.shape[0])), ctx = ctx)
    D_wave = get_D_wave(A_wave)
    D_wave_ = get_D_(D_wave)
    A_hat = nd.dot(nd.dot(D_wave_, A_wave), D_wave_)

    net = st_gcn_ijcai()
    net.initialize(ctx = ctx)

    trainer = Trainer(net.collect_params(), optimizer, {'learning_rate': learning_rate})
    batch_size = 16
    training_dataloader = gluon.data.DataLoader(gluon.data.ArrayDataset(training_data, training_target), batch_size = batch_size, shuffle = True)
    testing_dataloader = gluon.data.DataLoader(gluon.data.ArrayDataset(test_data, test_target), batch_size = batch_size, shuffle = False)

    if not os.path.exists('stgcn_params'):
        os.mkdir('stgcn_params')

    loss_list = []
    test_loss_list = []
    for epoch in range(epochs):
        t = time.time()
        
        loss_list_tmp = []
        for x, y in training_dataloader:
            with autograd.record():
                output = net(x)
                l = loss(output, y)
            l.backward()
            loss_list_tmp.append(l.asscalar())
            trainer.step(batch_size)
            
        loss_list.append( sum(loss_list_tmp) / len(loss_list_tmp) )
        
        test_loss_list_tmp = []
        for x, y in testing_dataloader:
            output = net(x)
            test_loss_list_tmp.append(loss(output, y).asscalar())
            
        test_loss_list.append( sum(test_loss_list_tmp) / len(test_loss_list_tmp) )
        
        print('training loss(MSE):', loss_list[-1])
        print('testing loss(MSE):', test_loss_list[-1])
        print('time:', time.time() - t)
        print()

        with open('results.log', 'a') as f:
            f.write('training loss(MSE): %s'%(loss_list[-1]))
            f.write('\n')
            f.write('testing loss(MSE): %s'%(test_loss_list[-1]))
            f.write('\n\n')
        
        if (epoch + 1) % 5 == 0:
            filename = 'stgcn_params/stgcn.params_%s'%(epoch)
            net.save_params(filename)
        
        if (epoch + 1) % 50 == 0:
            trainer.set_learning_rate(trainer.learning_rate * 0.5)