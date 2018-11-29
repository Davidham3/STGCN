# -*- coding:utf-8 -*-

import mxnet as mx
from mxnet.gluon import nn
from mxnet import nd
from mxnet import autograd
from mxnet import gluon

import pandas as pd
from lib.utils import *

class cheb_conv(nn.Block):
    '''
    K-order chebyshev graph convolution
    '''
    def __init__(self, num_of_filters, K, cheb_polys, **kwargs):
        '''
        Parameters
        ----------
        num_of_filters: int
        
        num_of_features: int, num of input features
        
        K: int, up K - 1 order chebyshev polynomials will use in this convolution
        
        '''
        super(cheb_conv, self).__init__(**kwargs)
        self.K = K
        self.num_of_filters = num_of_filters
        self.cheb_polys = cheb_polys
        
        # shape of theta is (self.K, num_of_features, num_of_filters)
        with self.name_scope():
            self.Theta = self.params.get('Theta', allow_deferred_init = True)
    
    def forward(self, x):
        '''
        Chebyshev graph convolution operation
    
        Parameters
        ----------
        x: mx.ndarray, graph signal matrix, shape is (batch_size, N, F, T_{r-1}), F is the num of features

        Returns
        ----------
        mx.ndarray, shape is (batch_size, N, self.num_of_filters, T_{r-1})
        
        '''
        batch_size, num_of_features, num_of_vertices, num_of_timesteps = x.shape
        
        self.Theta.shape = (self.K, self.num_of_filters, num_of_features)
        self.Theta._finish_deferred_init()
        
        outputs = []
        for time_step in range(num_of_timesteps):
            graph_signal = x[:, :, :, time_step]
            output = nd.zeros(shape = (self.num_of_filters, num_of_vertices, batch_size), ctx = x.context)
            for k in range(self.K):
                T_k = self.cheb_polys[k]
                theta_k = self.Theta.data()[k]
                rhs = nd.concat(*[nd.dot(graph_signal[idx], T_k).expand_dims(-1) for idx in range(batch_size)], dim = -1)
                output = output + nd.dot(theta_k, rhs)
            outputs.append(output.transpose((2, 0, 1)).expand_dims(-1))
        return nd.relu(nd.concat(*outputs, dim = -1))
    
class temporal_conv_layer(nn.Block):
    def __init__(self, num_of_filters, K_t, **kwargs):
        super(temporal_conv_layer, self).__init__(**kwargs)
        
        if isinstance(num_of_filters, int) and num_of_filters % 2 != 0:
            raise ValueError("num of filters in time convolution must be even integers")
            
        self.num_of_filters = num_of_filters
        with self.name_scope():
            self.conv = nn.Conv2D(channels = num_of_filters, kernel_size = (1, K_t))
            self.residual_conv = nn.Conv2D(channels = num_of_filters // 2, kernel_size = (1, K_t))
        
    def forward(self, x):
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        
        conv_output = self.conv(x)
        
        P = conv_output[:, : self.num_of_filters // 2, :, :]
        Q = conv_output[:, self.num_of_filters // 2: , :, :]
        assert P.shape == Q.shape
        
        return P * nd.sigmoid(Q) + self.residual_conv(x)

class ST_block(nn.Block):
    def __init__(self, backbone, **kwargs):
        super(ST_block, self).__init__(**kwargs)
        
        num_of_time_conv_filters1 = backbone['num_of_time_conv_filters1']
        num_of_time_conv_filters2 = backbone['num_of_time_conv_filters2']
        K_t = backbone['K_t']
        num_of_cheb_filters = backbone['num_of_cheb_filters']
        K = backbone['K']
        cheb_polys = backbone['cheb_polys']
        
        with self.name_scope():
            self.time_conv1 = temporal_conv_layer(num_of_time_conv_filters1, K_t)
            self.cheb_conv = cheb_conv(num_of_cheb_filters, K, cheb_polys)
            self.time_conv2 = temporal_conv_layer(num_of_time_conv_filters2, K_t)
            self.bn = nn.BatchNorm()
            
    def forward(self, x):
        '''
        Parameters
        ----------
        x: mx.ndarray, shape is batch_size, num_of_features, num_of_vertices, num_of_timesteps
        '''
        return self.bn(self.time_conv2(self.cheb_conv(self.time_conv1(x))))

class STGCN(nn.Block):
    def __init__(self, backbones, final_num_of_time_filters, **kwargs):
        super(STGCN, self).__init__(**kwargs)
        
        self.final_num_of_time_filters = final_num_of_time_filters
        
        self.st_blocks = []
        for backbone in backbones:
            self.st_blocks.append(ST_block(backbone))
            self.register_child(self.st_blocks[-1])
        
        with self.name_scope():
            self.final_time_conv_weight = self.params.get("conv_weight", allow_deferred_init = True)
            self.final_time_conv_bias = self.params.get('conv_bias', allow_deferred_init = True)
            self.final_fc_weight = self.params.get('fc_weight', allow_deferred_init = True)
            self.final_fc_bias = self.params.get('fc_bias', allow_deferred_init = True)
        
    def forward(self, x):
        output = x
        for block in self.st_blocks:
            output = block(output)
        
        batch_size, num_of_features, num_of_vertices, num_of_timesteps = output.shape
        
        self.final_time_conv_weight.shape = (num_of_features * num_of_timesteps, self.final_num_of_time_filters)
        self.final_time_conv_weight._finish_deferred_init()
        
        self.final_time_conv_bias.shape = (1, self.final_num_of_time_filters)
        self.final_time_conv_bias._finish_deferred_init()
        
        final_conv_output =  nd.dot(output.transpose((0, 2, 1, 3)).reshape(batch_size, num_of_vertices, -1), 
                                    self.final_time_conv_weight.data()) + self.final_time_conv_bias.data()
        
        batch_size, num_of_vertices, num_of_features = final_conv_output.shape
        
        self.final_fc_weight.shape = (num_of_features, 1)
        self.final_fc_weight._finish_deferred_init()
        self.final_fc_bias.shape = (1, )
        self.final_fc_bias._finish_deferred_init()
        
        return nd.dot(final_conv_output, self.final_fc_weight.data()) + self.final_fc_bias.data()
    
if __name__ == "__main__":
    ctx = mx.cpu()
    distance_df = pd.read_csv('../data/METR-LA/preprocessed/distance.csv', dtype={'from': 'int', 'to': 'int'})
    num_of_vertices = 207
    A = get_adjacency_matrix(distance_df, num_of_vertices, 0.1)
    L_tilde = scaled_Laplacian(A)
    cheb_polys = [nd.array(i, ctx = ctx) for i in cheb_polynomial(L_tilde, 3)]
    backbones = [
    {
        'num_of_time_conv_filters1': 64,
        'num_of_time_conv_filters2': 64,
        'K_t': 3,
        'num_of_cheb_filters': 32,
        'K': 3,
        'cheb_polys': cheb_polys
    },
    {
        'num_of_time_conv_filters1': 64,
        'num_of_time_conv_filters2': 64,
        'K_t': 3,
        'num_of_cheb_filters': 32,
        'K': 3,
        'cheb_polys': cheb_polys
    }]
    net = STGCN(backbones, 64)
    net.initialize(ctx = ctx)
    print(net(nd.random_uniform(shape = (16, 1, 207, 12))).shape)