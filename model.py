# -*- coding:utf-8 -*-

import pandas as pd

import mxnet as mx
from mxnet.gluon import nn
from mxnet import nd
from mxnet import autograd
from mxnet import gluon

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
        
        K: int, up K-order chebyshev polynomials will be used in this convolution
        
        '''
        super(cheb_conv, self).__init__(**kwargs)
        self.K = K
        self.num_of_filters = num_of_filters
        self.cheb_polys = cheb_polys
        
        # shape of theta is (self.K, num_of_features, num_of_filters)
        with self.name_scope():
            self.Theta = self.params.get('Theta', allow_deferred_init=True)
    
    def forward(self, x):
        '''
        Chebyshev graph convolution operation
    
        Parameters
        ----------
        x: mx.ndarray, graph signal matrix, shape is (batch_size, num_of_features, num_of_vertices, num_of_timesteps)

        Returns
        ----------
        mx.ndarray, shape is (batch_size, self.num_of_filters, num_of_vertices, num_of_timesteps)
        
        '''
        batch_size, num_of_features, num_of_vertices, num_of_timesteps = x.shape
        
        self.Theta.shape = (self.K, self.num_of_filters, num_of_features)
        self.Theta._finish_deferred_init()
        
        outputs = []
        
        # ChebNet GCN will run for each time step
        for time_step in range(num_of_timesteps):
            
            # shape is (batch_size, num_of_features, num_of_vertices)
            graph_signal = x[:, :, :, time_step]
            
            output = nd.zeros(shape=(self.num_of_filters, num_of_vertices, batch_size), ctx=x.context)
            
            for k in range(self.K):
                
                # shape of T_k is (num_of_vertices, num_of_vertices)
                T_k = self.cheb_polys[k]
                
                # shape of theta_k is (num_of_filters, num_of_features)
                theta_k = self.Theta.data()[k]
                
                # shape of rhs is (num_of_features, num_of_vertices, batch_size)
                rhs = nd.concat(*[nd.dot(graph_signal[idx], T_k).expand_dims(-1) for idx in range(batch_size)], dim=-1)
                
                output = output + nd.dot(theta_k, rhs)
            
            # add ChebNet output to list outputs
            outputs.append(output.transpose((2, 0, 1)).expand_dims(-1))
        
        # concatenate all GCN output and activate them
        return nd.relu(nd.concat(*outputs, dim=-1))
    
class temporal_conv_layer(nn.Block):
    '''
    temporal convolution with GLU
    '''
    def __init__(self, num_of_filters, K_t, **kwargs):
        '''
        Parameters
        ----------
        num_of_filters: int, number of temporal convolutional filters
        
        K_t: int, length of filters
        
        '''
        super(temporal_conv_layer, self).__init__(**kwargs)
        
        if isinstance(num_of_filters, int) and num_of_filters % 2 != 0:
            raise ValueError("num of filters in time convolution must be even integers")
            
        self.num_of_filters = num_of_filters
        with self.name_scope():
            self.conv = nn.Conv2D(channels=num_of_filters, kernel_size=(1, K_t))
            self.residual_conv = nn.Conv2D(channels=num_of_filters // 2, kernel_size=(1, K_t))
        
    def forward(self, x):
        '''
        Parameters
        ----------
        x: mx.ndarray, shape is (batch_size, num_of_features, num_of_vertices, num_of_timesteps)
        
        
        Returns
        ----------
        mx.ndarray, shape is (batch_size, num_of_filters/2, num_of_vertices, num_of_timesteps - K_t)
        
        '''
        
        # shape is (batch_size, num_of_filters, num_of_vertices, num_of_timesteps - K_t)
        conv_output = self.conv(x)
        
        P = conv_output[:, : self.num_of_filters // 2, :, :]
        Q = conv_output[:, self.num_of_filters // 2: , :, :]
        assert P.shape == Q.shape
        
        return (P + self.residual_conv(x)) * nd.sigmoid(Q)

class ST_block(nn.Block):
    def __init__(self, backbone, **kwargs):
        super(ST_block, self).__init__(**kwargs)
        
        # number of first temporal convolution's filters
        num_of_time_conv_filters1 = backbone['num_of_time_conv_filters1']
        
        # number of second temporal convolution's filters
        num_of_time_conv_filters2 = backbone['num_of_time_conv_filters2']
        
        # length of temporal convolutional filter
        K_t = backbone['K_t']
        
        # number of spatial convolution's filters
        num_of_cheb_filters = backbone['num_of_cheb_filters']
        
        # number of the order of chebNet
        K = backbone['K']
        
        # list of chebyshev polynomials from first-order to K-order
        cheb_polys = backbone['cheb_polys']
        
        with self.name_scope():
            self.time_conv1 = temporal_conv_layer(num_of_time_conv_filters1, K_t)
            self.cheb_conv = cheb_conv(num_of_cheb_filters, K, cheb_polys)
            self.time_conv2 = temporal_conv_layer(num_of_time_conv_filters2, K_t)
            self.ln = nn.LayerNorm(axis=1)
            
    def forward(self, x):
        '''
        Parameters
        ----------
        x: mx.ndarray, shape is (batch_size, num_of_features, num_of_vertices, num_of_timesteps)
        
        Returns
        ----------
        mx.ndarray, shape is 
            (batch_size, num_of_time_conv_filters2 / 2, num_of_vertices, num_of_timesteps - 2(K_t - 1) )
        
        '''
        return self.ln(self.time_conv2(self.cheb_conv(self.time_conv1(x))))

class STGCN(nn.Block):
    def __init__(self, backbones, num_of_last_time_conv_filters, **kwargs):
        super(STGCN, self).__init__(**kwargs)
        
        # two ST blocks
        self.st_blocks = []
        for backbone in backbones:
            self.st_blocks.append(ST_block(backbone))
            self.register_child(self.st_blocks[-1])
        
        # extra three convolutional structure to map output into label space
        with self.name_scope():
            self.last_time_conv = temporal_conv_layer(num_of_last_time_conv_filters, 4)
            self.final_conv = nn.Conv2D(channels=128, kernel_size=(1, 1), activation='sigmoid')
            self.conv_output = nn.Conv2D(channels=1, kernel_size = (1, 1))
        
    def forward(self, x):
        '''
        Parameters
        ----------
        x: mx.ndarray, shape is (batch_size, num_of_features, num_of_vertices, num_of_timesteps)
        

        Returns
        ----------
        mx.ndarray, shape is (batch_size, num_of_vertices, num_points_for_prediction)
        '''
        output = x
        for block in self.st_blocks:
            output = block(output)
        return self.conv_output(self.final_conv(self.last_time_conv(output)))[:, 0, :, :]
    
if __name__ == "__main__":
    ctx = mx.cpu()
    distance_df = pd.read_csv('data/test_data1/distance.csv', dtype={'from': 'int', 'to': 'int'})
    num_of_vertices = 307
    A = get_adjacency_matrix(distance_df, num_of_vertices, 0.1)
    L_tilde = scaled_Laplacian(A)
    cheb_polys = [nd.array(i, ctx = ctx) for i in cheb_polynomial(L_tilde, 3)]
    backbones = [
        {
            'num_of_time_conv_filters1': 32,
            'num_of_time_conv_filters2': 64,
            'K_t': 3,
            'num_of_cheb_filters': 32,
            'K': 1,
            'cheb_polys': cheb_polys
        },
        {
            'num_of_time_conv_filters1': 32,
            'num_of_time_conv_filters2': 128,
            'K_t': 3,
            'num_of_cheb_filters': 32,
            'K': 1,
            'cheb_polys': cheb_polys
        }
    ]
    net = STGCN(backbones, 128)
    net.initialize(ctx=ctx)
    print(net(nd.random_uniform(shape=(16, 1, num_of_vertices, 12))).shape)
