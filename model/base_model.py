# -*- coding:utf-8 -*-

from .base_layers import *


class STGCN(nn.Block):
    def __init__(self, n_his, order_of_cheb, Kt, blocks, keep_prob,
                 cheb_polys, **kwargs):
        super(STGCN, self).__init__(**kwargs)
        self.model = nn.Sequential()
        for idx, channels in enumerate(blocks):
            self.model.add(St_conv_block(order_of_cheb,
                                         Kt,
                                         channels,
                                         keep_prob,
                                         cheb_polys))
            n_his -= 2 * (Kt - 1)

        if n_his > 1:
            self.model.add(Output_layer(blocks[-1][-1], n_his))

    def forward(self, x):
        '''
        Parameters
        ----------
        x: mx.ndarray, shape is (batch_size, 1, n_his, num_of_vertices)

        '''
        return self.model(x)
