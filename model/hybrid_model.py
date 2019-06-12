# -*- coding:utf-8 -*-

from .hybrid_layers import *


class STGCN(nn.HybridBlock):
    def __init__(self, n_his, order_of_cheb, Kt, blocks, keep_prob,
                 num_of_vertices, cheb_polys, **kwargs):
        super(STGCN, self).__init__(**kwargs)
        Ko = n_his
        self.model = nn.HybridSequential()
        for idx, channels in enumerate(blocks):
            self.model.add(
                St_conv_block(order_of_cheb=order_of_cheb,
                              Kt=Kt,
                              channels=channels,
                              num_of_vertices=num_of_vertices,
                              keep_prob=keep_prob,
                              T=n_his - 2 * idx * (Kt - 1),
                              cheb_polys=cheb_polys,
                              activation='GLU')
            )
            Ko -= 2 * (Kt - 1)

        if Ko > 1:
            self.model.add(
                Output_layer(c_in=blocks[-1][-1],
                             T=Ko,
                             num_of_vertices=num_of_vertices,
                             activation='GLU')
            )

    def hybrid_forward(self, F, x):
        return self.model(x)
