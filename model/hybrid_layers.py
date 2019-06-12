# -*- coding:utf-8 -*-

from mxnet import nd
import mxnet as mx
from mxnet.gluon import nn


class Gconv(nn.HybridBlock):
    def __init__(self, order_of_cheb, c_in, c_out, num_of_vertices, **kwargs):
        super(Gconv, self).__init__(**kwargs)
        self.order_of_cheb = order_of_cheb
        self.c_in = c_in
        self.c_out = c_out
        self.num_of_vertices = num_of_vertices
        with self.name_scope():
            self.theta = nn.Dense(c_out, activation=None, flatten=False)

    def hybrid_forward(self, F, x, cheb_polys):
        '''
        Parameters
        ----------
        x: nd.array, shape is (batch_size * time_step, num_of_vertices, c_in)

        cheb_polys: nd.array,
                shape is (num_of_vertices, order_of_cheb * num_of_vertices)

        Returns
        ----------
        shape is (batch_size * time_step, num_of_vertices, c_out)
        '''

        # (batch_size * c_in, num_of_vertices)
        x_tmp = x.transpose((0, 2, 1)).reshape((-1, self.num_of_vertices))

        # (batch_size, c_in, order_of_cheb, num_of_vertices)
        x_mul = F.dot(x_tmp, cheb_polys).reshape((-1,
                                                 self.c_in,
                                                 self.order_of_cheb,
                                                 self.num_of_vertices))

        # batch_size, num_of_vertices, c_in * order_of_cheb
        x_ker = x_mul.transpose((0, 3, 1, 2)) \
                     .reshape((-1, self.num_of_vertices,
                               self.c_in * self.order_of_cheb))

        x_gconv = self.theta(x_ker)
        return x_gconv


class Temporal_conv_layer(nn.HybridBlock):
    def __init__(self, Kt, c_in, c_out, activation='relu', **kwargs):
        super(Temporal_conv_layer, self).__init__(**kwargs)
        self.Kt = Kt
        self.c_in = c_in
        self.c_out = c_out
        self.activation = activation
        with self.name_scope():
            if c_in > c_out:
                self.res_conv = nn.Conv2D(c_out, kernel_size=(1, 1),
                                          activation=None, use_bias=False)
            if activation == 'GLU':
                self.conv = nn.Conv2D(2 * c_out, (Kt, 1), activation=None)
            elif activation == 'relu':
                self.conv = nn.Conv2D(c_out, (Kt, 1), activation=None)
            else:
                self.conv = nn.Conv2D(c_out, (Kt, 1), activation=activation)

    def hybrid_forward(self, F, x):
        '''
        Parameters
        ----------
        x: nd.array, shape is (batch_size, c_in, time_step, num_of_vertices)

        Returns
        ----------
        shape is (batch_size, c_out, time_step - Kt + 1, num_of_vertices)

        '''

        if self.c_in == self.c_out:
            x_input = x
        elif self.c_in > self.c_out:
            x_input = self.res_conv(x)
        else:
            padding = F.broadcast_axis(
                        F.slice(
                            F.zeros_like(x),
                            begin=(None, None, None, None),
                            end=(None, 1, None, None)
                        ), axis=1, size=self.c_out - self.c_in)
            x_input = F.concat(x, padding, dim=1)

        x_input = F.slice(x_input,
                          begin=(None, None, self.Kt - 1, None),
                          end=(None, None, None, None))

        x_conv = self.conv(x)
        if self.activation == 'GLU':
            x_conv = self.conv(x)
            x_conv1 = F.slice(x_conv,
                              begin=(None, None, None, None),
                              end=(None, self.c_out, None, None))
            x_conv2 = F.slice(x_conv,
                              begin=(None, self.c_out, None, None),
                              end=(None, None, None, None))
            return (x_conv1 + x_input) * F.sigmoid(x_conv2)
        if self.activation == 'relu':
            return F.relu(x_conv + x_input)
        return x_conv


class Spatio_conv_layer(nn.HybridBlock):
    def __init__(self, order_of_cheb, c_in, c_out, num_of_vertices, T,
                 cheb_polys, **kwargs):
        super(Spatio_conv_layer, self).__init__(**kwargs)
        self.order_of_cheb = order_of_cheb
        self.c_in = c_in
        self.c_out = c_out
        self.num_of_vertices = num_of_vertices
        self.T = T
        self.cheb_polys = self.params.get_constant('cheb_polys',
                                                   value=cheb_polys)
        with self.name_scope():
            if c_in > c_out:
                self.res_conv = nn.Conv2D(c_out, kernel_size=(1, 1),
                                          activation=None, use_bias=False)
            self.gconv = Gconv(order_of_cheb, c_in, c_out, num_of_vertices)

    def hybrid_forward(self, F, x, cheb_polys):
        '''
        Parameters
        ----------
        x: nd.array, shape is (batch_size, c_in, time_step, num_of_vertices)

        cheb_polys: nd.array,
                shape is (num_of_vertices, order_of_cheb * num_of_vertices)

        Returns
        ----------
        shape is (batch_size, c_out, time_step, num_of_vertices)
        '''

        if self.c_in == self.c_out:
            x_input = x
        elif self.c_in > self.c_out:
            x_input = self.res_conv(x)
        else:
            padding = F.broadcast_axis(F.zeros_like(x), axis=1,
                                       size=self.c_out - self.c_in)
            x_input = F.concat(x, padding, dim=1)

        x_tmp = x.transpose((0, 2, 3, 1)) \
                 .reshape((-1, self.num_of_vertices, self.c_in))

        x_gconv = self.gconv(x_tmp, cheb_polys)

        x_gc = x_gconv.reshape((-1, self.T, self.num_of_vertices, self.c_out))\
                      .transpose((0, 3, 1, 2))

        x_gc = F.slice(x_gc,
                       begin=(None, None, None, None),
                       end=(None, self.c_out, None, None))
        return F.relu(x_gc + x_input)


class St_conv_block(nn.HybridBlock):
    def __init__(self, order_of_cheb, Kt, channels, num_of_vertices, keep_prob,
                 T, cheb_polys, activation='GLU', **kwargs):
        super(St_conv_block, self).__init__(**kwargs)
        c_si, c_t, c_oo = channels
        self.order_of_cheb = order_of_cheb
        self.Kt = Kt
        self.keep_prob = keep_prob
        self.seq = nn.HybridSequential()
        self.seq.add(
            Temporal_conv_layer(Kt, c_si, c_t, activation),
            Spatio_conv_layer(order_of_cheb, c_t, c_t,
                              num_of_vertices, T - (Kt - 1), cheb_polys),
            Temporal_conv_layer(Kt, c_t, c_oo),
            nn.LayerNorm(axis=1),
            nn.Dropout(1 - keep_prob)
        )

    def hybrid_forward(self, F, x):
        '''
        Parameters
        ----------
        x: nd.array,
           shape is (batch_size, channels[0], time_step, num_of_vertices)

        Returns
        ----------
        shape is (batch_size, channels[-1],
                  time_step - 2(Kt - 1), num_of_vertices)
        '''
        return self.seq(x)


class Output_layer(nn.HybridBlock):
    def __init__(self, c_in, T, num_of_vertices, activation='GLU', **kwargs):
        super(Output_layer, self).__init__(**kwargs)
        self.c_in = c_in
        self.layer = nn.HybridSequential()
        self.layer.add(
            Temporal_conv_layer(T, c_in, c_in, activation),
            nn.LayerNorm(axis=1),
            Temporal_conv_layer(1, c_in, c_in, 'sigmoid'),
            nn.Conv2D(1, (1, 1), activation=None)
        )

    def hybrid_forward(self, F, x):
        '''
        Parameters
        ----------
        x: nd.array, shape is (batch_size, c_in, time_step, num_of_vertices)

        Returns
        ----------
        shape is (batch_size, 1, 1, num_of_vertices)
        '''
        return self.layer(x)
