# -*- coding:utf-8 -*-

from mxnet import nd
import mxnet as mx
from mxnet.gluon import nn


class Gconv(nn.Block):
    def __init__(self, order_of_cheb, c_out, **kwargs):
        super(Gconv, self).__init__(**kwargs)
        self.order_of_cheb = order_of_cheb
        with self.name_scope():
            self.theta = nn.Dense(c_out, activation=None,
                                  flatten=False, use_bias=False)

    def forward(self, x, cheb_polys):
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

        _, num_of_vertices, c_in = x.shape

        # (batch_size * c_in, num_of_vertices)
        x_tmp = x.transpose((0, 2, 1)).reshape((-1, num_of_vertices))

        # (batch_size, c_in, order_of_cheb, num_of_vertices)
        x_mul = nd.dot(x_tmp, cheb_polys).reshape((-1,
                                                   c_in,
                                                   self.order_of_cheb,
                                                   num_of_vertices))

        # (batch_size, num_of_vertices, c_in * order_of_cheb)
        x_ker = x_mul.transpose((0, 3, 1, 2)) \
                     .reshape((-1, num_of_vertices, c_in * self.order_of_cheb))

        return self.theta(x_ker)


class Temporal_conv_layer(nn.Block):
    def __init__(self, Kt, c_in, c_out, activation='relu', **kwargs):
        super(Temporal_conv_layer, self).__init__(**kwargs)
        self.Kt = Kt
        self.c_out = c_out
        self.activation = activation
        with self.name_scope():
            self.align = Align_layer(c_in, c_out, None)
            if activation == 'GLU':
                self.conv = nn.Conv2D(2 * c_out, (Kt, 1), activation=None)
            elif activation == 'relu':
                self.conv = nn.Conv2D(c_out, (Kt, 1), activation=None)
            else:
                self.conv = nn.Conv2D(c_out, (Kt, 1), activation=activation)

    def forward(self, x):
        '''
        Parameters
        ----------
        x: nd.array, shape is (batch_size, c_in, time_step, num_of_vertices)

        Returns
        ----------
        shape is (batch_size, c_out, time_step - Kt + 1, num_of_vertices)

        '''

        x_input = self.align(x)[:, :, self.Kt - 1:, :]

        x_conv = self.conv(x)
        if self.activation == 'GLU':
            x_conv = self.conv(x)
            x_conv1, x_conv2 = nd.split(x_conv, axis=1, num_outputs=2)
            return (x_conv1 + x_input) * nd.sigmoid(x_conv2)
        if self.activation == 'relu':
            return nd.relu(x_conv + x_input)
        return x_conv


class decrease_layer(nn.Block):
    def __init__(self, c_out, activation=None, **kwargs):
        super(decrease_layer, self).__init__(**kwargs)
        self.c_out = c_out
        with self.name_scope():
            self.layer = nn.Conv2D(c_out, (1, 1),
                                   activation=activation, use_bias=False)

    def forward(self, x):
        return self.layer(x)


class increase_layer(nn.Block):
    def __init__(self, c_out, activation=None, **kwargs):
        super(increase_layer, self).__init__(**kwargs)
        self.c_out = c_out

    def forward(self, x):
        batch_size, c, T, num_of_vertices = x.shape
        zeros = nd.zeros(shape=(batch_size, self.c_out - c,
                                T, num_of_vertices),
                         ctx=x.context)
        return nd.concat(x, zeros, dim=1)


class Align_layer(nn.Block):
    def __init__(self, c_in, c_out, activation=None, **kwargs):
        super(Align_layer, self).__init__(**kwargs)
        self.c_in, self.c_out = c_in, c_out
        if self.c_in < self.c_out:
            self.layer = increase_layer(c_out, None)
        elif self.c_in > self.c_out:
            self.layer = decrease_layer(c_out, None)

    def forward(self, x):
        if self.c_in == self.c_out:
            return x
        return self.layer(x)


class Spatio_conv_layer(nn.Block):
    def __init__(self, order_of_cheb, c_in, c_out,
                 cheb_polys, **kwargs):
        super(Spatio_conv_layer, self).__init__(**kwargs)
        self.c_out = c_out
        self.cheb_polys = self.params.get_constant('cheb_polys',
                                                   value=cheb_polys)
        with self.name_scope():
            self.align = Align_layer(c_in, c_out, None)
            self.gconv = Gconv(order_of_cheb, c_out)

    def forward(self, x):
        '''
        Parameters
        ----------
        x: nd.array, shape is (batch_size, c_in, time_step, num_of_vertices)

        Returns
        ----------
        shape is (batch_size, c_out, time_step, num_of_vertices)
        '''
        batch_size, c_in, T, num_of_vertices = x.shape
        x_input = self.align(x)

        x_tmp = x.transpose((0, 2, 3, 1)) \
                 .reshape((-1, num_of_vertices, c_in))

        x_gconv = self.gconv(x_tmp, self.cheb_polys.data())

        x_gc = x_gconv.reshape((-1, T, num_of_vertices, self.c_out)) \
                      .transpose((0, 3, 1, 2))

        x_gc = x_gc[:, : self.c_out, :, :]
        return nd.relu(x_gc + x_input)


class St_conv_block(nn.Block):
    def __init__(self, order_of_cheb, Kt, channels, keep_prob,
                 cheb_polys, activation='GLU', **kwargs):
        super(St_conv_block, self).__init__(**kwargs)
        c_si, c_t, c_oo = channels
        self.order_of_cheb = order_of_cheb
        self.Kt = Kt
        self.keep_prob = keep_prob
        self.seq = nn.Sequential()
        self.seq.add(
            Temporal_conv_layer(Kt, c_si, c_t, activation),
            Spatio_conv_layer(order_of_cheb, c_t, c_t, cheb_polys),
            Temporal_conv_layer(Kt, c_t, c_oo),
            nn.LayerNorm(axis=1),
            nn.Dropout(1 - keep_prob)
        )

    def forward(self, x):
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


class Output_layer(nn.Block):
    def __init__(self, c_in, T, activation='GLU', **kwargs):
        super(Output_layer, self).__init__(**kwargs)
        self.c_in = c_in
        with self.name_scope():
            self.layer = nn.Sequential()
            self.layer.add(
                Temporal_conv_layer(T, c_in, c_in, activation),
                nn.LayerNorm(axis=1),
                Temporal_conv_layer(1, c_in, c_in, 'sigmoid'),
                nn.Conv2D(1, (1, 1), activation=None)
            )

    def forward(self, x):
        '''
        Parameters
        ----------
        x: nd.array, shape is (batch_size, c_in, time_step, num_of_vertices)

        Returns
        ----------
        shape is (batch_size, 1, 1, num_of_vertices)
        '''
        return self.layer(x)
