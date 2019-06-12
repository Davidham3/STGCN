# -*- coding: utf-8 -*-

import sys
import unittest

import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet import autograd

sys.path.append('../model')


class Test(unittest.TestCase):

    def test_stblock(self):

        from model.base_layers import St_conv_block, Output_layer

        num_of_vertices = 228
        cheb_polys = nd.random_uniform(shape=(num_of_vertices,
                                              num_of_vertices * 3))
        net = gluon.nn.Sequential()
        net.add(
            St_conv_block(3, 3, [1, 32, 64], 1.0, cheb_polys),
            St_conv_block(3, 3, [64, 32, 128], 1.0, cheb_polys),
            Output_layer(128, 4)
        )
        net.initialize()

        x = nd.random_uniform(shape=(8, 1, 12, num_of_vertices))
        o = net(x)
        y = nd.random_uniform(shape=o.shape)

        trainer = gluon.Trainer(net.collect_params(), 'adam')
        trainer.set_learning_rate(1e-3)
        loss = gluon.loss.L2Loss()
        with autograd.record():
            l = loss(net(x), y)
        l.backward()
        trainer.step(8)
        self.assertEqual((8, 1, 1, num_of_vertices), o.shape)
        self.assertIsInstance(l.mean().asscalar().item(), float)

    def test_stgcn(self):

        from model import base_model

        ctx = mx.gpu(1)
        num_of_vertices = 228
        batch_size = 8
        cheb_polys = nd.random_uniform(shape=(num_of_vertices,
                                              num_of_vertices * 3),
                                       ctx=ctx)
        blocks = [[1, 32, 64], [64, 32, 128]]
        x = nd.random_uniform(shape=(batch_size, 1, 12, num_of_vertices),
                              ctx=ctx)
        y = nd.random_uniform(shape=(batch_size, 1, 1, num_of_vertices),
                              ctx=ctx)

        net = base_model.STGCN(12, 3, 3, blocks, 1.0, cheb_polys)
        net.initialize(ctx=ctx)
        self.assertEqual((batch_size, 1, 1, num_of_vertices), net(x).shape)

        trainer = gluon.Trainer(net.collect_params(), 'adam')
        trainer.set_learning_rate(1e-3)
        loss = gluon.loss.L2Loss()

        for i in range(5):
            with autograd.record():
                l = loss(net(x), y)
            l.backward()
            trainer.step(batch_size)
            self.assertIsInstance(l.mean().asscalar().item(), float)
            print(l.mean().asscalar())

if __name__ == '__main__':
    unittest.main()
