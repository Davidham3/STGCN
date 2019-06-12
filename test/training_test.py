# -*- coding: utf-8 -*-

import sys
import unittest

import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet import autograd

sys.path.append('../model')


class Test(unittest.TestCase):

    def test_predict(self):
        from model import hybrid_model
        from model import trainer

        ctx = mx.gpu(1)
        num_of_vertices = 228
        batch_size = 8
        cheb_polys = nd.random_uniform(shape=(num_of_vertices,
                                              num_of_vertices * 3),
                                       ctx=ctx)
        blocks = [[1, 32, 64], [64, 32, 128]]
        x = nd.random_uniform(shape=(batch_size, 1, 12, num_of_vertices))

        net = hybrid_model.STGCN(12, 3, 3, blocks, 1.0,
                                 num_of_vertices, cheb_polys)
        net.initialize(ctx=ctx)
        net.hybridize()

        y = trainer.predict_batch(net, ctx, x, 12)

        self.assertEqual(y.shape, (batch_size, 1, 12, num_of_vertices))

    def test_evaluate(self):
        from model import hybrid_model
        from model import trainer
        from data_loader.data_utils import data_gen
        import numpy as np
        from mxboard import SummaryWriter
        import os
        import shutil

        ctx = mx.gpu(1)
        num_of_vertices = 897
        batch_size = 50

        PeMS_dataset = data_gen('datasets/PeMSD7_V_897.csv', 24)
        print('>> Loading dataset with Mean: {0:.2f}, STD: {1:.2f}'.format(
            PeMS_dataset.mean,
            PeMS_dataset.std
        ))

        test = PeMS_dataset['test'].transpose((0, 3, 1, 2))
        test_x, test_y = test[:100, :, : 12, :], test[:100, :, 12:, :]
        test_loader = gluon.data.DataLoader(
            gluon.data.ArrayDataset(nd.array(test_x), nd.array(test_y)),
            batch_size=batch_size,
            shuffle=False
        )
        print(test_x.shape, test_y.shape)

        cheb_polys = nd.random_uniform(shape=(num_of_vertices,
                                              num_of_vertices * 3))
        blocks = [[1, 32, 64], [64, 32, 128]]
        x = nd.random_uniform(shape=(batch_size, 1, 12, num_of_vertices),
                              ctx=ctx)

        net = hybrid_model.STGCN(12, 3, 3, blocks, 1.0,
                                 num_of_vertices, cheb_polys)
        net.initialize(ctx=ctx)
        net.hybridize()
        net(x)

        ground_truth = (np.concatenate([y.asnumpy() for x, y in test_loader],
                                       axis=0) *
                        PeMS_dataset.std +
                        PeMS_dataset.mean)[:100]

        if os.path.exists('test_logs'):
            shutil.rmtree('test_logs')
        sw = SummaryWriter('test_logs', flush_secs=5)

        trainer.evaluate(net, ctx, ground_truth, test_loader,
                         12, PeMS_dataset.mean, PeMS_dataset.std, sw, 0)
        self.assertEqual(os.path.exists('test_logs'), True)
        sw.close()
        if os.path.exists('test_logs'):
            shutil.rmtree('test_logs')

if __name__ == '__main__':
    unittest.main()
