# -*- coding:utf-8 -*-

import os
import argparse

import mxnet as mx
from mxnet import nd

from utils import math_graph
from data_loader import data_utils
from model.trainer import model_train

ctx = mx.gpu(0)

parser = argparse.ArgumentParser()
parser.add_argument('--num_of_vertices', type=int, default=228)
parser.add_argument('--n_his', type=int, default=12)
parser.add_argument('--n_pred', type=int, default=12)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--save', type=int, default=10)
parser.add_argument('--order_of_cheb', type=int, default=3)
parser.add_argument('--kt', type=int, default=3)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--opt', type=str, default='adam')
parser.add_argument('--keep_prob', type=float, default=1.0)
parser.add_argument('--adj_path', type=str,
                    default='datasets/PeMSD7_W_228.csv')
parser.add_argument('--time_series_path', type=str,
                    default='datasets/PeMSD7_V_228.csv')

args = parser.parse_args()
print('Training configs: {}'.format(args))

n_his, n_pred = args.n_his, args.n_pred
order_of_cheb = args.order_of_cheb

# blocks: settings of channel size in st_conv_blocks / bottleneck design
blocks = [[1, 32, 64], [64, 32, 128]]


adj = math_graph.weight_matrix(args.adj_path)
L = math_graph.scaled_laplacian(adj)
cheb_polys = nd.array(math_graph.cheb_poly_approx(L, order_of_cheb))

# Data Preprocessing
PeMS_dataset = data_utils.data_gen(args.time_series_path, n_his + n_pred)
print('>> Loading dataset with Mean: {0:.2f}, STD: {1:.2f}'.format(
    PeMS_dataset.mean,
    PeMS_dataset.std
))

if __name__ == '__main__':
    import shutil
    logdir = './logdir'
    if os.path.exists(logdir):
        shutil.rmtree(logdir)
    model_train(blocks, args, PeMS_dataset, cheb_polys, ctx, logdir=logdir)
