# -*- coding:utf-8 -*-

import json
import time
import os
import shutil
import configparser
import argparse

import numpy as np
from sklearn.preprocessing import StandardScaler

import mxnet as mx
from mxnet import nd
from mxnet import autograd
from mxnet import gluon
from mxnet.gluon import Trainer
from mxboard import SummaryWriter

from lib import utils
from model import STGCN

##########
# configuration part

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    help="configuration file path",
    required=True
)
parser.add_argument(
    "--logdir",
    type=str,
    help="tensorboard logdir",
    required=True
)
parser.add_argument(
    "--rm",
    type=bool,
    help="remove log dir if exists",
    required=False
)
args = parser.parse_args()

config = configparser.ConfigParser()
print('read configuration file: {}'.format(args.config))
config.read(args.config)

data_configs = config['Data']

adj_filename = data_configs['adj_filename']
graph_signal_filename = data_configs['graph_signal_filename']
num_of_vertices = int(data_configs['num_of_vertices'])
params_dir = data_configs['params_dir']

model_configs = config['Model']

orders_of_cheb = int(model_configs['orders_of_cheb'])

ctx = model_configs['ctx']
if ctx.startswith('cpu'):
    ctx = mx.cpu()
elif ctx.startswith('gpu'):
    ctx = mx.gpu(int(ctx[ctx.find('-') + 1:]))
else:
    raise ValueError("context seems to be wrong!")

optimizer = model_configs['optimizer']
learning_rate = float(model_configs['learning_rate'])

batch_size = model_configs['batch_size']
epochs = int(model_configs['epochs'])

decay_interval = model_configs['decay_interval']
if decay_interval == 'None':
    decay_interval = None
decay_rate = model_configs['decay_rate']
if decay_rate == 'None':
    decay_rate = None

##########

A, indices_dict = utils.get_adj_matrix(adj_filename, num_of_vertices)
X = utils.data_preprocess(indices_dict, graph_signal_filename)

L_tilde = utils.scaled_Laplacian(A)
cheb_polys = [nd.array(i, ctx=ctx)
              for i in utils.cheb_polynomial(L_tilde, orders_of_cheb)]

# training: validation: testing = 6: 2: 2
split_line1 = int(X.shape[2] * 0.6)
split_line2 = int(X.shape[2] * 0.8)

train_original_data = X[:, :, : split_line1]
val_original_data = X[:, :, split_line1: split_line2]
test_original_data = X[:, :, split_line2:]

training_data, training_target = utils.build_dataset(
    train_original_data, 12, 1)

val_data, val_target = utils.build_dataset(
    val_original_data, 12, 1)

testing_data, testing_target = utils.build_dataset(
    test_original_data, 12, 12)

# Z-score preprocessing
assert num_of_vertices == training_data.shape[2]
_, num_of_features, _, _ = training_data.shape

transformer = StandardScaler()
training_data_norm = transformer.fit_transform(
    training_data.reshape(training_data.shape[0], -1))\
    .reshape(training_data.shape[0], num_of_features, num_of_vertices, 12)
val_data_norm = transformer.transform(
    val_data.reshape(val_data.shape[0], -1))\
    .reshape(val_data.shape[0], num_of_features, num_of_vertices, 12)
testing_data_norm = transformer.transform(
    testing_data.reshape(testing_data.shape[0], -1))\
    .reshape(testing_data.shape[0], num_of_features, num_of_vertices, 12)

training_data_norm = nd.array(training_data_norm, ctx=ctx)
val_data_norm = nd.array(val_data_norm, ctx=ctx)
testing_data_norm = nd.array(testing_data_norm, ctx=ctx)

training_target = nd.array(training_target, ctx=ctx)
val_target = nd.array(val_target, ctx=ctx)
testing_target = nd.array(testing_target, ctx=ctx)

print('training data shape:',
      training_data_norm.shape, training_target.shape)
print('validation data shape:', val_data_norm.shape, val_target.shape)
print('testing data shape:', testing_data_norm.shape, testing_target.shape)

# model initialization
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

loss_function = gluon.loss.L2Loss()

trainer = Trainer(net.collect_params(), optimizer)
trainer.set_learning_rate(learning_rate)
training_dataloader = gluon.data.DataLoader(
    gluon.data.ArrayDataset(training_data_norm, training_target),
    batch_size=batch_size, shuffle=False)
validation_dataloader = gluon.data.DataLoader(
    gluon.data.ArrayDataset(val_data_norm, val_target),
    batch_size=batch_size, shuffle=False)
testing_dataloader = gluon.data.DataLoader(
    gluon.data.ArrayDataset(testing_data_norm, testing_target),
    batch_size=batch_size, shuffle=False)

if os.path.exists(args.logdir):
    if args.rm:
        shutil.rmtree(args.logdir)
    else:
        raise ValueError("log dir exists!")

if os.path.exists(params_dir):
    if args.rm:
        shutil.rmtree(params_dir)
    else:
        raise ValueError("params dir exists!")

if not os.path.exists(params_dir):
    os.mkdir(params_dir)

sw = SummaryWriter(args.logdir, flush_secs=5)

utils.train_model(net, sw, 0, 0,
                  training_dataloader, validation_dataloader,
                  testing_dataloader, epochs, loss_function,
                  trainer, decay_interval, decay_rate)
