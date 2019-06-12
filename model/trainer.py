# -*- coding:utf-8 -*-

import time
import os

import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet import autograd
from mxboard import SummaryWriter

# from model.tester import model_inference

from . import base_model
from . import hybrid_model

import utils


def model_train(blocks, args, dataset, cheb_polys, ctx, logdir='./logdir'):
    '''
    Parameters
    ----------
    blocks: list[list], model structure, e.g. [[1, 32, 64], [64, 32, 128]]

    args: argparse.Namespace

    dataset: Dataset

    cheb_polys: mx.ndarray,
                shape is (num_of_vertices, order_of_cheb * num_of_vertices)

    ctx: mx.context.Context

    logdir: str, path of mxboard logdir

    '''

    num_of_vertices = args.num_of_vertices
    n_his, n_pred = args.n_his, args.n_pred
    order_of_cheb, Kt = args.order_of_cheb, args.kt
    batch_size, epochs = args.batch_size, args.epochs
    opt = args.opt
    keep_prob = args.keep_prob

    # data
    train = dataset['train'].transpose((0, 3, 1, 2))
    val = dataset['val'].transpose((0, 3, 1, 2))
    test = dataset['test'].transpose((0, 3, 1, 2))

    train_x, train_y = train[:, :, : n_his, :], train[:, :, n_his:, :]
    val_x, val_y = val[:, :, : n_his, :], val[:, :, n_his:, :]
    test_x, test_y = test[:, :, : n_his, :], test[:, :, n_his:, :]

    print(train_x.shape, train_y.shape, val_x.shape,
          val_y.shape, test_x.shape, test_y.shape)

    train_loader = gluon.data.DataLoader(
        gluon.data.ArrayDataset(nd.array(train_x), nd.array(train_y)),
        batch_size=batch_size,
        shuffle=False
    )
    val_loader = gluon.data.DataLoader(
        gluon.data.ArrayDataset(nd.array(val_x), nd.array(val_y)),
        batch_size=batch_size,
        shuffle=False
    )
    test_loader = gluon.data.DataLoader(
        gluon.data.ArrayDataset(nd.array(test_x), nd.array(test_y)),
        batch_size=batch_size,
        shuffle=False
    )

    ground_truth = (np.concatenate([y.asnumpy() for x, y in test_loader],
                                   axis=0) *
                    dataset.std +
                    dataset.mean)

    # model
    model = hybrid_model.STGCN(n_his=n_his,
                               order_of_cheb=order_of_cheb,
                               Kt=Kt,
                               blocks=blocks,
                               keep_prob=keep_prob,
                               num_of_vertices=num_of_vertices,
                               cheb_polys=cheb_polys)
    model.initialize(ctx=ctx, init=mx.init.Xavier())
    model.hybridize()

    # loss function
    loss = gluon.loss.L2Loss()

    # trainer
    trainer = gluon.Trainer(model.collect_params(), args.opt)
    trainer.set_learning_rate(args.lr)

    if not os.path.exists('params'):
        os.mkdir('params')

    sw = SummaryWriter(logdir=logdir, flush_secs=5)
    train_step = 0
    val_step = 0

    for epoch in range(epochs):
        start_time = time.time()
        for x, y in train_loader:
            tmp = nd.concat(x, y, dim=2)
            for pred_idx in range(n_pred):
                end_idx = pred_idx + n_his
                x_ = tmp[:, :, pred_idx: end_idx, :]
                y_ = tmp[:, :, end_idx: end_idx + 1, :]
                with autograd.record():
                    l = loss(model(x_.as_in_context(ctx)),
                             y_.as_in_context(ctx))
                l.backward()
                sw.add_scalar(tag='training_loss',
                              value=l.mean().asscalar(),
                              global_step=train_step)
                trainer.step(x.shape[0])
                train_step += 1

        val_loss_list = []
        for x, y in val_loader:
            pred = predict_batch(model, ctx, x, n_pred)
            val_loss_list.append(loss(pred, y).mean().asscalar())
        sw.add_scalar(tag='val_loss',
                      value=sum(val_loss_list) / len(val_loss_list),
                      global_step=val_step)

        evaluate(model, ctx, ground_truth, test_loader, n_pred,
                 dataset.mean, dataset.std, sw, val_step)
        val_step += 1

        if (epoch + 1) % args.save == 0:
            model.save_parameters('params/{}.params'.format(epoch + 1))

    sw.close()


def predict_batch(model, ctx, x, n_pred):
    '''
    Parameters
    ----------
    x: mx.ndarray, shape is (batch_size, 1, n_his, num_of_vertices)

    Returns
    ----------
    mx.ndarray, shape is (batch_size, 1, n_pred, num_of_vertices)
    '''
    predicts = []
    for pred_idx in range(n_pred):
        x_input = nd.concat(x, *predicts, dim=2)[:, :, - n_pred:, :]
        predicts.append(model(x_input.as_in_context(ctx))
                        .as_in_context(mx.cpu()))
    return nd.concat(*predicts, dim=2)


def predict(model, ctx, data_loader, n_pred):
    '''
    predict n_pred time steps

    Returns
    ----------
    mx.ndarray

    '''

    predictions = []
    for x, _ in data_loader:
        predictions.append(predict_batch(model, ctx, x, n_pred))
    predictions = nd.concat(*predictions, dim=0)
    return predictions


def evaluate(model, ctx, ground_truth, test_loader,
             n_pred, mean, std, sw, step):
    '''
    evaluate model on testing set

    Parameters
    ----------
    ground_truth: np.ndarray,
                  shape is (num_of_samples, 1, n_pred, num_of_vertices)

    test_loader: gluon.data.DataLoader, contains x and y

    n_pred: int

    mean: int

    std: int

    sw: mxboard.SummaryWriter

    step: int

    '''

    predictions = predict(model, ctx, test_loader, n_pred).asnumpy()
    pred = utils.math_utils.z_inverse(predictions, mean, std)

    mape = utils.math_utils.masked_mape_np(ground_truth, pred, 0)
    mae = utils.math_utils.MAE(ground_truth, pred)
    rmse = utils.math_utils.RMSE(ground_truth, pred)

    sw.add_scalar(tag='MAPE', value=mape, global_step=step)
    sw.add_scalar(tag='MAE', value=mae, global_step=step)
    sw.add_scalar(tag='RMSE', value=rmse, global_step=step)

    print('step: {}, MAPE: {}, MAE: {}, RMSE: {}'.format(step, mape,
                                                         mae, rmse))
