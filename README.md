# STGCN
This is an implementation of [Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting](https://arxiv.org/abs/1709.04875v3) which has been accepted by IJCAI 2018.

# Requirements
mxnet >= 1.3.0  
pandas  
numpy

# Usage
```
python main.py
```

You can construct your adjacency matrix and graph signal matrix like the function `data_preprocess` in 'main.py'.

Using
```python
from lib.utils import *

L_tilde = scaled_Laplacian(A.asnumpy())
cheb_polys = [nd.array(i, ctx = ctx) for i in cheb_polynomial(L_tilde, 3)
```
to compute chebyshev polynomials.

Using
```python
from model import STGCN

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
net.initialize(ctx = mx.cpu()) # or mx.gpu(0)
```
to initialize a new model.

# Dataset
dataset comes from [PEMS](http://pems.dot.ca.gov/), we sampled a little from Bay area. You can get sampled data from "data" folder.

The file "distance.csv" contains the distance between two stations, which we linked together.

The file "graph_signal_data_small.txt" contains the time series of each station, it's in a json format, you can use the function "data_preprocessing" in main.py to read it.