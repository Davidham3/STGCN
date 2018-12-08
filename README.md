# STGCN
This is an implementation of [Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting](https://arxiv.org/abs/1709.04875v3) which has been accepted by IJCAI 2018.

# requirements
mxnet >= 1.3.0
pandas
numpy

# Dataset
dataset comes from [PEMS](http://pems.dot.ca.gov/), we sampled a little from Bay area. You can get sampled data from "data" folder.

The file "distance.csv" contains the distance between two stations, which we linked together.

The file "graph_signal_data_small.txt" contains the time series of each station, it's in a json format, you can use the function "data_preprocessing" in main.py to read it.