# STGCN

This is an mxnet version implementation of Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting which has been accepted by IJCAI 2018.

# Dataset
dataset comes from [STGCN_IJCAI-18](https://github.com/VeritasYin/STGCN_IJCAI-18).

# Requirements

mxnet >= 1.4.1 and scipy

or

Docker with [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

# Usage

Unzip datasets before you run the code.
```
cd datasets
tar -zxvf PeMSD7.tar.gz
```

```
python main.py
```

or use docker:

```bash
# build
cd docker
docker build -t stgcn/mxnet:1.4.1_gpu_cu100_mkl_py35 .

# run
cd ..
docker run -d -it --rm --runtime=nvidia -v $PWD:/mxnet --name stgcn stgcn/mxnet:1.4.1_gpu_cu100_mkl_py35 python3 main.py
docker logs stgcn
```
