# Video-Prediction using seq2seq ConvLSTM in Pytorch

## ConvLSTM       [Unsupervised Learning of Video Representations using LSTMs](https://arxiv.org/abs/1502.04681)
ConvLSTM applies convolutional layers at the beginning of LSTMs, which implements convolution to input X and hidden state H.
![CLSTM_dynamics](https://user-images.githubusercontent.com/7113894/59357391-15c73e00-8d2b-11e9-8234-9d51a90be5dc.png)

The ConvLSTM cell is implemented in [this repo](https://github.com/ndrplz/ConvLSTM_pytorch)
and the seq2seq ConvLSTM is implemented in [this repo](https://github.com/holmdk/Video-Prediction-using-PyTorch)

## Getting Started
Make sure you have numpy, torch, torchvision installed.
```bash
git clone https://github.com/Yuanceli/video-prediction.git
cd ./video-prediction
python train.py
```

## MNIST
Thanks to this guide [Video Prediction using ConvLSTM Autoencoder](https://holmdk.github.io/2020/04/02/video_prediction.html)
### Dataset: 
1. [Unsupervised Learning of Video Representations using LSTMs](http://www.cs.toronto.edu/~nitish/unsupervised_video/) contains 10,000 sequences each of length 20 showing 2 digits moving in a 64 x 64 frame.
2. or you can generate the frames using [this repo](https://github.com/tychovdo/MovingMNIST)

## PONG
IN PROGRESS
