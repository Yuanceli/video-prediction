# -*- coding: utf-8 -*-
"""
@author: Yuance Li
"""
import os
import argparse
import logging
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from MovingMNIST import MovingMNIST
from ConvLSTM import EncoderDecoderConvLSTM
from utils import create_array, generate_video


def training(dataloader, num_epochs):
    # initialize
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    model = EncoderDecoderConvLSTM(nf=args.n_hidden_dim, in_chan=1).to(device)
    path = os.path.join(args.store_dir, 'model.pth')
    criterion=nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr= args.lr)
    # train
    training_loss = []
    logger.info(f'Started training on {device}')
    for epoch in range(num_epochs):
        running_loss = 0
        start = time.time()
        for batch in dataloader:        # (b, t, c, h, w)
            batch = batch.to(device)
            x, y = batch[:, 0:10, :, :, :], batch[:, 10:, :, :, :].squeeze()
            # optimize step
            optimizer.zero_grad()
            y_hat = model(x, future_seq=10).squeeze()
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            # loss
            running_loss  += loss.item()
        epoch_loss = running_loss / len(dataloader)
        training_loss.append(epoch_loss)
        end = time.time() - start
        if epoch % 30 == 0:
            torch.save(model.state_dict(), path)
        logger.info(f'epoch: {epoch} \t loss: {epoch_loss} \t time: {end//60}min')
    logger.info('Finished training')
    torch.save(model.state_dict(), path)
    return training_loss

def testing(dataloader):
    # load
    model = EncoderDecoderConvLSTM(nf=args.n_hidden_dim, in_chan=1)
    path = os.path.join(args.store_dir, 'model.pth')
    model.load_state_dict(torch.load(path))
    # test
    criterion=nn.MSELoss()
    for batch in dataloader:
        x, y = batch[:, 0:10, :, :, :], batch[:, 10:, :, :, :].squeeze()
        y_hat = model(x, future_seq=10).squeeze()
        testing_loss = criterion(y_hat, y)
        video_frames = create_array(y_hat, y)
        generate_video(video_array=video_frames)
        break        # only evaluate one batch
    return testing_loss

def train_dataloader(batch_size):
    train_data = MovingMNIST(
        train=True,         # 60000
        data_root=os.getcwd() + '/data',
        seq_len=20,
        image_size=64,
        deterministic=True,
        num_digits=2)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=False)
    return train_loader

def test_dataloader(batch_size):
    test_data = MovingMNIST(
        train=False,        # 10000
        data_root=os.getcwd() + '/data',
        seq_len=20,
        image_size=64,
        deterministic=True,
        num_digits=2)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False)
    return test_loader

def get_logger(store_dir):
    log_path = os.path.join(store_dir, 'pred.log')
    logger = logging.Logger('train_status', level=logging.DEBUG)
    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(logging.Formatter('%(levelname)s\t%(message)s'))
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s\t%(levelname)s\t%(message)s'))
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)
    return logger

if __name__ == '__main__':
    # hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train for')
    parser.add_argument('--n_hidden_dim', type=int, default=64, help='number of hidden dim for ConvLSTM layers')
    parser.add_argument('--store-dir', dest='store_dir',
                             default=os.path.join('experiments', time.strftime("%Y-%m-%d %H.%M.%S")),
                             help='Path to directory to store experiment results (default: ./experiments/<timestamp>/')
    args = parser.parse_args()
    if not os.path.exists(args.store_dir):
        os.makedirs(args.store_dir)
        
    # log
    logger = get_logger(args.store_dir)
    
    # training
    train_loader = train_dataloader(batch_size=args.batch_size)
    logger.info(f'{len(train_loader.dataset)} sequences, / {len(train_loader)} batches')
    training_loss = training(dataloader=train_loader, num_epochs=args.epochs)

    # testing
    test_loader = test_dataloader(batch_size=args.batch_size)
    logger.info(f'{len(test_loader.dataset)} sequences, / {len(test_loader)} batches')
    testing_loss = testing(dataloader=test_loader)