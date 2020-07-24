# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 22:02:56 2020
@author: Yuance Li
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from MovingMNIST import MovingMNIST
from seq2seq_ConvLSTM import EncoderDecoderConvLSTM
from generate_video import create_array, generate_video


def training(dataloader, num_epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    model = EncoderDecoderConvLSTM(nf=args.n_hidden_dim, in_chan=1).to(device)
    criterion=nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr= args.lr)
    
    training_loss = []
    for epoch in range(num_epochs):
        running_loss = 0
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
        print('epoch: ', epoch, ' loss: ', epoch_loss)
    
    torch.save(model.state_dict(), './model.pth')
    return training_loss

def testing(dataloader):
    model = EncoderDecoderConvLSTM(nf=args.n_hidden_dim, in_chan=1)
    model.load_state_dict(torch.load('./model.pth'))
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
        train=True,
        data_root=os.getcwd() + '/data',
        seq_len=20,
        image_size=64,
        deterministic=True,
        num_digits=2)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True)
    return train_loader

def test_dataloader(batch_size):
    test_data = MovingMNIST(
        train=False,
        data_root=os.getcwd() + '/data',
        seq_len=20,
        image_size=64,
        deterministic=True,
        num_digits=2)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=True)
    return test_loader

if __name__ == '__main__':
    # hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train for')
    parser.add_argument('--n_hidden_dim', type=int, default=64, help='number of hidden dim for ConvLSTM layers')
    args = parser.parse_args()
    
    # training
    train_loader = train_dataloader(batch_size=args.batch_size)
    training_loss = training(dataloader=train_loader, num_epochs=args.epochs)
    
    # testing
    test_loader = test_dataloader(batch_size=args.batch_size)
    testing_loss = testing(dataloader=test_loader)