from __future__ import print_function

import sys
# sys.path.append("../")

import pickle
import numpy as np
import os
import gzip
import matplotlib.pyplot as plt
import torch
from utils import *
from imitation_learning.agent.bc_agent import BCAgent
from tensorboard_evaluation import Evaluation
from torch.utils.data import DataLoader
from evaluate import AverageMeter, accuracy, eval_fn


def read_data(datasets_dir="data", frac = 0.1):
    """
    This method reads the states and actions recorded in drive_manually.py
    and splits it into training/ validation set with respect to frac.
    """
    print("... read data")
    # data_file = os.path.join(datasets_dir, 'data.pkl.gzip')
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_file = os.path.join(root_dir, datasets_dir + "/data.pkl.gzip")
  
    f = gzip.open(data_file,'rb')
    data = pickle.load(f)

    # get images as features and actions as targets
    X = np.array(data["state"]).astype('float32')
    y = np.array(data["action"]).astype('float32')

    # split data into training and validation set
    n_samples = len(data["state"])
    X_train, y_train = X[:int((1-frac) * n_samples)], y[:int((1-frac) * n_samples)]
    X_valid, y_valid = X[int((1-frac) * n_samples):], y[int((1-frac) * n_samples):]
    print("x")
    print(X_train.shape)
    print("y")
    print(y_train.shape)
    return X_train, y_train, X_valid, y_valid


def preprocessing(X_train, y_train, X_valid, y_valid, history_length=1):

    # TODO: preprocess your data here.
    # 1. convert the images in X_train/X_valid to gray scale. If you use rgb2gray() from utils.py, the output shape (96, 96, 1)
    X_train = rgb2gray(X_train)
    X_valid = rgb2gray(X_valid)

    # 2. you can train your model with discrete actions (as you get them from read_data) by discretizing the action space 
    #    using action_to_id() from utils.py.
    # cut y_train into parts in which we need the actions
    number_of_parts = y_train.shape[0] // history_length
    number_cut_off = y_train.shape[0] % history_length
    if number_cut_off != 0:
        y_train = y_train[:-number_cut_off]
        X_train = X_train[:-number_cut_off]
    y_action_frames = np.split(y_train, number_of_parts)
    actions = [action_to_id(frame) for frame in y_action_frames]
    y_train = np.array(actions).astype("int64")
    # ToDO: preprocessing for valid data fix
    y_valid = action_to_id(y_valid)

    # History:
    # At first you should only use the current image as input to your network to learn the next action. Then the input states
    # have shape (96, 96, 1). Later, add a history of the last N images to your state so that a state has shape (96, 96, N).
    
    return X_train, y_train, X_valid, y_valid


def train_model(X_train, y_train, X_valid, y_valid, batch_size, epochs, lr, model_dir="./models", \
                                                                            tensorboard_dir="./tensorboard"):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # create result and model folders
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    print("... train model")


    # TODO: specify your agent with the neural network in agents/bc_agent.py 
    agent = BCAgent(device=device, lr=lr)
    
    tensorboard_eval = Evaluation(tensorboard_dir, "agent", ["train_loss", "train_acc", "val_loss", "val_acc"])

    # TODO: implement the training
    # 
    # 1. write a method sample_minibatch and perform an update step
    # 2. compute training/ validation accuracy and loss for the batch and visualize them with tensorboard. You can watch the progress of
    #    your training *during* the training in your web browser
    # 
    # training loop
    remainder = len(X_train) % batch_size
    if remainder > 0:
        # Discard the remainder batch
        X_train = X_train[:-remainder]
        y_train = y_train[:-remainder]

    remainder = len(X_valid) % batch_size
    if remainder > 0:
        X_valid = X_valid[:-remainder]
        y_valid = y_valid[:-remainder]

    # split training data into batches
    n_batches = len(X_train) // batch_size
    X_batches = np.split(X_train, n_batches)
    y_batches = np.split(y_train, n_batches)

    # split valid data into batches
    n_batches_valid = len(X_train) // batch_size
    X_valid_batches = np.split(X_valid, n_batches_valid)
    y_valid_batches = np.split(y_valid, n_batches_valid)

    losses = AverageMeter()
    score = AverageMeter()
    for epoch in range(epochs):
        for i in range(n_batches):
            X_batch = X_batches[i]
            y_batch = y_batches[i]
            loss, acc = agent.update(X_batch, y_batch, device=device, batch_size=batch_size)
            n = X_batch.shape[0]
            losses.update(loss.item(), n)
            score.update(acc.item(), n)

        eval_score, eval_loss = eval_fn(agent, X_valid_batches, y_valid_batches, n_batches_valid, device)
        # compute training/ validation accuracy and write it to tensorboard
        tensorboard_eval.write_episode_data(epoch+1, {"train_loss": losses.avg, "train_acc": score.avg})
        tensorboard_eval.write_episode_data(epoch + 1, {"val_loss": losses.avg, "val_acc": score.avg})
      
    # TODO: save your agent
    model_dir = agent.save(os.path.join(model_dir, "agent.pt"))
    print("Model saved in file: %s" % model_dir)


if __name__ == "__main__":

    # read data    
    X_train, y_train, X_valid, y_valid = read_data("data")

    # preprocess data
    X_train, y_train, X_valid, y_valid = preprocessing(X_train, y_train, X_valid, y_valid, history_length=1)

    # train model (you can change the parameters!)
    train_model(X_train, y_train, X_valid, y_valid, batch_size=32, epochs=15, lr=1e-4)
 
