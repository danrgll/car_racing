from __future__ import print_function

import sys
# sys.path.append("/..")

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
from torchsummary import summary


def read_data(datasets_dir="data", frac = 0.1):
    """
    This method reads the states and actions recorded in drive_manually.py
    and splits it into training/ validation set with respect to frac.
    """
    print("... read data")
    # data_file = os.path.join(datasets_dir, 'data.pkl.gzip')
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_file = os.path.join(root_dir, datasets_dir + "/big_data.pkl.gzip")
  
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


def preprocessing(X_train, y_train, X_valid, y_valid, history_length=2):
    """preprocess data"""
    # 1. convert the images in X_train/X_valid to gray scale. If you use rgb2gray() from utils.py, the output shape (96, 96, 1)
    X_train = rgb2gray(X_train)
    X_valid = rgb2gray(X_valid)



    # 2. you can train your model with discrete actions (as you get them from read_data) by discretizing the action space 
    #    using action_to_id() from utils.py.
    # cut y_train into parts in which we need the actions
    # number_of_parts = y_train.shape[0] // history_length
    # number_cut_off = y_train.shape[0] % history_length
    # if number_cut_off != 0:
    #    y_train = y_train[:-number_cut_off]
    #    X_train = X_train[:-number_cut_off]
    # y_action_frames = np.split(y_train, number_of_parts)
    # actions = [action_to_id(frame) for frame in y_action_frames]
    # y_train = np.array(actions).astype("int64")
    num_sequences = X_train.shape[0] - history_length
    sequences = []
    X_seq_train = []
    actions = []
    for i in range(num_sequences):
        seq = X_train[i:i + history_length]
        # print(seq)
        # action = y_train[i: i + history_length]
        action = y_train[i+ history_length]
        print(action)
        action = action_to_id(action)
        X_seq_train.append([seq])
        actions.append(action)
    y_train = np.array(actions).astype("int64")
    X_train = np.array(X_seq_train).astype("float32")
    print("after histroy")
    print(y_train.shape)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[2], X_train.shape[3], X_train.shape[4]))
    print(X_train.shape)

    num_sequences = X_valid.shape[0] - history_length
    X_seq_valid = []
    actions = []
    for i in range(num_sequences):
        seq = X_valid[i:i + history_length]
        # print(seq)
        # action = y_valid[i: i + history_length]
        action = y_valid[i + history_length]
        print(action)
        action = action_to_id(action)
        X_seq_valid.append([seq])
        actions.append(action)
    y_valid = np.array(actions).astype("int64")
    X_valid= np.array(X_seq_valid).astype("float32")
    print("after histroy valid")
    print(y_valid.shape)
    X_valid = X_valid.reshape((X_valid.shape[0], X_valid.shape[2], X_valid.shape[3], X_valid.shape[4]))
    print(X_valid.shape)



    """

    # preprocess valid data
    number_of_parts = y_valid.shape[0] // history_length
    number_cut_off = y_valid.shape[0] % history_length
    if number_cut_off != 0:
        y_valid = y_valid[:-number_cut_off]
        X_valid = X_valid[:-number_cut_off]
    y_action_frames = np.split(y_valid, number_of_parts)
    actions = [action_to_id(frame) for frame in y_action_frames]
    y_valid = np.array(actions).astype("int64")
    """
    # 1500 if history_length 1
    # 1200 if history_length 3
    X_train, y_train = balance_data(X_train, y_train, [0, 1, 2, 3], 1200)
    # mean = np.mean(X_train)
    print("mean")
    # std = np.std(X_train)

    # X_train = (X_train - mean) / std
    # print(X_train)
    # X_valid = (X_valid - mean) / std

    # X_train = X_train / 255
    # X_valid = X_valid / 255
    # print(X_train)

    # shuffle train data
    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)
    X_train = X_train[indices]
    y_train = y_train[indices]

    return X_train, y_train, X_valid, y_valid


def balance_data(data, labels, actions, max_number):
    print("balance data")
    print(labels)
    for action in actions:
        print(action)
        action_indices = np.where(labels == action)
        action_indices = action_indices[0].astype("int")
        print(action_indices)
        print(len(action_indices))
        # print(action_indices[0])
        # print(len(action_indices[0]))
        if len(action_indices) > max_number:
            indices = np.random.choice(action_indices, size=max_number, replace=False)
            # choosen_action_data_points = data[indices]

            other_actions_indeces = np.where(labels != action)
            other_actions_indeces = other_actions_indeces[0].astype("int")
            print(other_actions_indeces)
            print(type(other_actions_indeces))
            balanced_indices = np.concatenate((indices, other_actions_indeces), axis=0)
            # use the balanced_indices to get the balanced dataset
            balanced_data = data[balanced_indices]
            balanced_labels = labels[balanced_indices]
            data = balanced_data
            labels = balanced_labels
            print("Balanced Data Size")
            print(data.shape)
    return data, labels


def train_model(X_train, y_train, X_valid, y_valid, batch_size, epochs, lr, history_length, model_dir="./models", \
                                                                            tensorboard_dir="./tensorboard"):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # create result and model folders
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    print("... train model")


    # TODO: specify your agent with the neural network in agents/bc_agent.py 
    agent = BCAgent(history_length=history_length, device=device, lr=lr)
    summary(agent, (history_length, 96, 96), device='cuda' if torch.cuda.is_available() else 'cpu')
    
    tensorboard_eval = Evaluation(tensorboard_dir, "agent", ["train_loss", "train_acc", "val_loss", "val_acc"])

    # TODO: implement the training
    # 
    # 1. write a method sample_minibatch and perform an update step
    # 2. compute training/ validation accuracy and loss for the batch and visualize them with tensorboard. You can watch the progress of
    #    your training *during* the training in your web browser
    # 
    # training loop
    print(X_train.shape)
    print(y_train.shape)
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
    n_train_batches = len(X_train) // batch_size
    X_batches = np.split(X_train, n_train_batches)
    y_batches = np.split(y_train, n_train_batches)
    # print(X_batches.shape)
    # print(y_batches.shape)

    # split valid data into batches
    n_batches_valid = len(X_valid) // batch_size
    X_valid_batches = np.split(X_valid, n_batches_valid)
    y_valid_batches = np.split(y_valid, n_batches_valid)

    losses = AverageMeter()
    score = AverageMeter()
    for epoch in range(epochs):
        for i in range(n_train_batches):
            X_batch = X_batches[i]
            y_batch = y_batches[i]
            loss, acc = agent.update(X_batch, y_batch, device=device, batch_size=batch_size, history_length=history_length)
            n = X_batch.shape[0]
            losses.update(loss.item(), n)
            score.update(acc.item(), n)

        eval_score, eval_loss = eval_fn(agent, X_valid_batches, y_valid_batches, n_batches_valid, device,
                                        history_length)
        # compute training/ validation accuracy and write it to tensorboard
        tensorboard_eval.write_episode_data(epoch+1, {"train_loss": losses.avg, "train_acc": score.avg})
        print("Epoch:" + str(epoch))
        print("Training:")
        print("train loss:" + str(losses.avg))
        print("train acc:" + str(score.avg))
        tensorboard_eval.write_episode_data(epoch + 1, {"val_loss": eval_loss, "val_acc": eval_score})
        print("Validation:")
        print("val loss:" + str(eval_loss))
        print("val_acc:" + str(eval_score))
        losses.reset()
        score.reset()


      
    # TODO: save your agent
    model_dir = agent.save(os.path.join(model_dir, "agent.pt"))
    print("Model saved in file: %s" % model_dir)
    tensorboard_eval.close_session()


if __name__ == "__main__":
    # concatenate datasets
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_file = os.path.join(root_dir, "data" + "/data1.pkl.gzip")
    data_file2 = os.path.join(root_dir, "data" + "/data2.pkl.gzip")
    # concatenate(data_file, data_file2)
    print("concatenated datasets")
    history_length = 3

    # read data    
    X_train, y_train, X_valid, y_valid = read_data("data")

    # preprocess data
    X_train, y_train, X_valid, y_valid = preprocessing(X_train, y_train, X_valid, y_valid, history_length=history_length)

    # train model (you can change the parameters!)
    train_model(X_train, y_train, X_valid, y_valid, batch_size=64, epochs=15, lr=5e-4, history_length=history_length)
 
