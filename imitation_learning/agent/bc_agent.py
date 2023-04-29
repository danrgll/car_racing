import torch
from torch import nn
from imitation_learning.agent.networks import CNN
from evaluate import accuracy


class BCAgent:
    
    def __init__(self, history_length, device, lr):
        # TODO: Define network, loss function, optimizer
        self.net = CNN(history_length=history_length)
        self.criterion = nn.CrossEntropyLoss().to(device)
        # self.optimizer = torch.optim.SGD(self.net.parameters(), lr=lr)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

    def update(self, X_batch, y_batch, device, batch_size, history_length):
        self.optimizer.zero_grad()
        # transform input to tensors
        X_tensor = torch.from_numpy(X_batch)
        X_tensor = X_tensor.to(device)
        y_tensor = torch.from_numpy(y_batch)
        y_tensor = y_tensor.to(device)
        print("input")
        print(X_tensor.device)
        print(self.net.model.device)
        # reshape tensor from (batchsize, hight, width) to (batchsize, 1, hight, width)
        X_tensor = X_tensor.view((batch_size, history_length, 96, 96))
        # forward + backward + optimize
        prediction = self.net(X_tensor)
        loss = self.criterion(prediction, y_tensor)
        acc = accuracy(prediction, y_tensor)
        loss.backward()
        self.optimizer.step()
        return loss, acc

    def predict(self, X, get_loss=False):
        """predict Tensor Xr"""
        pred = self.net.forward(X)
        if get_loss is True:
            _, outputs = torch.max(pred.data, 1)
            return outputs, pred
        else:
            _, outputs = torch.max(pred.data, 1)
            return outputs

    def load(self, file_name):
        self.net.load_state_dict(torch.load(file_name))

    def save(self, file_name):
        torch.save(self.net.state_dict(), file_name)

