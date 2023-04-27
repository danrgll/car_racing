import torch
import numpy as np

class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(logits, labels):
    preds = torch.argmax(logits, dim=1)
    return torch.sum(preds == labels) / len(labels)


def eval_fn(agent, X_valid, y_valid, n_batches, batchsize, device):
    """
    Evaluation method
    """
    score = AverageMeter()
    losses = AverageMeter()
    agent.eval()

    # transform input to tensors (batch)
    X_tensor = torch.from_numpy(X_valid)
    X_tensor = X_tensor.to(device)
    y_tensor = torch.from_numpy(y_valid)
    y_tensor = y_tensor.to(device)
    # reshape tensor from (batchsize, hight, width) to (batchsize, 1, hight, width)
    X_tensor = X_tensor.view((X_tensor.size(0), 1, 96, 96))
    with torch.no_grad():  # no gradient needed

            images = X_tensor.to(device)
            labels = y_tensor.to(device)

            outputs, pred = agent.predict(images, get_loss=True)
            loss = agent.criterion(pred, labels)
            losses.update(loss.item(), images.size(0))
            acc = accuracy(outputs, labels)
            score.update(acc.item(), images.size(0))

    return score.avg, losses.avg