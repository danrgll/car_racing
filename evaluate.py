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


def eval_fn(agent, X_batches_valid, y_batches_valid, n_batches, device, history_length):
    """
    Evaluation method
    """
    score = AverageMeter()
    losses = AverageMeter()
    agent.net.eval()

    with torch.no_grad():  # no gradient needed
        for i in range(n_batches):
            X_batch = X_batches_valid[i]
            y_batch = y_batches_valid[i]

            # transform data to tensors (batch)
            images = torch.from_numpy(X_batch)
            images = images.to(device)
            labels = torch.from_numpy(y_batch)
            labels = labels.to(device)
            # reshape tensor from (batchsize, hight, width) to (batchsize, 1, hight, width)
            images = images.view((images.size(0), history_length, 96, 96))

            outputs, pred = agent.predict(images, get_loss=True)
            loss = agent.criterion(pred, labels)
            losses.update(loss.item(), images.size(0))
            acc = accuracy(pred, labels)
            score.update(acc.item(), images.size(0))

    return score.avg, losses.avg