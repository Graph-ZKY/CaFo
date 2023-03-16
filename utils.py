"""
Implementation of necessary functions
"""
import torch
import numpy as np
import random
import torch.nn.functional as F


def one_hot(y, num_labels):
    h = torch.zeros(size=(y.shape[0], num_labels), dtype=torch.float32, device=y.device)
    h[range(y.shape[0]), y] = 1.0
    return h


def loss_MSE(outputs, labels):
    diff = outputs - labels
    loss = torch.sqrt(diff * diff).sum(1).mean()
    return loss

def loss_cross_entropy(outputs, labels):
    loss = - (labels * torch.log(F.softmax(outputs,dim=1))).sum(1).mean()
    return loss

def loss_sparsemax(outputs, labels):
    p, ta_z = sparsemax(outputs)
    valid = (p > 0.0).float()
    loss = - outputs * labels + 0.5 * valid * (outputs * outputs - ta_z * ta_z) + 0.5
    loss = loss.sum(1).mean()
    return loss



def jaccobian_sparsemax(outputs, labels):
    s, _ = sparsemax(outputs)
    jacc = s - labels
    return jacc

def jaccobian_cross_entropy(outputs, labels):
    s = F.softmax(outputs,dim=1)
    jacc = s - labels
    return jacc

def jaccobian_MSE(outputs, labels):
    jacc = 2 * (outputs - labels)
    return jacc


def sparsemax(z):
    sorted_z = z.sort(descending=True)[0]
    k_z = torch.zeros(size=(sorted_z.shape[0], 1), dtype=torch.int64, device=z.device)
    acc_z = sorted_z.clone()
    for k in range(1, sorted_z.shape[1]):
        acc_z[:, k] = acc_z[:, k - 1] + sorted_z[:, k]
        valid = (1.0 + (k + 1) * sorted_z[:, k]) > acc_z[:, k]
        k_z[valid] = k
    k_z = k_z.squeeze(-1)
    ta_z = (acc_z[range(acc_z.shape[0]), k_z] - 1.0) / (1 + k_z).float()
    ta_z = ta_z.unsqueeze(-1).repeat(1, z.shape[1])
    delta = z - ta_z
    p = F.relu(delta,inplace=True)
    return p, ta_z


def evaluate(labels, labels_te, y, y_te):
    train_error=1.0 - labels.eq(y).float().mean().item()
    test_error=1.0 - labels_te.eq(y_te).float().mean().item()
    return train_error,test_error

def overlay_y_on_x(x, y, num_classes):
    x_ = x.clone()
    x_[:, :num_classes] *= 0.0
    x_[range(x.shape[0]), y] = x.max()
    return x_

def append_y_on_x(x, y, num_classes):
    x_ = x.clone()
    y_ = torch.zeros(size = (x.shape[0], num_classes), device = x.device)
    y_[range(x.shape[0]), y] = x.max()
    xx = torch.cat((x_, y_), dim = 1)
    return xx

def set_seed(seed, cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False




