"""
Implementation of Cafo network
"""

import torch
import os
import torch.nn.functional as F
from block import Block
from ff_layer import Layer
from utils import overlay_y_on_x


class CaFo(torch.nn.Module):

    def __init__(self, name, loss_fn, num_classes, layers):
        super().__init__()
        self.name = name
        self.loss_fn = loss_fn
        self.num_classes = num_classes
        self.layers = layers

    def train_back_propagation(self, x, y, x_te):
        inputs = x
        inputs_te = x_te
        labels = torch.zeros((x.shape[0], self.num_classes), dtype=torch.float32, device=x.device)
        labels_te = torch.zeros((x_te.shape[0], self.num_classes), dtype=torch.float32, device=x_te.device)
        for i, layer in enumerate(self.layers):
            print('training layer', i, '...')
            features = layer.forward(inputs).detach()
            features_te = layer.forward(inputs_te).detach()

            features_flat = features.view(features.shape[0], -1)
            features_flat_te = features_te.view(features_te.shape[0], -1)

            layer.train_back_propagation(features_flat, y)

            outputs = F.softmax(layer.fc(features_flat.cuda()),dim=1)
            outputs_te = F.softmax(layer.fc(features_flat_te.cuda()),dim=1)

            labels += outputs.cpu()
            labels_te += outputs_te.cpu()

            # for next iteration
            inputs = features  # .detach()
            inputs_te = features_te  # .detach()
        return labels.argmax(1), labels_te.argmax(1)

    def train_layer_optimization(self, x, y, x_te):
        inputs = x
        inputs_te = x_te
        predicts = torch.zeros((x.shape[0], self.num_classes), dtype=torch.float32, device=x.device)
        predicts_te = torch.zeros((x_te.shape[0], self.num_classes), dtype=torch.float32, device=x_te.device)
        for i, layer in enumerate(self.layers):
            print('training layer', i, '...')
            features = layer.forward(inputs).detach()
            features_flat = features.view(features.shape[0], -1)
            features_te = layer.forward(inputs_te).detach()
            features_flat_te = features_te.view(features_te.shape[0], -1)

            if self.loss_fn == 'MSE':
                # err=[0.1260,0.3486]
                layer.train_closeform_MSE(features_flat, y)
                outputs = F.softmax(layer.linear(features_flat.cuda()),dim=1)
                outputs_te = F.softmax(layer.linear(features_flat_te.cuda()),dim=1)
            elif self.loss_fn == 'CE':
                # err=[0.1617,0.3257]
                layer.train_gradient_descent(features_flat, y)
                outputs = F.softmax(layer.linear(features_flat.cuda()),dim=1)
                outputs_te = F.softmax(layer.linear(features_flat_te.cuda()),dim=1)
            elif self.loss_fn == 'SL':
                # err=[0.0967,0.3365]
                layer.train_gradient_descent(features_flat, y)
                outputs, _ = layer.sparsemax(layer.linear(features_flat.cuda()))
                outputs_te, _ = layer.sparsemax(layer.linear(features_flat_te.cuda()))
            else:
                os.error('unknown loss function: ' + self.loss_fn)

            predicts += outputs.cpu()
            predicts_te += outputs_te.cpu()


            # for next iteration
            inputs = features
            inputs_te = features_te

        return predicts.argmax(1), predicts_te.argmax(1)

    def train(self, x, y, x_te):
        if self.loss_fn=='BP':
            labels,labels_te=self.train_back_propagation(x,y,x_te)
        else:
            labels, labels_te = self.train_layer_optimization(x, y, x_te)
        return labels,labels_te


def construct_CaFo(name, loss_fn, num_classes, num_epochs, num_batches, input_channels, input_size,n_blocks=3,step=0.01,lamda=0.0):
    # size computation
    # N = (W - F + 2P) / S + 1
    # W: input size
    # F: kernel size
    # P: padding
    # S: stride

    layers=[]
    inp_channels,out_channels = input_channels,32
    wp = input_size
    assert 0<n_blocks<4
    for i in range(n_blocks):
        out_channels = 32*pow(4,i)
        conv = torch.nn.Sequential(
            torch.nn.Conv2d(inp_channels, out_channels, 3, 1, 1),
            torch.nn.ReLU(True),
            # torch.nn.Sigmoid(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.BatchNorm2d(out_channels, eps=1e-4)
        )
        wc = int((wp - 3 + 2 * 1) / 1 + 1)
        wp = int((wc - 2 + 2 * 0) / 2 + 1)
        out_features = int(out_channels * wp * wp)
        inp_channels=out_channels
        layers += [Block(loss_fn, num_classes, num_epochs, num_batches, conv, out_features,step,lamda)]


    net = CaFo(name, loss_fn, num_classes, layers)
    return net