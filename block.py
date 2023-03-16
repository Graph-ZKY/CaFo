"""
Implementation of CaFo blocks
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from utils import one_hot,loss_MSE,jaccobian_MSE,loss_cross_entropy,jaccobian_cross_entropy,loss_sparsemax,jaccobian_sparsemax
from tqdm import tqdm

class Block(nn.Module):
    def __init__(self, loss_fn, num_classes, num_epochs, num_batches, conv_module, out_features,step,lamda):
        super().__init__()
        self.loss_fn = loss_fn
        self.conv_module = conv_module
        self.out_features = out_features
        self.conv_module.to('cpu')
        self.num_classes = num_classes
        self.lamda = lamda  # regularization factor
        assert loss_fn in ['BP','CE','SL','MSE']
        if loss_fn in ['BP','CE','SL']:
            self.num_epochs = num_epochs
            self.num_batches = num_batches
            self.step=step
            if loss_fn=='BP':
                self.fc = nn.Linear(in_features=self.out_features, out_features=num_classes, bias=False)
                self.fc.to('cuda')
                self.opt = Adam(self.fc.parameters(), lr=step, eps=1e-4)   # lr=0.001

    def forward(self, x):
        x = self.conv_module(x)
        return x

    def batch_forward(self, x, num_batches):
        input = x[:1, :, :, :]
        output = self.forward(input)

        fea = torch.zeros(size=(x.shape[0], output.shape[1], output.shape[2], output.shape[3]), device=x.device)
        batch_size = int(x.shape[0] / num_batches)
        for i in range(num_batches):
            batch_x = x[i * batch_size:(i + 1) * batch_size, :, :, :]
            batch_y = self.forward(batch_x).detach()
            fea[i * batch_size:(i + 1) * batch_size, :, :, :] = batch_y

        return fea

    def linear(self, x):
        return torch.matmul(x, self.W)

    def train_closeform_MSE(self, features, y):
        inputs = features.cuda()
        labels = one_hot(y, num_labels=self.num_classes).cuda()
        A = torch.matmul(inputs.t(), inputs)  # .cpu()
        B = torch.matmul(inputs.t(), labels)  # .cpu()
        # self.W = torch.matmul(torch.pinverse(A), B)  # .cuda()
        try:
            self.W = torch.linalg.solve(A, B)
        except:
            self.W = torch.matmul(torch.pinverse(A), B)

        return self.W

    def compute_loss(self, outputs, labels):
        if self.loss_fn == 'MSE':
            loss = loss_MSE(outputs, labels)
        elif self.loss_fn == 'CE':
            loss = loss_cross_entropy(outputs, labels)
        elif self.loss_fn == 'SL':
            loss = loss_sparsemax(outputs, labels)
        else:
            loss = 0.0
        # regularization term
        if self.lamda > 1e-4:
            loss += 0.5 * self.lamda * self.W.norm(p='fro')
        return loss

    def compute_gradient(self, inputs, outputs, labels):
        if self.loss_fn == 'MSE':
            jacc = jaccobian_MSE(outputs, labels)
        elif self.loss_fn == 'CE':
            jacc = jaccobian_cross_entropy(outputs, labels)
        elif self.loss_fn == 'SL':
            jacc = jaccobian_sparsemax(outputs, labels)
        else:
            jacc = None
        jacc = jacc.unsqueeze(1)
        xx = inputs.unsqueeze(-1)
        delta_W = torch.matmul(xx, jacc).mean(0)
        # regularization term
        if self.lamda > 1e-4:
            delta_W += self.lamda * self.W
        return delta_W

    def train_gradient_descent(self, features, y):
        num_samples = features.shape[0]
        num_dims = features.shape[1]
        self.W = torch.ones(size=(num_dims, self.num_classes), dtype=torch.float32, device='cuda')
        inputs = features.cuda()
        labels = one_hot(y, num_labels=self.num_classes).cuda()

        step = self.step
        for idx in range(self.num_epochs):
            batch_size = int(num_samples / self.num_batches)
            loss1 = []
            loss2 = []
            for k in range(self.num_batches):
                input_batch = inputs[k * batch_size: (k + 1) * batch_size, :]
                label_batch = labels[k * batch_size: (k + 1) * batch_size, :]

                outputs_batch = self.linear(input_batch)
                loss1 += [self.compute_loss(outputs_batch, label_batch)]
                delta_W = self.compute_gradient(input_batch, outputs_batch, label_batch)
                self.W += - step * delta_W
                outputs_batch = self.linear(input_batch)
                loss2 += [self.compute_loss(outputs_batch, label_batch)]
            if idx % 10 == 0:
                loss1 = sum(loss1) / len(loss1)
                loss2 = sum(loss2) / len(loss2)
                print(idx, ': loss1=', loss1, ', loss2=', loss2, ', step=', step)

        return self.W

    def train_back_propagation(self, features, y):
        y = one_hot(y, num_labels=self.num_classes)
        criterion = nn.CrossEntropyLoss()
        for i in tqdm(range(self.num_epochs)):
            avg_loss = []
            for k in range(self.num_batches):
                batch_size = int(features.shape[0] / self.num_batches)
                labels = y[k * batch_size: (k + 1) * batch_size].cuda()
                fea_batch = features[k * batch_size: (k + 1) * batch_size, :].cuda()

                #    fea_batch = nn.functional.dropout(fea_batch, p=self.dropout_p)

                # do not backprogation to convolution module, only train the fc layer
                fea_batch.detach()

                outputs = self.fc(fea_batch)
                # outputs = self.softmax(outputs)

                # loss = - (labels * torch.log(outputs)).sum(1).mean()
                loss = criterion(outputs, labels.argmax(1))
                avg_loss.append(loss)

                # this backward just compute the derivative and hence
                # is not considered backpropagation.
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

            if i % 50 == 0:
                avg_loss = sum(avg_loss) / len(avg_loss)
                print('loss=', avg_loss)
