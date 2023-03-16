<<<<<<< HEAD
# CaFo
A pytorch implementation of Cascaded Forward (CaFo) Algorithm
=======
# Cascaded Forward Algorithm

A pytorch implementation of the Cascaded Forward (CaFo) Algorithm 


### Dependencies
* PyTorch 1.0+
* tqdm


```
pip install torch torchvision torchaudio
pip install tqdm
```
### Datasets

* MNIST http://yann.lecun.com/exdb/mnist/
* CIFAR10 https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
* CIFAR100 https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
* Mini-ImageNet https://studio.brainpp.com/dataset/2313?name=mini_imagenet

To execute the code on Mini-ImageNet, please preprocess the original dataset:
```
mkdir mini_imagenet
python imagenet_process.py
```


### Image Classification

Image classification on four datasets with three different strategies: mean square loss (MSE),
cross entropy (CE) and sparsemax loss (SL). In addition, the backpropagation (BP) optimization strategy
for cross entropy loss is also provided.

CIFAR10
```
python train.py --data CIFAR10 --loss_fn MSE
python train.py --data CIFAR10 --loss_fn CE --num_epochs 5000
python train.py --data CIFAR10 --loss_fn SL --num_epochs 5000
python train.py --data CIFAR10 --loss_fn BP --num epochs 5000
```

CIFAR100
```
python train.py --data CIFAR100 --loss_fn MSE
python train.py --data CIFAR100 --loss_fn CE --num_epochs 1000
python train.py --data CIFAR100 --loss_fn SL --num_epochs 1000
python train.py --data CIFAR100 --loss_fn BP --num epochs 1000
```


MNIST
```
python train.py --data MNIST --loss_fn MSE
python train.py --data MNIST --loss_fn CE --num_epochs 5000
python train.py --data MNIST --loss_fn SL --num_epochs 5000
python train.py --data MNIST --loss_fn BP --num epochs 5000
```

Mini-ImageNet
```
python train.py --data ImageNet --loss_fn MSE
python train.py --data ImageNet --loss_fn CE --num_epochs 1000
python train.py --data ImageNet --loss_fn SL --num_epochs 1000
python train.py --data ImageNet --loss_fn BP --num epochs 1000
```


