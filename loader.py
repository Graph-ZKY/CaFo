"""
Implementation of dataloader
"""
import torch
import torchvision
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def load_data_cafo(args):
    if args.data == 'CIFAR10':
        train_loader, test_loader = CIFAR10_loaders()
        num_classes, input_channels, input_size=10,3,32
    elif args.data == 'CIFAR100':
        train_loader, test_loader = CIFAR100_loaders()
        num_classes, input_channels, input_size = 100, 3, 32
    elif args.data == 'MNIST':
        train_loader, test_loader = MNIST_loaders()
        num_classes, input_channels, input_size = 10, 1, 28
    elif args.data == 'ImageNet':
        train_loader, test_loader = ImageNet_loaders()
        num_classes, input_channels, input_size = 100, 3, 84
    else:
        raise Exception
    return train_loader,test_loader,num_classes,input_channels,input_size







def MNIST_loaders(train_batch_size=50000, test_batch_size=10000):
    transform = Compose([
        #    transforms.Resize(224),
        ToTensor(),
        Normalize((0.1307,), (0.3081,))
    ])
    train_loader = DataLoader(
        MNIST('./data/', train=True,
              download=True,
              transform=transform),
        batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(
        MNIST('./data/', train=False,
              download=True,
              transform=transform),
        batch_size=test_batch_size, shuffle=False)
    return train_loader, test_loader


def CIFAR10_loaders(train_batch_size=50000, test_batch_size=10000):
    transform_train = transforms.Compose([
        # 对原始32*32图像四周各填充4个0像素（40*40），然后随机裁剪成32*32
        #    transforms.RandomCrop(32, padding=4),
        # 按0.5的概率水平翻转图片
        #    transforms.RandomHorizontalFlip(),

        #    transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_test = transforms.Compose([
        #    transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    trainset = torchvision.datasets.CIFAR10('./data/', train=True,
                                            download=True, transform=transform_train)
    train_loader = DataLoader(trainset, batch_size=train_batch_size, shuffle=True)

    testset = torchvision.datasets.CIFAR10('./data/', train=False,
                                           download=True, transform=transform_test)
    test_loader = DataLoader(testset, batch_size=test_batch_size, shuffle=False)
    return train_loader, test_loader


def CIFAR100_loaders(train_batch_size=50000, test_batch_size=10000):
    transform_train = transforms.Compose([
        # 对原始32*32图像四周各填充4个0像素（40*40），然后随机裁剪成32*32
        #    transforms.RandomCrop(32, padding=4),
        # 按0.5的概率水平翻转图片
        #    transforms.RandomHorizontalFlip(),

        #    transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_test = transforms.Compose([
        #    transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    trainset = torchvision.datasets.CIFAR100('./data/', train=True,
                                             download=True, transform=transform_train)
    train_loader = DataLoader(trainset, batch_size=train_batch_size, shuffle=True)

    testset = torchvision.datasets.CIFAR100('./data/', train=False,
                                            download=True, transform=transform_test)
    test_loader = DataLoader(testset, batch_size=test_batch_size, shuffle=False)
    return train_loader, test_loader



def ImageNet_loaders(train_batch_size=50000, test_batch_size=10000):
    transform_train = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(84),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transform_test = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(84),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset_train = datasets.ImageFolder('./mini_imagenet/images_CDD/train', transform_train)
    dataset_test = datasets.ImageFolder('./mini_imagenet/images_CDD/test', transform_test)
    # dataset_val = datasets.ImageFolder('data/val', transform)

    # 上面这一段是加载测试集的
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=train_batch_size, shuffle=True)  # 训练集
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=test_batch_size, shuffle=False)  # 测试集

    return train_loader, test_loader



