import torch
import argparse
from loader import load_data_cafo
from utils import evaluate,set_seed
from model import construct_CaFo



def run_CaFo(args):

    torch.cuda.set_device(args.gpu)
    train_loader, test_loader,num_classes,input_channels,input_size = load_data_cafo(args)
    x, y = next(iter(train_loader))
    x_te, y_te = next(iter(test_loader))

    net = construct_CaFo(name=args.data,
                               loss_fn=args.loss_fn,
                               num_classes=num_classes,
                               num_epochs=args.num_epochs,
                               num_batches=args.num_batches,
                               input_channels=input_channels,
                               input_size=input_size)

    labels, labels_te = net.train(x, y, x_te)
    train_error,test_error=evaluate(labels, labels_te, y, y_te)
    print('train error:', train_error)
    print('test error:', test_error)







if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Cascaded Forward Algorithm')
    parser.add_argument('--data', type=str, default='CIFAR10',
                        help='dataset for training (default: CIFAR10)')
    parser.add_argument('--seed', type=int, default=1234,
                        help='random seed')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU id')
    parser.add_argument('--num_epochs', type=int, default=5000,
                        help='number of epochs for GD (default: 5000)')
    parser.add_argument('--step', type=float, default=0.01,
                        help='optimization step for GD (default: 0.001)/learning rate for BP')
    parser.add_argument('--loss_fn', type=str, default="MSE",
                        help='MSE(mean square error)/CE(cross entropy)/SL(sparsemax loss)/BP(backpropagation)')
    parser.add_argument('--num_batches', type=int, default=1,
                        help='number of batches for BP, CE and SL')
    parser.add_argument('--lamda', type=float, default=0.0,
                        help='regularization factor')
    parser.add_argument('--num_blocks', type=int, default=3,
                        help='number of blocks')
    args = parser.parse_args()

    set_seed(args.seed)


    run_CaFo(args)


