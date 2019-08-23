import argparse
import random
import time

from torch.optim import SGD
from models import resnet_convnet
import torch
from torchvision import transforms, datasets
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR

from models.alexnet import AlexNet
from models.simple import Simple
from quantization_sgd import QuantizedSGD
import sys

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            sys.stdout.flush()


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, metavar="M",
                        help="SGD weight decay (default: 1e-4)")
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument("--quantization-type", type=str, default="none",
                        help="Quantization Type (default: none)")
    parser.add_argument("--quantization-level", type=int, default=16,
                        help="Quantization Level (default: 16)")
    parser.add_argument("--bucket-size", type=int, default=512,
                        help="Quantization bucket size (default: 512)")
    parser.add_argument("--optimizer", type=str, default="standard",
                        help="Optimizer (default: standard)")
    parser.add_argument("--model", type=str, default="resnet_20",
                        help="Model (default: standard)")

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    print('Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_set = datasets.CIFAR10(root='data/cifar10', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                               shuffle=True, **kwargs)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_set = datasets.CIFAR10(root='data/cifar10', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size,
                                              shuffle=False, **kwargs)

    if args.model == "resnet_20":
        model = resnet_convnet.resnet_convnet(depth=20).to(device)
        args.lr = 0.1
        args.momentum = 0.9
        args.weight_decay = 1e-4
    elif args.model == "alexnet":
        model = AlexNet().to(device)
    elif args.model == "simple":
        model = Simple().to(device)

    if args.optimizer == "standard":
        optimizer = SGD(model.parameters(),
                        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == "quantization":
        if args.quantization_type == "none":
            optimizer = SGD(model.parameters(),
                            lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        else:
            optimizer = QuantizedSGD(model.parameters(),
                                     args.quantization_type, args.quantization_level, args.bucket_size,
                                     lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # test(args, model, device, test_loader)

    # Trying to do as in convnet
    scheduler = None
    if args.model == "resnet_20":
        scheduler = MultiStepLR(optimizer, milestones=[81, 122, 164])

    for epoch in range(args.epochs):
        start = time.time()
        train(args, model, device, train_loader, optimizer, epoch)
        end = time.time()
        print("Epoch {} with time {}".format(epoch, end - start))
        test(args, model, device, test_loader)
        if scheduler != None:
            scheduler.step()

    print(args.lr, args.momentum)
    torch.save(model.state_dict(), "results/cifar/cnn_%s_%d_%f_%f.pt" %
               (args.quantization_type, args.quantization_level, args.lr, args.momentum))


if __name__ == '__main__':
    main()