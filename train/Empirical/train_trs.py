import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
from tensorboardX import SummaryWriter

import sys
import os

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.append(parentdir)

from utils.Empirical.architectures import ARCHITECTURES
from utils.Empirical.datasets import DATASETS

from utils.Empirical.utils_ensemble import AverageMeter, accuracy, test, copy_code, requires_grad_, evaltrans
from utils.Empirical.datasets import get_dataset
from utils.Empirical.architectures import get_architecture
from train.Empirical.trainer import TRS_Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('dataset', type=str, choices=DATASETS)
parser.add_argument('arch', type=str, choices=ARCHITECTURES)
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch', default=128, type=int, metavar='N',
                    help='batchsize (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--lr_step_size', type=int, default=40,
                    help='How often to decrease learning by gamma.')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--num-models', type=int, required=True)

parser.add_argument('--resume', action='store_true',
                    help='if true, tries to resume training from existing checkpoint')
parser.add_argument('--adv-training', action='store_true')
parser.add_argument('--epsilon', default=512, type=float)
parser.add_argument('--num-steps', default=4, type=int)

# DRT Training params
parser.add_argument('--coeff', default=2.0, type=float, required=True)
parser.add_argument('--lamda', default=2.0, type=float, required=True)
parser.add_argument('--scale', default=5.0, type=float, required=True)
parser.add_argument('--plus-adv', action='store_true')
parser.add_argument('--adv-eps', default=0.2, type=float)
parser.add_argument('--init-eps', default=0.1, type=float)

args = parser.parse_args()

if args.adv_training:
    mode = f"adv_{args.epsilon}_{args.num_steps}_{args.coeff}_{args.lamda}_{args.scale}"
else:
    mode = f"vanilla_{args.coeff}_{args.lamda}_{args.scale}"

args.outdir = f"/{args.dataset}/trs/{mode}/"

if (args.plus_adv):
    args.outdir += "%.2f-%.2f/" % (args.init_eps, args.adv_eps)
else:
    args.outdir += "0.0-0.0/"

args.epsilon /= 256.0

if (args.resume):
    args.outdir = "resume" + args.outdir
else:
    args.outdir = "scratch" + args.outdir

args.outdir = "logs/Empirical/" + args.outdir


def main():
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    copy_code(args.outdir)

    train_dataset = get_dataset(args.dataset, 'train')
    test_dataset = get_dataset(args.dataset, 'test')
    pin_memory = (args.dataset == "imagenet")
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch,
                              num_workers=args.workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch,
                             num_workers=args.workers, pin_memory=pin_memory)

    model = []
    for i in range(args.num_models):
        submodel = get_architecture(args.arch, args.dataset)
        submodel = nn.DataParallel(submodel)
        model.append(submodel)
    print("Model loaded")

    criterion = nn.CrossEntropyLoss().cuda()

    param = list(model[0].parameters())
    for i in range(1, args.num_models):
        param.extend(list(model[i].parameters()))

    # optimizer = optim.SGD(param, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = optim.Adam(param, lr=args.lr, weight_decay=args.weight_decay, eps=1e-7)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)

    model_path = os.path.join(args.outdir, 'checkpoint.pth.tar')
    writer = SummaryWriter(args.outdir)

    if (args.resume):
        base_classifier = "logs/Empirical/scratch/" + args.dataset + "/vanilla/checkpoint.pth.tar"
        print(base_classifier)
        for i in range(3):
            checkpoint = torch.load(base_classifier + ".%d" % (i))
            print("Load " + base_classifier + ".%d" % (i))
            model[i].load_state_dict(checkpoint['state_dict'])
            model[i].train()
        print("Loaded...")

    for epoch in range(args.epochs):

        TRS_Trainer(args, train_loader, model, criterion, optimizer, epoch, device, writer)
        test(test_loader, model, criterion, epoch, device, writer)
        evaltrans(args, test_loader, model, criterion, epoch, device, writer)

        scheduler.step(epoch)

        for i in range(args.num_models):
            model_path_i = model_path + ".%d" % (i)
            torch.save({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model[i].state_dict(),
                'optimizer': optimizer.state_dict(),
            }, model_path_i)


if __name__ == "__main__":
    main()
