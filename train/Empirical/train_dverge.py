
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse, random
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
from train.Empirical.trainer import DVERGE_Trainer
from utils.Empirical.third_party.distillation import DistillationLoader

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

# DVERGE Training params
parser.add_argument('--distill-eps', default=0.35, type=float,
                   help='perturbation budget for distillation')
parser.add_argument('--distill-alpha', default=0.035, type=float,
                   help='step size for distillation')
parser.add_argument('--distill-steps', default=10, type=int,
                   help='number of steps for distillation')
parser.add_argument('--distill-layer', default=None, type=int,
                   help='which layer is used for distillation, only useful when distill-fixed-layer is True')
parser.add_argument('--distill-rand-start', default=False, action="store_true",
                   help='whether use random start for distillation')
parser.add_argument('--distill-no-momentum', action="store_false", dest='distill_momentum',
                   help='whether use momentum for distillation')
parser.add_argument('--depth', default=20, type=int)
parser.add_argument('--adv-eps', default=0.2, type=float)
args = parser.parse_args()

if args.adv_training:
    mode = f"adv_{args.epsilon}_{args.num_steps}_{args.distill_eps}_{args.distill_alpha}_{args.distill_steps}"
else:
    mode = f"vanilla_{args.distill_eps}_{args.distill_alpha}_{args.distill_steps}"

if (args.distill_layer != None):
    mode += f"_{args.disill_layer}"
else:
    mode += "_rand"

args.outdir = f"/{args.dataset}/dverge/{mode}/"

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

    optimizers, schedulers = [], []
    for i in range(args.num_models):
        optimizer = optim.SGD(model[i].parameters(), lr=args.lr, momentum=args.momentum,
                        weight_decay=args.weight_decay)
        optimizers.append(optimizer)

    for optimizer in optimizers:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)
        schedulers.append(scheduler)


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


    distill_cfg = {'eps': args.distill_eps,
                        'alpha': args.distill_alpha,
                        'steps': args.distill_steps,
                        'layer': args.distill_layer,
                        'rand_start': args.distill_rand_start,
                        'before_relu': True,
                        'momentum': args.distill_momentum
                        }



    loader = DistillationLoader(train_loader, train_loader)
    for epoch in range(args.epochs):

        if args.distill_layer == None:
            distill_cfg['layer'] = random.randint(1, args.depth)
        DVERGE_Trainer(args, loader, model, criterion, optimizers, epoch, distill_cfg, device, writer)
        test(test_loader, model, criterion, epoch, device, writer)
        evaltrans(args, test_loader, model, criterion, epoch, device, writer)

        for i in range(len(schedulers)):
            schedulers[i].step()

        for i in range(args.num_models):
            model_path_i = model_path + ".%d" % (i)
            torch.save({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model[i].state_dict(),
                'optimizer': optimizers[i].state_dict(),
            }, model_path_i)


if __name__ == "__main__":
    main()
