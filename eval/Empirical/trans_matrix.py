import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import sys
import os

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.append(parentdir)

from advertorch.attacks import GradientSignAttack, LinfBasicIterativeAttack, LinfPGDAttack, LinfMomentumIterativeAttack, \
    CarliniWagnerL2Attack, JacobianSaliencyMapAttack
from advertorch.attacks.utils import attack_whole_dataset

from utils.Empirical.datasets import DATASETS, get_dataset

from utils.Empirical.architectures import get_architecture
from models.ensemble import Ensemble
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('dataset', type=str, choices=DATASETS)
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("outfile", type=str, help="output file")
parser.add_argument('--num-models', type=int, required=True)
parser.add_argument('--adv-eps', default=0.2, type=float)
args = parser.parse_args()

def gen_plot(args, transmat):
    import itertools
    plt.figure(figsize=(6, 6))
    plt.yticks(np.arange(0, 3, step=1))
    plt.xticks(np.arange(0, 3, step=1))
    cmp = plt.get_cmap('Blues')
    plt.imshow(transmat, interpolation='nearest', cmap=cmp, vmin=0, vmax=100.0)
    plt.title("Transfer attack success rate")
    plt.colorbar()
    thresh = 50.0
    for i, j in itertools.product(range(transmat.shape[0]), range(transmat.shape[1])):
        plt.text(j, i, "{:0.2f}".format(transmat[i, j]),
                 horizontalalignment="center",
                 color="white" if transmat[i, j] > thresh else "black")

    plt.ylabel('Target model')
    plt.xlabel('Base model')
    outpath = "./outputs/Empirical/TransMatrix/"

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    plt.savefig(outpath + "%s.pdf" % (args.outfile), bbox_inches='tight')

def main():
    models = []
    for i in range(args.num_models):
        checkpoint = torch.load(args.base_classifier + ".%d" % (i))
        model = get_architecture(checkpoint["arch"], args.dataset)
        model = nn.DataParallel(model).cuda()
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        models.append(model)

    ensemble = Ensemble(models)
    ensemble.eval()

    print ('Model loaded')

    test_dataset = get_dataset(args.dataset, 'test')
    pin_memory = (args.dataset == "imagenet")
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=128,
                             num_workers=4, pin_memory=pin_memory)

    trans = np.zeros((3, 3))

    adv = []
    loss_fn = nn.CrossEntropyLoss()

    for i in range(len(models)):
        curmodel = models[i]
        adversary = LinfPGDAttack(
            curmodel, loss_fn=loss_fn, eps=args.adv_eps,
            nb_iter=5, eps_iter=args.adv_eps / 10, rand_init=True, clip_min=0., clip_max=1.,
            targeted=False)
        adv.append(adversary)

    for i in range(len(models)):
        test_iter = tqdm(test_loader, desc='Batch', leave=False, position=2)
        _, label, pred, advpred = attack_whole_dataset(adv[i], test_iter, device="cuda")
        for j in range(len(models)):
            for r in range((_.size(0) - 1) // 200 + 1):
                inputc = _[r * 200: min((r + 1) * 200, _.size(0))]
                y = label[r * 200: min((r + 1) * 200, _.size(0))]
                __ = adv[j].predict(inputc)
                output = (__).max(1, keepdim=False)[1]
                trans[i][j] += (output == y).sum().item()
            trans[i][j] /= len(label)

    print((1. - trans) * 100.)
    gen_plot(args, trans)

if __name__ == "__main__":
    main()
