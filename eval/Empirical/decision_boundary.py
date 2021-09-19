import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse

import sys
import os

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.append(parentdir)

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
args = parser.parse_args()

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
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1,
                         num_workers=4, pin_memory=pin_memory)
dataiter = iter(test_loader)

max_eps = 10.0
eps_step = 0.2
np.random.seed(0)
lab2color = np.random.uniform(0,1,size=(10,3))
fig = plt.figure(figsize=(10,8))

correct = 0
for i in range(20):
    plt.subplot(4, 5, i+1)
    xi, yi = dataiter.next()
    xi = xi.cuda()
    yi = yi.cuda()
    xi.requires_grad = True
    pred = ensemble(xi)

    _, out = torch.max(pred, dim=1)

    correct += (out.item() == yi.item())
    xi.grad = None
    pred[0][yi.item()].backward()
    g1 = xi.grad.data.detach()
    g1 = g1 / g1.norm()

    g2 = torch.FloatTensor(np.random.randn(*g1.shape)).cuda()
    g2 = g2 / g2.norm()
    g2 = g2 - torch.dot(g1.view(-1),g2.view(-1)) * g1
    g2 = g2 / g2.norm()
    assert torch.dot(g1.view(-1),g2.view(-1)) < 1e-6

    x_epss = y_epss = np.arange(-max_eps, max_eps + eps_step, eps_step)
    to_plt = []
    for j, x_eps in enumerate(x_epss):
        x_inp = xi + x_eps * g1 + torch.FloatTensor(y_epss.reshape(-1, 1, 1, 1)).cuda() * g2
        pred = ensemble(x_inp)
        pred_c = torch.max(pred, dim=1)[1].cpu().detach().numpy()
        to_plt.append(lab2color[pred_c])

    to_plt = np.array(to_plt)

    plt.imshow(to_plt)
    plt.plot((len(x_epss)-1)/2,(len(y_epss)-1)/2,'ro')
    plt.axvline((len(x_epss)-1)/2, ymin=0.5, color='k', ls='--')
    plt.axis('off')


outpath = "./outputs/Empirical/Boundary/"

if not os.path.exists(outpath):
    os.makedirs(outpath)

fig.savefig(outpath + "%s.pdf" % (args.outfile), bbox_inches='tight')
plt.close(fig)