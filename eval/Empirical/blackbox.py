from __future__ import print_function


import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from advertorch.utils import NormalizeByChannelMeanStd
from advertorch.attacks import GradientSignAttack, LinfBasicIterativeAttack, LinfPGDAttack, LinfMomentumIterativeAttack, CarliniWagnerL2Attack, JacobianSaliencyMapAttack
from advertorch.attacks.utils import multiple_mini_batch_attack, attack_whole_dataset
from models.resnet import ResNet
import numpy as np
from models.ensemble import Ensemble
from tqdm import tqdm
import arguments, utils
class ModelWrapper(nn.Module):
    def __init__(self, model, normalizer):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.normalizer = normalizer

    def forward(self, x):
        x = self.normalizer(x)
        return self.model(x)

    def get_features(self, x, layer, before_relu=True):
        x = self.normalizer(x)
        return self.model.get_features(x, layer, before_relu)



# Subtracting pixel mean improves accuracy
subtract_pixel_mean = True

# Computed depth from supplied model parameter n
n = 3
depth = n * 6 + 2
version = 1

# Model name, depth and version
model_type = 'ResNet%dv%d' % (depth, version)
print(model_type)
import argparse, random
# Load the data

#print ("%s MODEL!!"%FLAGS.transfer_model)
leaky_relu = False #('gal' in FLAGS.transfer_model)

import torch
from advertorch.utils import NormalizeByChannelMeanStd


#mean = np.array([0.4914, 0.4822, 0.4465])
#std = np.array([0.2023, 0.1994, 0.2010])
mean = torch.tensor([0.5, ], dtype=torch.float32).cuda()
std = torch.tensor([0.5, ], dtype=torch.float32).cuda()
normalizer = NormalizeByChannelMeanStd(mean=mean, std=std)

#state_dict = torch.load('checkpoints/seed_0/advt/3_ResNet20_eps_0.200/epoch_120.pth')
state_dict = torch.load('checkpoints/advabstrs/seed_0/3_ResNet20_plus_adv_100.00_20.00_0.40/epoch_120.pth')
torch_models = []
iter_m = state_dict.keys()
for i in iter_m:
    model = ResNet(depth=depth, leaky_relu=leaky_relu)
    model = ModelWrapper(model, normalizer)  # Need to wrap together for loading
    model = nn.DataParallel(model)
    model.load_state_dict(state_dict[i])
    model.eval()
    model = model.cuda()
    torch_models.append(model)

torch_ens = Ensemble(torch_models)
torch_ens.eval()

import pickle
epslist = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

bsz = 100
kwargs = {'num_workers': 4,
          'batch_size': bsz,
          'shuffle': False,
          'pin_memory': True}

transform_test = transforms.Compose([
    transforms.ToTensor(),
])
random.seed(0)
subset_idx = random.sample(range(10000), 1000)
testset = Subset(datasets.MNIST(root="./data", train=False,
                                transform=transform_test,
                                download=True), subset_idx)

acc = []
for eps in epslist:
    with open("transfer/ntransfer_eps%.2f.pkl" % (eps), "rb") as tf:
        advx = pickle.load(tf)
    print(len(advx))
    N = advx[0].shape[0]
    print(N)
    testloader = DataLoader(testset, **kwargs)
    correct = 0
    allcnt = 0
    for i, data in enumerate(testloader):
        X, y = data
        X = X.cuda()
        y = y.cuda()
        preds = []
        for j in range(3 * 3 * 2):
            adv = torch.from_numpy((advx[j])[i * bsz:(i + 1)*bsz]).cuda()
            pred = torch_ens(adv).max(1,keepdim=False)[1]
            preds.append(pred)
        benign = torch_ens(X).max(1,keepdim=False)[1]
        preds.append(benign)
        correct += ((preds[0] == y) & (preds[1] == y) & (preds[2] == y) &
                    (preds[3] == y) & (preds[4] == y) & (preds[5] == y) &
                    (preds[6] == y) & (preds[7] == y) & (preds[8] == y) &
                    (preds[9] == y) & (preds[10] == y) & (preds[11] == y) &
                    (preds[12] == y) & (preds[13] == y) & (preds[14] == y) &
                    (preds[15] == y) & (preds[16] == y) & (preds[17] == y) & (preds[18] == y)
                    ).sum().item()
        allcnt += (preds[18] == y).sum().item()
    print(correct, allcnt)
    acc.append(correct * 100 / allcnt)
    print(correct * 100 / allcnt)
print(acc)



