from __future__ import print_function


import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torch.nn.functional as F
from advertorch.utils import NormalizeByChannelMeanStd
from advertorch.attacks import GradientSignAttack, LinfBasicIterativeAttack, LinfPGDAttack, LinfMomentumIterativeAttack, CarliniWagnerL2Attack, JacobianSaliencyMapAttack
from advertorch.attacks.utils import multiple_mini_batch_attack, attack_whole_dataset
from models.resnet import ResNet
import numpy as np
from models.ensemble import Ensemble
from advertorch.utils import to_one_hot
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

class CarliniWagnerLoss(nn.Module):
    """
    Carlini-Wagner Loss: objective function #6.
    Paper: https://arxiv.org/pdf/1608.04644.pdf
    """
    def __init__(self, conf=50.):
        super(CarliniWagnerLoss, self).__init__()
        self.conf = conf

    def forward(self, input, target):
        """
        :param input: pre-softmax/logits.
        :param target: true labels.
        :return: CW loss value.
        """
        num_classes = input.size(1)
        label_mask = to_one_hot(target, num_classes=num_classes).float()
        correct_logit = torch.sum(label_mask * input, dim=1)
        wrong_logit = torch.max((1. - label_mask) * input, dim=1)[0]
        loss = -F.relu(correct_logit - wrong_logit + self.conf).sum()
        return loss

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


#state_dict = torch.load('checkpoints/gal/seed_0/3_ResNet20MNIST/epoch_60.pth')#transfer_pth/%s.pth'%FLAGS.transfer_model)
#transfer_pth/%s.pth'%FLAGS.transfer_model)

#mean = np.array([0.4914, 0.4822, 0.4465])
#std = np.array([0.2023, 0.1994, 0.2010])
mean = torch.tensor([0.5, ], dtype=torch.float32).cuda()
std = torch.tensor([0.5, ], dtype=torch.float32).cuda()
normalizer = NormalizeByChannelMeanStd(mean=mean, std=std)

model_list = [3, 5, 8]
eps = 0.4

surrogate = []
for num_models in model_list:
    state_dict = torch.load('checkpoints/baseline/seed_0/%d_ResNet20/epoch_60.pth' % (num_models))
    torch_models = []
    iter_m = state_dict.keys()
    for i in iter_m:
        model = ResNet(depth=depth, leaky_relu=leaky_relu)
        model = ModelWrapper(model, normalizer) # Need to wrap together for loading
        model = nn.DataParallel(model)
        model.load_state_dict(state_dict[i])
        model.eval()
        model = model.cuda()
        torch_models.append(model)

    torch_ens = Ensemble(torch_models)
    torch_ens.eval()
    surrogate.append(torch_ens)

print ('Model loaded')
#print (torch_ens)
kwargs = {'num_workers': 4,
          'batch_size': 100,
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

transfer = []
print(subset_idx)
for curmodel in surrogate:
    for i in range(2):
        for j in range(3):
            testloader = DataLoader(testset, **kwargs)
            loss_fn = nn.CrossEntropyLoss() if (i == 1) else CarliniWagnerLoss(conf=.1)
            adversary = LinfPGDAttack(
                curmodel, loss_fn=loss_fn, eps=eps,
                nb_iter=50, eps_iter=eps / 10, rand_init=True, clip_min=0., clip_max=1.,
                targeted=False)
            test_iter = tqdm(testloader, desc='Batch', leave=False, position=2)
            _, label, pred, advpred = attack_whole_dataset(adversary, test_iter, device="cuda")
            _ = _.cpu().detach().numpy()
            transfer.append(_)
            print(curmodel, i, j)

import pickle
with open("transfer/ntransfer_eps%.2f.pkl" % (eps), "wb") as tf:
    pickle.dump(transfer, tf)

