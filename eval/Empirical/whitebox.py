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
    CarliniWagnerL2Attack, JacobianSaliencyMapAttack, ElasticNetL1Attack
from advertorch.attacks.utils import attack_whole_dataset

from utils.Empirical.datasets import DATASETS, get_dataset

from utils.Empirical.architectures import get_architecture
from models.ensemble import Ensemble
import matplotlib
matplotlib.use('Agg')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('dataset', type=str, choices=DATASETS)
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("attack_type", type=str, help="choose from [fgsm, pgd, mim, bim, jsma, cw, ela]")
parser.add_argument('--num-models', type=int, required=True)
parser.add_argument('--adv-eps', default=0.2, type=float)
parser.add_argument('--adv-steps', default=10, type=int)
parser.add_argument('--random-start', default=1, type=int)
parser.add_argument('--coeff', default=0.1, type=float) # for jsma, cw, ela

args = parser.parse_args()



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
    testloader = DataLoader(test_dataset, shuffle=False, batch_size=128,
                             num_workers=4, pin_memory=pin_memory)

    loss_fn = nn.CrossEntropyLoss()

    correct_or_not = []
    for i in range(args.random_start):
        print("Phase %d" % (i))
        torch.manual_seed(i)
        test_iter = tqdm(testloader, desc='Batch', leave=False, position=2)

        if (args.attack_type == "pgd"):
            adversary = LinfPGDAttack(
                ensemble, loss_fn=loss_fn, eps=args.adv_eps,
                nb_iter=args.adv_steps, eps_iter=args.adv_eps / 10, rand_init=True, clip_min=0., clip_max=1.,
                targeted=False)
        elif (args.attack_type == "fgsm"):
            adversary = GradientSignAttack(
                ensemble, loss_fn=loss_fn, eps=args.adv_eps,
                clip_min=0., clip_max=1., targeted=False)
        elif (args.attack_type == "mim"):
            adversary = LinfMomentumIterativeAttack(
                ensemble, loss_fn=loss_fn, eps=args.adv_eps,
                nb_iter=args.adv_steps, eps_iter=args.adv_eps / 10, clip_min=0., clip_max=1.,
                targeted=False)
        elif (args.attack_type == "bim"):
            adversary = LinfBasicIterativeAttack(
                ensemble, loss_fn=loss_fn, eps=args.adv_eps,
                nb_iter=args.adv_steps, eps_iter=args.adv_eps / args.steps, clip_min=0., clip_max=1.,
                targeted=False)
        elif (args.attack_type == "cw"):
            adversary = CarliniWagnerL2Attack(
                ensemble, confidence=0.1, max_iterations=1000, clip_min=0., clip_max=1.,
                targeted=False, num_classes=10, binary_search_steps=1, initial_const=args.coeff)

        elif (args.attack_type == "ela"):
            adversary = ElasticNetL1Attack(
                ensemble, initial_const=args.coeff, confidence=0.1, max_iterations=100, clip_min=0., clip_max=1.,
                targeted=False, num_classes=10
            )
        elif (args.attack_type == "jsma"):
            adversary = JacobianSaliencyMapAttack(
                ensemble, clip_min=0., clip_max=1., num_classes=10, gamma=args.coeff)

        _, label, pred, advpred = attack_whole_dataset(adversary, test_iter, device="cuda")

        correct_or_not.append(label == advpred)
            
    correct_or_not = torch.stack(correct_or_not, dim=-1).all(dim=-1)

    print("")
    if (args.attack_type == "cw" or args.attack_type == "ela"):
        print("%s (c = %.2f): %.2f (Before), %.2f (After)" % (args.attack_type.upper(), args.adv_eps,
            100. * (label == pred).sum().item() / len(label),
            100. * correct_or_not.sum().item() / len(label)))
    elif (args.attack_type == "jsma"):
        print("%s (gamma = %.2f): %.2f (Before), %.2f (After)" % (args.attack_type.upper(), args.adv_eps,
            100. * (label == pred).sum().item() / len(label),
            100. * correct_or_not.sum().item() / len(label)))
    else:
        print("%s (eps = %.2f): %.2f (Before), %.2f (After)" % (args.attack_type.upper(), args.adv_eps,
            100. * (label == pred).sum().item() / len(label),
            100. * correct_or_not.sum().item() / len(label)))


if __name__ == '__main__':
    main()
