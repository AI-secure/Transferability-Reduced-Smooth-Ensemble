import os

gpu = [0,2,3]

gpuid = str(gpu[0])
for i in range(1, len(gpu)):
	gpuid += "," + str(gpu[i])

datasets = "mnist" # "cifar10", "imagenet"
model_type = None

if (datasets == "cifar10"): model_type = "cifar_resnet20"
elif (datasets == "mnist"): model_type = "mnist_resnet20" # "lenet" or "mnist_resnet20"
elif (datasets == "imagenet"): model_type = "resnet50"

adv_train = False
plus_adv = True

eps = 255
steps = 4

### TRS params
trs_coeff = 2.0
trs_lamda = 2.0
trs_scale = 5.0

trs_initeps = 0.1
trs_adveps = 0.2

### GAL params
gal_coeff = 0.5

### ADP params
adp_alpha = 2.0
adp_beta = 0.5

### DVERGE params
distill_eps = 0.35
distill_alpha = 0.035
distill_steps = 10
distill_layer = None


init_lr = 0.001
resume = False
num_models = 3

method = "dverge" # "dverge", "gal", "adp", "vanilla", "trs"

if not os.path.exists("train_scripts"):
	os.makedirs("train_scripts")

if not os.path.exists("train_scripts/Empirical"):
	os.makedirs("train_scripts/Empirical")

tmp = "train_scripts/Empirical/" + datasets + "_run_" + method


if (adv_train == True):
	tmp += "adv_%d_%d" % (eps, steps)

if (method == "trs"):
	tmp += "_%.2f_%.2f_%.2f" % (trs_coeff, trs_lamda, trs_scale)
	if (plus_adv == True):
		tmp += "_%.2f_%.2f" % (trs_initeps, trs_adveps)
elif (method == "gal"):
	tmp += "_%.2f" % (gal_coeff)
elif (method == "adp"):
	tmp += "_%.2f_%.2f" % (adp_alpha, adp_beta)
elif (method == "dverge"):
	tmp += "_%.2f_%.3f_%d" % (distill_eps, distill_alpha, distill_steps)
	if (distill_layer != None):
		tmp += "_%d" % (distill_layer)
	else:
		tmp += "_rand"


if (resume == True):
	tmp += "_resume"
else:
	tmp += "_scratch"

tmp += ".sh"
print(tmp)

import os

commd = "CUDA_VISIBLE_DEVICES=" + gpuid + " python train/Empirical/train_%s.py " % (method) + datasets + " " + \
		model_type + " --lr %.6f " % (init_lr)


if (resume == True): commd += "--resume "

if (adv_train == True):
	commd += "--adv-training --epsilon %d --num-steps %d " % (eps, steps)

if (method == "trs"):
	commd += "--coeff %.2f --lamda %.2f --scale %.2f " % (trs_coeff, trs_lamda, trs_scale)
	if (plus_adv == True):
		commd += "--init-eps %.2f --adv-eps %.2f " % (trs_initeps, trs_adveps)
elif (method == "gal"):
	commd += "--coeff %.2f " % (gal_coeff)
elif (method == "adp"):
	commd += "--alpha %.2f --beta %.2f " % (adp_alpha, adp_beta)
elif (method == "dverge"):
	commd += "--distill-eps %.2f --distill-alpha %.3f --distill-steps %d " % (distill_eps, distill_alpha, distill_steps)
	if (distill_layer != None):
		commd += "--distill-layer %d " % (distill_layer)

commd += "--num-models %d" % (num_models)


print(commd)
os.system("echo \"" + commd + "\" > " + tmp)
