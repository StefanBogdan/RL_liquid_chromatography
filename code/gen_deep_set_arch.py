import numpy as np
import pandas as pd
from chromatography import *
from separation_utility import *
from torch import optim, tensor
import torch.nn as nn
import matplotlib.pyplot as plt
import time

alists = []
alists.append(pd.read_csv('../data/GilarSample.csv'))
alists.append(pd.read_csv('../data/Alizarin.csv'))
alists.append(pd.read_csv('../data/Peterpeptides.csv'))
alists.append(pd.read_csv('../data/Roca.csv'))
alists.append(pd.read_csv('../data/Peter32.csv'))
alists.append(pd.read_csv('../data/Eosin.csv'))
alists.append(pd.read_csv('../data/Controlmix2.csv'))
alists.append(pd.read_csv('../data/Gooding.csv'))
# GilarSample - 8 analytes
# Peterpeptides - 32 analytes
# Roca - 14 analytes
# Peter32 - 32 analytes
# Eosin - 20 analytes
# Alizarin - 16 analytes
# Controlmix2 - 17 analytes
# Gooding - 872 analytes

# Parameters
all_analytes = pd.concat(alists, sort=True).reset_index()[['k0', 'S', 'lnk0']]
activations = [nn.ELU, nn.ReLU, nn.Tanh]
width = [5, 10, 20]
kwargs = {
    'num_episodes' : 25_000, 
    'sample_size' : 10,
    'batch_size' : 1, 
    'lr' : .05, 
    'optim' : torch.optim.SGD,
    'lr_decay_factor' : 0.75,
    'lr_milestones' : 5000,
    'print_every' : 25_001,
    'baseline' : .55,
    'max_norm' : 1.5,
    'max_rand_analytes' : 40,
    'min_rand_analytes' : 8,
    'rand_prob' : 1.,
    'h' : 0.001,
    'run_time' : 1.
}

N = 9
M = 20

losses_deep_set = np.zeros((N, M, kwargs['num_episodes']))
test_losses_deep_set = np.zeros((N, M, kwargs['num_episodes']))

delta_taus = np.ones(10) * 1/(10)
for i in range(N):
    print(f"{i}")
    for m in range(M):
        alist_train = all_analytes.sample(frac=0.5)
        alist_test = all_analytes.loc[lambda a: ~a.index.isin(alist_train.index.values)]
        print(f"  {m}")
        #Policies
        pol_50_50 = PolicyGeneral(
            phi = nn.Sequential(
                PermEqui2_max(2, width[i % 3]),
                activations[i // 3](),
                PermEqui2_max(width[i % 3], width[i % 3]),
                activations[i // 3](),
                PermEqui2_max(width[i % 3], width[i % 3]),
                activations[i // 3](),
            ),
            rho = nn.Sequential(
                nn.Linear(width[i % 3], 5),
                nn.ELU(inplace=True),
                nn.Linear(5, 5),
                nn.ELU(inplace=True),
                Rho(n_steps=len(delta_taus), hidden=5, in_dim=5, sigma_max=.3, sigma_min=.01),
            )
        )
        
        # Run Exp
        loss, loss_test = reinforce_gen(
            alists = [alist_train], 
            test_alist = alist_test,
            policy = pol_50_50, 
            delta_taus = delta_taus, 
            **kwargs
        )

        losses_deep_set[i, m] = loss
        test_losses_deep_set[i, m] = loss_test

np.savez_compressed("../results/general_perf_deep_set_arch_loss_50_50", losses_50_50=losses_deep_set)
np.savez_compressed("../results/general_perf_deep_set_arch_test_losses_50_50", test_losses_50_50=test_losses_deep_set)