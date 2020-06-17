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
    'rand_prob' : 1.,
    'h' : 0.001,
    'run_time' : 1.
}
N = 5
M = 30

losses_50_50 = np.zeros((N, M, kwargs['num_episodes']))
test_losses_50_50 = np.zeros((N, M, kwargs['num_episodes']))
losses_100 = np.zeros((N, M, kwargs['num_episodes']))


delta_taus = np.ones(10) * 1/(10)
for n in range(N):
    for i in range(M):
        alist_train = all_analytes.sample(frac=0.5)
        alist_test = all_analytes.loc[lambda a: ~a.index.isin(alist_train.index.values)]
        print(f"  {i}")
        #Policies
        pol_50_50 = PolicyGeneral(
            phi = nn.Sequential(
                PermEqui2_max(2, 5),
                nn.ELU(inplace=True),
                PermEqui2_max(5, 5),
                nn.ELU(inplace=True),
                PermEqui2_max(5, 5),
                nn.ELU(inplace=True),
            ),
            rho = nn.Sequential(
                nn.Linear(5, 5),
                nn.ELU(inplace=True),
                nn.Linear(5, 5),
                nn.ELU(inplace=True),
                Rho(n_steps=len(delta_taus), hidden=5, in_dim=5, sigma_max=.3, sigma_min=.01),
            )
        )
        pol_100 = PolicyGeneral(
            phi = nn.Sequential(
                PermEqui2_max(2, 5),
                nn.ELU(inplace=True),
                PermEqui2_max(5, 5),
                nn.ELU(inplace=True),
                PermEqui2_max(5, 5),
                nn.ELU(inplace=True),
            ),
            rho = nn.Sequential(
                nn.Linear(5, 5),
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
            min_rand_analytes = 8 * (n + 1),
            max_rand_analytes = 8 * (n + 1),
            **kwargs
        )
        loss_100, _ = reinforce_gen(
            alists = [all_analytes], 
            test_alist = None,
            policy = pol_100, 
            delta_taus = delta_taus,
            min_rand_analytes = 8 * (n + 1),
            max_rand_analytes = 8 * (n + 1),
            **kwargs
        )

        losses_50_50[n,i] = loss
        test_losses_50_50[n,i] = loss_test
        losses_100[n,i] = loss_100

np.savez_compressed("../results/general_perf_vs_nr_analytes_losses_50", losses_50_50=losses_50_50)
np.savez_compressed("../results/general_perf_vs_nr_analytes_test_losses_50_50", test_losses_50_50=test_losses_50_50)
np.savez_compressed("../results/general_perf_vs_nr_analytes_losses_100", losses_100=losses_100)