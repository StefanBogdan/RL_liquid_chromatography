import numpy as np
import pandas as pd
from chromatography import *
from separation_utility import *
import torch
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
all_analytes = pd.concat(alists[3:], sort=True).reset_index()[['k0', 'S', 'lnk0']]

kwargs = {
    'num_episodes' : 20_000, 
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
kwargs_ft = {
    'num_episodes' : 6000, 
    'sample_size':  10, 
    'lr': .05, 
    'optim' : torch.optim.SGD,
    'lr_decay_factor': .75,
    'lr_milestones':  1000,
    'print_every':  6001,
    'baseline': 0.55,
    'max_norm': 1.5,
    'beta': .0,
    'weights': [1., 1.],
    'h': .001,
    'run_time' : 1.  
}
N = 300

# Experiments
exp_8 = ExperimentAnalytes(k0 = alists[0].k0.values, S = alists[0].S.values, h=0.001, run_time=1.0)
exp_16 = ExperimentAnalytes(k0 = alists[1].k0.values, S = alists[1].S.values, h=0.001, run_time=1.0)
exp_32 = ExperimentAnalytes(k0 = alists[2].k0.values, S = alists[2].S.values, h=0.001, run_time=1.0)
# Final Results 
dist_8 = np.zeros((N,))
dist_16 = np.zeros((N,))
dist_32 = np.zeros((N,))
dist_ft_8 = np.zeros((N,))
dist_ft_16 = np.zeros((N,))
dist_ft_32 = np.zeros((N,))

for n in range(0, N):
    delta_taus = np.ones(3) * 1/(3)
    print(f"{n}")
    #Policies
    pol = PolicyGeneral(
        phi = nn.Sequential(
            PermEqui2_max(2, 5),
            nn.Tanh(),
            PermEqui2_max(5, 5),
            nn.Tanh(),
            PermEqui2_max(5, 5),
            nn.Tanh(),
        ),
        rho = nn.Sequential(
            nn.Linear(5, 5),
            nn.Tanh(),
            nn.Linear(5, 5),
            nn.Tanh(),
            Rho(n_steps=len(delta_taus), hidden=5, in_dim=5, sigma_max=.3, sigma_min=.01, non_linearity=torch.tanh),
        )
    )
    # Run Exp
    reinforce_gen(
        alists = [], 
        random_alist = all_analytes,
        test_alist = None,
        policy = pol, 
        delta_taus = delta_taus, 
        **kwargs
    )
    
    mu_8, _ = pol.forward(torch.tensor(alists[0][['S', 'lnk0']].values, dtype=torch.float32))
    mu_16, _ = pol.forward(torch.tensor(alists[1][['S', 'lnk0']].values, dtype=torch.float32))
    mu_32, _ = pol.forward(torch.tensor(alists[2][['S', 'lnk0']].values, dtype=torch.float32))
    exp_8.reset()
    exp_16.reset()
    exp_32.reset()
    
    exp_8.run_all(mu_8.tolist(), delta_taus)
    exp_16.run_all(mu_16.tolist(), delta_taus)
    exp_32.run_all(mu_32.tolist(), delta_taus)
    dist_8[n] = exp_8.loss()
    dist_16[n] = exp_16.loss()
    dist_32[n] = exp_32.loss()
    
    _,_,mu_8,_ = reinforce_single_from_gen(
        alist = alists[0], 
        policy= pol, 
        delta_taus= delta_taus,   
        **kwargs_ft
    )
    
    _,_,mu_16,_ = reinforce_single_from_gen(
        alist = alists[1], 
        policy= pol, 
        delta_taus= delta_taus,   
        **kwargs_ft
    )
    
    _,_,mu_32,_ = reinforce_single_from_gen(
        alist = alists[2], 
        policy= pol, 
        delta_taus= delta_taus,   
        **kwargs_ft
    )
    
    exp_8.reset()
    exp_8.run_all(mu_8[-1], delta_taus)
    exp_16.reset()
    exp_16.run_all(mu_16[-1], delta_taus)
    exp_32.reset()
    exp_32.run_all(mu_32[-1], delta_taus)
    
    dist_ft_8[n] = exp_8.loss()
    dist_ft_16[n] = exp_16.loss()
    dist_ft_32[n] = exp_32.loss()

(
    np.savez_compressed(
        "../results/general_dist_not_in_train_3", 
        dist_8=dist_8, 
        dist_16=dist_16, 
        dist_32=dist_32, 
        dist_ft_8=dist_ft_8, 
        dist_ft_16=dist_ft_16, 
        dist_ft_32=dist_ft_32, 
        )
)