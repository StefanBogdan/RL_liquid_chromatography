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
all_analytes = pd.concat(alists, sort=True).reset_index()[['k0', 'S', 'lnk0']]

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
    'rand_prob' : .8,
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
N = 20

# Final Results 
start_ft = time.perf_counter()
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
        alists = alists[0:3], 
        random_alist = all_analytes,
        test_alist = None,
        policy = pol, 
        delta_taus = delta_taus, 
        **kwargs
    )

    mu_32, _ = pol.forward(torch.tensor(alists[2][['S', 'lnk0']].values, dtype=torch.float32))
    
    _,_,mu_32,_ = reinforce_single_from_gen(
        alist = alists[2], 
        policy= pol, 
        delta_taus= delta_taus,   
        **kwargs_ft
    )
end_ft = time.perf_counter()


# Final Results 
start = time.perf_counter()
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
        alists = alists[0:3], 
        random_alist = all_analytes,
        test_alist = None,
        policy = pol, 
        delta_taus = delta_taus, 
        **kwargs
    )

end = time.perf_counter()
gen_time = np.array([end - start, end_ft - start_ft])/N

(
    np.savetxt(
        "../results/time_genset_plus_ft.txt",
        gen_time
        )
)