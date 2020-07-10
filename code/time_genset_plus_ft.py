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

kwargs_gen = {
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

kwargs_one = {
    'num_episodes' : 6000,
    'sample_size' : 10,
    'lr' : .05,
    'optim' : torch.optim.SGD,
    'lr_decay_factor' : .5,
    'lr_milestones' : 1000,
    'print_every' : 6001,
    'baseline' : 0.55,
    'max_norm' : 2.
}
N = 10

delta_taus = np.ones(3) * 1/(3)

# Final Results 
start_ft = time.perf_counter()
for n in range(0, N):
    
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
        **kwargs_gen
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
        **kwargs_gen
    )

end = time.perf_counter()
sigma_max = 0.3


exp= ExperimentAnalytes(k0 = alists[2].k0.values, S = alists[2].S.values, h=0.001, run_time=1.0)


########### three n_step
loss = 3.
# Grid Search
print("Grid Search 3")
start_gs_3 = time.perf_counter()
for phi_1 in np.linspace(0, 1, 100):
    for phi_2 in np.linspace(0, 1, 100):
        for phi_3 in np.linspace(0, 1, 100):
            exp.reset()
            exp.run_all([phi_1, phi_2, phi_3], delta_taus)
            if exp.loss() < loss:
                loss = exp.loss()
                best_3 = [phi_1, phi_2, phi_3]
end_gs_3 = time.perf_counter()
best_loss_gs[iterat, 2] = loss

# RL
print("RL 3")
start_rl_3 = time.perf_counter()
for n in range(N):
    pol = PolicySingle(len(delta_taus), sigma_max = sigma_max)
    reinforce_one_set(
            exp, 
            pol, 
            delta_taus=delta_taus, 
            **kwargs_one
        )
end_rl_3 = time.perf_counter()

# RL
print("RL 3")
start_rl_tau = time.perf_counter()
for n in range(N):
    pol = PolicySingleTime(len(delta_taus), sigma_max = sigma_max)
    reinforce_delta_tau(
            exp, 
            pol, 
            **kwargs_one
        )
end_rl_tau = time.perf_counter()



rl_time_ms = (end_rl_3 - start_rl_3)/N
rl_time_tau_ms = (end_rl_tau - start_rl_tau)/N
gen_set_ms = (end - start)/N
fine_tune = (end_ft - start_ft)/N - gen_set_ms
grid_search_time_ms = end_gs_3 - start_gs_3
times = np.array([rl_time_ms, rl_time_tau_ms, gen_set_ms, fine_tune, grid_search_time_ms])

(
    np.savetxt(
        "../results/time_genset_plus_ft_new.txt",
        times
        )
)