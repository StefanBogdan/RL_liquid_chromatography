import numpy as np
import pandas as pd
from chromatography import ExperimentAnalytes
from separation_utility import *
from torch import optim, tensor
import matplotlib.pyplot as plt
import time

alists = []
alists.append(pd.read_csv(f'../data/GilarSample.csv'))
alists.append(pd.read_csv(f'../data/Alizarin.csv'))
alists.append(pd.read_csv(f'../data/Peterpeptides.csv'))
alists.append(pd.read_csv(f'../data/Roca.csv'))
alists.append(pd.read_csv(f'../data/Peter32.csv'))
alists.append(pd.read_csv(f'../data/Eosin.csv'))
alists.append(pd.read_csv(f'../data/Controlmix2.csv'))
# GilarSample - 8 analytes
# Peterpeptides - 32 analytes
# Roca - 14 analytes
# Peter32 - 32 analytes
# Eosin - 20 analytes
# Alizarin - 16 analytes
# Controlmix2 - 17 analytes
# Gooding - 872 analytes

# Parameters
sigma_max = 0.3

kwargs = {
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
M = 300
# Experiments
exp_8 = ExperimentAnalytes(k0 = alists[0].k0.values, S = alists[0].S.values, h=0.001, run_time=1.0)
exp_16 = ExperimentAnalytes(k0 = alists[1].k0.values, S = alists[1].S.values, h=0.001, run_time=1.0)
exp_32 = ExperimentAnalytes(k0 = alists[2].k0.values, S = alists[2].S.values, h=0.001, run_time=1.0)
# Final Results 
dist_8 = np.zeros((M,))
dist_16 = np.zeros((M,))
dist_32 = np.zeros((M,))
len_8 = 0
len_16 = 0
len_32 = 0
delta_taus = np.ones(3) * 1/(3)

for i in range(M):
    print(f"  {i}")
    #Policies
    pol_8 = PolicySingle(len(delta_taus), sigma_max = sigma_max)
    pol_16 = PolicySingle(len(delta_taus), sigma_max = sigma_max)
    pol_32 = PolicySingle(len(delta_taus), sigma_max = sigma_max)
    # Run Exp 8
    reinforce_one_set(
            exp_8, 
            pol_8, 
            delta_taus=delta_taus, 
            **kwargs
        )
    # Run Exp 16
    reinforce_one_set(
            exp_16, 
            pol_16, 
            delta_taus=delta_taus, 
            **kwargs
        )
    # Run Exp 32
    reinforce_one_set(
            exp_32, 
            pol_32, 
            delta_taus=delta_taus, 
            **kwargs
        )

    exp_8.reset()
    exp_16.reset()
    exp_32.reset()
    mu_8, _ = pol_8.forward()
    mu_16, _ = pol_16.forward()
    mu_32, _ = pol_32.forward()
    exp_8.run_all(mu_8.detach().numpy(), delta_taus)
    exp_16.run_all(mu_16.detach().numpy(), delta_taus)
    exp_32.run_all(mu_32.detach().numpy(), delta_taus)
    
    len_8 +=  len(exp_8.delta_taus)
    len_16 += len(exp_16.delta_taus)
    len_32 += len(exp_32.delta_taus)

    dist_8[i] = exp_8.loss()
    dist_16[i] = exp_16.loss()
    dist_32[i] = exp_32.loss()
avg_len = np.array([len_8, len_16, len_32])/M

(
    np.savez_compressed(
        "../results/distribution_single_const_3_steps", 
        nr_analytes_8=dist_8, 
        nr_analytes_16=dist_16, 
        nr_analytes_32=dist_32,
        avg_len=avg_len
    )
)