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
N = 8
M = 50
# Experiments
exp_8 = ExperimentAnalytes(k0 = alists[0].k0.values, S = alists[0].S.values, h=0.001, run_time=1.0)
exp_16 = ExperimentAnalytes(k0 = alists[1].k0.values, S = alists[1].S.values, h=0.001, run_time=1.0)
exp_32 = ExperimentAnalytes(k0 = alists[2].k0.values, S = alists[2].S.values, h=0.001, run_time=1.0)
# Final Results 
dist_tau_8 = np.zeros((N, M))
dist_tau_16 = np.zeros((N, M))
dist_tau_32 = np.zeros((N, M))

for n in range(0, N): 
    print(f"{n}")
    for i in range(M):
        print(f"  {i}")
        #Policies
        pol_tau_8 = PolicySingleTime(n + 2, sigma_max = sigma_max)
        pol_tau_16 = PolicySingleTime(n + 2, sigma_max = sigma_max)
        pol_tau_32 = PolicySingleTime(n + 2, sigma_max = sigma_max)
        # Run Exp 8
        reinforce_delta_tau(
                exp_8, 
                pol_tau_8,
                **kwargs
            )
        # Run Exp 16
        reinforce_delta_tau(
                exp_16, 
                pol_tau_16,
                **kwargs
            )
        # Run Exp 32
        reinforce_delta_tau(
                exp_32, 
                pol_tau_32,
                **kwargs
            )
        
        exp_8.reset()
        exp_16.reset()
        exp_32.reset()
        mu_8, _ = pol_tau_8.forward()
        mu_16, _ = pol_tau_16.forward()
        mu_32, _ = pol_tau_32.forward()
        grads_8, delta_taus_8 = np.split(torch.cat((mu_8, tensor([1.])), 0).data.numpy(), 2)
        grads_16, delta_taus_16 = np.split(torch.cat((mu_16, tensor([1.])), 0).data.numpy(), 2)
        grads_32, delta_taus_32 = np.split(torch.cat((mu_32, tensor([1.])), 0).data.numpy(), 2)
        exp_8.run_all(grads_8, delta_taus_8)
        exp_16.run_all(grads_16, delta_taus_16)
        exp_32.run_all(grads_32, delta_taus_32)
        
        dist_tau_8[n, i] = exp_8.loss()
        dist_tau_16[n, i] = exp_16.loss()
        dist_tau_32[n, i] = exp_32.loss()


        np.savez_compressed("../results/distribution_single_delta_tau_perf_vs_n_steps", nr_analytes_8=dist_tau_8, nr_analytes_16=dist_tau_16, nr_analytes_32=dist_tau_32)