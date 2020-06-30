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
exps = [ExperimentAnalytes(k0 = alists[0].k0.values, S = alists[0].S.values, h=0.001, run_time=1.0),
    ExperimentAnalytes(k0 = alists[1].k0.values, S = alists[1].S.values, h=0.001, run_time=1.0),
       ExperimentAnalytes(k0 = alists[2].k0.values, S = alists[2].S.values, h=0.001, run_time=1.0)]
best_loss_gs = np.zeros((3, 4))
grid_search_time_ms = np.zeros((3, 4))
rl_time_ms = np.zeros((3, 4))
iterat = 0
for exp in exps:
    ########## one n_step
    loss = 3.
    delta_taus = [1.]
    print("Grid Search 1")
    # Grid Search
    start_gs_1 = time.perf_counter()
    for phi in np.linspace(0, 1, 100):
        exp.reset()
        exp.run_all([phi], delta_taus)
        if exp.loss() < loss:
            loss = exp.loss()
            best_1 = [phi]
    end_gs_1 = time.perf_counter()
    best_loss_gs[iterat, 0] = loss
    
    # RL
    print("RL 1")
    start_rl_1 = time.perf_counter()
    for n in range(10):
        pol = PolicySingle(len(delta_taus), sigma_max = sigma_max)
        reinforce_one_set(
                exp, 
                pol, 
                delta_taus=delta_taus, 
                **kwargs
            )
    end_rl_1 = time.perf_counter()


    ############ two n_step
    loss = 3.
    delta_taus = [.5, .5]
    # Grid Search
    print("Grid Search 2")
    start_gs_2 = time.perf_counter()
    for phi_1 in np.linspace(0, 1, 100):
        for phi_2 in np.linspace(0, 1, 100):
            exp.reset()
            exp.run_all([phi_1, phi_2], delta_taus)
            if exp.loss() < loss:
                loss = exp.loss()
                best_2 = [phi_1, phi_2]
    end_gs_2 = time.perf_counter()
    best_loss_gs[iterat, 1] = loss
    
    # RL
    print("RL 2")
    start_rl_2 = time.perf_counter()
    for n in range(10):
        pol = PolicySingle(len(delta_taus), sigma_max = sigma_max)
        reinforce_one_set(
                exp, 
                pol, 
                delta_taus=delta_taus, 
                **kwargs
            )
    end_rl_2 = time.perf_counter()
    
    
    ########### three n_step
    loss = 3.
    delta_taus = [.33, .33, .34]
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
    for n in range(10):
        pol = PolicySingle(len(delta_taus), sigma_max = sigma_max)
        reinforce_one_set(
                exp, 
                pol, 
                delta_taus=delta_taus, 
                **kwargs
            )
    end_rl_3 = time.perf_counter()
    
    ############# four n_step
    loss = 3.
    delta_taus = [.25, .25, .25, .25]
    # Grid Search
    print("Grid Search 4")
    start_gs_4 = time.perf_counter()
    for phi_1 in np.linspace(0, 1, 100):
        for phi_2 in np.linspace(0, 1, 100):
            for phi_3 in np.linspace(0, 1, 100):
                for phi_4 in np.linspace(0, 1, 100):
                    exp.reset()
                    exp.run_all([phi_1, phi_2, phi_3, phi_4], delta_taus)
                    if exp.loss() < loss:
                        loss = exp.loss()
                        best_4 = [phi_1, phi_2, phi_3, phi_4]
    end_gs_4 = time.perf_counter()
    best_loss_gs[iterat, 3] = loss
    
    # RL
    print("RL 4")
    start_rl_4 = time.perf_counter()
    for n in range(10):
        pol = PolicySingle(len(delta_taus), sigma_max = sigma_max)
        reinforce_one_set(
                exp, 
                pol, 
                delta_taus=delta_taus, 
                **kwargs
            )
    end_rl_4 = time.perf_counter()

    grid_search_time_ms[iterat] = np.array([end_gs_1 - start_gs_1, end_gs_2 - start_gs_2, end_gs_3 - start_gs_3, end_gs_4 - start_gs_4])
    rl_time_ms[iterat] = np.array([end_rl_1 - start_rl_1, end_rl_2 - start_rl_2, end_rl_3 - start_rl_3, end_rl_4 - start_rl_4])/10
    iterat += 1

(
    np.savez_compressed(
        "../results/grid_search", 
        best_loss_gs=best_loss_gs, 
        grid_search_time_ms=grid_search_time_ms, 
        rl_time_ms=rl_time_ms
    )
)
