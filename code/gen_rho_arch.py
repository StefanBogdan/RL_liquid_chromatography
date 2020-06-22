import numpy as np
import pandas as pd
from chromatography import *
from separation_utility import *
import torch 
import torch.nn.functional as F
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

class RhoReLU(nn.Module):
    def __init__(self, 
            n_steps: int, 
            hidden: int, 
            in_dim: int = 2, 
            sigma_max: float = .3, 
            sigma_min: float = .1
        ) -> None:
        """
        Constructor for PolicyTime torch Module.

        Parameters
        ----------
        n_steps: int
            Number of steps for piece-wise constant solvent strength program.
        hidden: int
            Number of nodes for the hidden layers
        in_dim: int
            length of the encoded analyte set (embedding), it is the input 
            to this network.
        sigma_min: float
            Minimal standard deviation of the solvent strength search space.
            Default value .0. (max value < 1.0)
        sigma_max: float
            Maximal standard deviation of the solvent strength search space.
            Default value .2. (max value is 1.0)
        """
        super().__init__()
        
        self.n_steps = n_steps
        self.hidden = hidden
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        self.sig = nn.Sigmoid()
        self.fc_mu_1 = nn.Linear(in_dim, hidden)
        self.fc_mu_2 = nn.Linear(hidden, n_steps)
        self.fc_sig_1 = nn.Linear(in_dim, hidden)
        self.fc_sig_2 = nn.Linear(hidden, n_steps)
          
    def forward(self, x):
        mu = F.relu(self.fc_mu_1(x))
        sigma = F.relu(self.fc_sig_1(x))
        
        mu = self.sig(self.fc_mu_2(mu)).squeeze(0)
        # limit sigma to be in range (sigma_min; sigma_max)
        sigma = self.sig(self.fc_sig_2(sigma)).squeeze(0) * (self.sigma_max - self.sigma_min) + self.sigma_min
        return mu, sigma

class RhoELU(nn.Module):
    def __init__(self, 
            n_steps: int, 
            hidden: int, 
            in_dim: int = 2, 
            sigma_max: float = .3, 
            sigma_min: float = .1
        ) -> None:
        """
        Constructor for PolicyTime torch Module.

        Parameters
        ----------
        n_steps: int
            Number of steps for piece-wise constant solvent strength program.
        hidden: int
            Number of nodes for the hidden layers
        in_dim: int
            length of the encoded analyte set (embedding), it is the input 
            to this network.
        sigma_min: float
            Minimal standard deviation of the solvent strength search space.
            Default value .0. (max value < 1.0)
        sigma_max: float
            Maximal standard deviation of the solvent strength search space.
            Default value .2. (max value is 1.0)
        """
        super().__init__()
        
        self.n_steps = n_steps
        self.hidden = hidden
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        self.sig = nn.Sigmoid()
        self.fc_mu_1 = nn.Linear(in_dim, hidden)
        self.fc_mu_2 = nn.Linear(hidden, n_steps)
        self.fc_sig_1 = nn.Linear(in_dim, hidden)
        self.fc_sig_2 = nn.Linear(hidden, n_steps)
          
    def forward(self, x):
        mu = F.elu(self.fc_mu_1(x))
        sigma = F.elu(self.fc_sig_1(x))
        
        mu = self.sig(self.fc_mu_2(mu)).squeeze(0)
        # limit sigma to be in range (sigma_min; sigma_max)
        sigma = self.sig(self.fc_sig_2(sigma)).squeeze(0) * (self.sigma_max - self.sigma_min) + self.sigma_min
        return mu, sigma

class RhoTanh(nn.Module):
    def __init__(self, 
            n_steps: int, 
            hidden: int, 
            in_dim: int = 2, 
            sigma_max: float = .3, 
            sigma_min: float = .1
        ) -> None:
        """
        Constructor for PolicyTime torch Module.

        Parameters
        ----------
        n_steps: int
            Number of steps for piece-wise constant solvent strength program.
        hidden: int
            Number of nodes for the hidden layers
        in_dim: int
            length of the encoded analyte set (embedding), it is the input 
            to this network.
        sigma_min: float
            Minimal standard deviation of the solvent strength search space.
            Default value .0. (max value < 1.0)
        sigma_max: float
            Maximal standard deviation of the solvent strength search space.
            Default value .2. (max value is 1.0)
        """
        super().__init__()
        
        self.n_steps = n_steps
        self.hidden = hidden
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        self.sig = nn.Sigmoid()
        self.fc_mu_1 = nn.Linear(in_dim, hidden)
        self.fc_mu_2 = nn.Linear(hidden, n_steps)
        self.fc_sig_1 = nn.Linear(in_dim, hidden)
        self.fc_sig_2 = nn.Linear(hidden, n_steps)
          
    def forward(self, x):
        mu = torch.tanh(self.fc_mu_1(x))
        sigma = torch.tanh(self.fc_sig_1(x))
        
        mu = self.sig(self.fc_mu_2(mu)).squeeze(0)
        # limit sigma to be in range (sigma_min; sigma_max)
        sigma = self.sig(self.fc_sig_2(sigma)).squeeze(0) * (self.sigma_max - self.sigma_min) + self.sigma_min
        return mu, sigma

# Parameters
all_analytes = pd.concat(alists, sort=True).reset_index()[['k0', 'S', 'lnk0']]
activations = [nn.ELU, nn.ReLU, nn.Tanh]
width = [5, 10, 20]
Rhos = [RhoELU, RhoReLU, RhoTanh]
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

losses_rho = np.zeros((N, M, kwargs['num_episodes']))
test_losses_rho = np.zeros((N, M, kwargs['num_episodes']))

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
                PermEqui2_max(2, 5),
                nn.ELU(inplace=True),
                PermEqui2_max(5, 5),
                nn.ELU(inplace=True),
                PermEqui2_max(5, 5),
                nn.ELU(inplace=True),
            ),
            rho = nn.Sequential(
                nn.Linear(5, width[i % 3]),
                activations[i // 3](),
                nn.Linear(width[i % 3], width[i % 3]),
                activations[i // 3](),
                Rhos[i // 3](n_steps=len(delta_taus), hidden=width[i % 3], in_dim=width[i % 3], sigma_max=.3, sigma_min=.01),
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

        losses_rho[i, m] = loss
        test_losses_rho[i, m] = loss_test


np.savez_compressed("../results/general_perf_rho_arch_loss_50_50", losses_50_50=losses_rho)
np.savez_compressed("../results/general_perf_rho_arch_test_losses_50_50", test_losses_50_50=test_losses_rho)