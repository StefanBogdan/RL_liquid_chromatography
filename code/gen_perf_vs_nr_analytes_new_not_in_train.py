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

losses_50_50 = np.zeros((M, N, kwargs['num_episodes']))
test_losses_50_50 = np.zeros((M, N, kwargs['num_episodes']))
def reinforce_gen_1(
        policy: PolicyGeneral, 
        delta_taus: Iterable[float],
        alists: Iterable[pd.DataFrame] = [],
        random_alist: pd.DataFrame = None,
        test_alist: pd.DataFrame = None,
        num_episodes: int = 1000, 
        sample_size: int = 10,
        batch_size : int = 10,
        lr: float = 1., 
        optim = torch.optim.SGD,
        lr_decay_factor: float = 1.,
        lr_milestones: Union[int, Iterable[int]] = 1000,
        rand_prob: float = .2,
        max_rand_analytes: int = 30,
        min_rand_analytes: int = 10,
        print_every: int = 100,
        baseline: float = 0.,
        max_norm: float = None,
        beta: float = .0,
        weights: list = [1., 1.],
        h = .001,
        run_time = 10.,
    ):

    losses_train = []
    perfect_loss = []
    losses_test = [[], [], [], [], []]
    exps = []
    Lengths = [8, 16, 24, 32, 40]

    if len(alists) == 0 and not isinstance(random_alist, pd.DataFrame):
        raise "'alists' and 'random_alist' cannot be empty at the same time!"

    if len(alists) == 0:
        rand_prob = 1.

    # Make ExperimentAnalytes object for the given analyte sets for time saving purpose
    for alist in alists:
        exps.append(ExperimentAnalytes(k0 = alist.k0.values, S = alist.S.values, h=h, run_time=run_time))

    num_exps = len(alists)

    if isinstance(random_alist, pd.DataFrame):
        all_analytes = random_alist[['k0', 'S', 'lnk0']]
    else:
        all_analytes = pd.concat(alists, sort=True)[['k0', 'S', 'lnk0']]

    # Optimizer
    optimizer = optim(policy.parameters(), lr)

    # LR sheduler
    if isinstance(lr_milestones, list) or isinstance(lr_milestones, np.ndarray):
        scheduler = MultiStepLR(optimizer, lr_milestones, gamma=lr_decay_factor)
    else:
        scheduler = StepLR(optimizer, lr_milestones, gamma=lr_decay_factor)

    J_batch = 0

    for n in range(num_episodes):
        # the set to use for the experiment.
        if random() < rand_prob:
            dataframe = all_analytes.sample(randint(min_rand_analytes, max_rand_analytes))
            input_data = torch.tensor(dataframe[['S', 'lnk0']].values, dtype=torch.float32)
            exp = ExperimentAnalytes(k0 = dataframe.k0.values, S = dataframe.S.values, h=h, run_time=run_time)

        else:
            # Choose a random set
            set_index = randint(0, num_exps - 1) 
            exp = exps[set_index]
            input_data = torch.tensor(alists[set_index][['S', 'lnk0']].values, dtype=torch.float32)

        for _l_ in range(5):
            test_dataframe = test_alist.sample(Lengths[_l_])
            test_data = torch.tensor(test_dataframe[['S', 'lnk0']].values, dtype=torch.float32)
            test_exp = ExperimentAnalytes(k0 = test_dataframe.k0.values, S = test_dataframe.S.values, h=h, run_time=run_time)
            mu, _ = policy.forward(test_data)
            test_exp.run_all(mu.data.numpy(), delta_taus)
            losses_test[_l_].append(test_exp.loss(weights) - test_exp.perfect_loss(weights))

        # compute distribution parameters (Normal)
        mu, sigma = policy.forward(input_data)

        # Sample some values from the actions distributions
        programs = sample(mu, sigma, sample_size)
        
        # Fit the sampled data to the constraint [0,1]
        constr_programs = programs.clone()
        constr_programs[constr_programs > 1] = 1
        constr_programs[constr_programs < 0] = 0
        
        J = 0
        expected_loss = 0
        for i in range(sample_size):
            exp.reset()            
            exp.run_all(constr_programs[i].data.numpy(), delta_taus)

            error = exp.loss(weights)
            expected_loss += error
            log_prob_ = log_prob(programs[i], mu, sigma)
            J += (error - baseline) * log_prob_ - beta * torch.exp(log_prob_) * log_prob_
        
        losses_train.append(expected_loss/sample_size - exp.perfect_loss(weights))
        if (n + 1) % print_every == 0:
            print(f"Training Loss: {losses_train[-1]}, epoch: {n+1}/{num_episodes}")

        J_batch += J/sample_size
        if (i + 1) % batch_size == 0:
            J_batch /= batch_size
            optimizer.zero_grad()
            # Calculate gradients
            J_batch.backward()

            if max_norm:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm)

            # Apply gradients
            optimizer.step()

            # learning rate decay
            scheduler.step()

            J_batch = 0
        
    return np.array(losses_train), np.array(losses_test)



delta_taus = np.ones(10) * 1/(10)

for i in range(M):
    alist_train = all_analytes.sample(frac=0.5)
    alist_test = all_analytes.loc[lambda a: ~a.index.isin(alist_train.index.values)]
    print(f"{i}")
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
    
    # Run Exp
    loss, loss_test = reinforce_gen_1(
        alists = [alist_train], 
        test_alist = alist_test,
        policy = pol_50_50, 
        delta_taus = delta_taus,
        min_rand_analytes = 8,
        max_rand_analytes = 40,
        **kwargs
    )


    losses_50_50[i] = loss
    test_losses_50_50[i] = loss_test


np.savez_compressed("../results/general_perf_vs_nr_analytes_losses_50_new", losses_50_50=losses_50_50)
np.savez_compressed("../results/general_perf_vs_nr_analytes_test_losses_50_50_new", test_losses_50_50=test_losses_50_50)
