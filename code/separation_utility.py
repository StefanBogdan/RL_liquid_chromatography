"""
Liquid Chromatography Separation Module
"""
from typing import Tuple, Iterable
import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch import optim, tensor
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from random import randint, random
from chromatography import ExperimentAnalytes


def log_prob(
        value: torch.Tensor, 
        mu: torch.Tensor, 
        sigma: torch.Tensor
    ) -> torch.Tensor:
    """
    Compute Log Probability for the Normal distribution.

    Parameters
    ----------
    value: torch.Tensor
        tensor of sampled values from the Multivariate/Normal distribution.
    mu: torch.Tensor
        tensor of means of Multivariate/Normal distribution.
    sigma: torch.Tensor
        tensor of standars deviations of Multivariate/Normal distribution.
        It is one dimensional tensor because all dimension are not correlated
        (A diagonal matrix Sigma is constructed from the 1-d tensor)

    Returns
    -------
    torch.Tensor
        log probability of the given value being sampled from the
        Multivariate/Normal distribution. (one element tensor)
    """

    if value.numel() == 1:
        return (
            Normal(mu, sigma)
            .log_prob(value)
            )
    else:
        return (
            MultivariateNormal(mu, torch.diag(sigma ** 2))
            .log_prob(value)
            )

def sample(
        mu: torch.Tensor, 
        sigma: torch.Tensor, 
        n_samples: int
    ) -> torch.Tensor:
    """
    Sample datapoints form a Multivariate/Normal distribution
    with given mu and sigmas, with no correlation between 
    dimensions.

    Parameters
    ----------
    mu: tensor.Torch
        tensor of means of the Multivariate/Normal distribution
    sigma: torch.Tensor
        tensor of standars deviations of Multivariate/Normal distribution.
        It is one dimensional tensor because all dimension are not correlated
        (A diagonal matrix Sigma is constructed from the 1-d tensor)
    n_samples: int
        number of datapoints to be sampled from the given distribution.

    Returns
    -------
    torch.Tensor
        A tensor with sampled datapoints.
    """
    if len(mu) == 1:
        return (
            Normal(
                mu, sigma
            )
            .sample((n_samples,1))
        )
    else:
        return (
            MultivariateNormal(
                mu, torch.diag(sigma ** 2)
            )
            .sample((n_samples,))
            )


class Policy(nn.Module):

    def __init__(self, 
            n_steps: int,
            sigma_min: float = .0, 
            sigma_max: float = .2
        ) -> None:
        """
        Constructor for Policy torch Module.

        Parameters
        ----------
        n_steps: int
            Number of steps for piece-wise constant solvent strength program.
        sigma_min: float
            Minimal standard deviation of the solvent strength search space.
            Default value .0. (max value < 1.0)
        sigma_max: float
            Maximal standard deviation of the solvent strength search space.
            Default value .2. (max value is 1.0)
        """


        if n_steps < 1:
            raise ValueError(f"'n_steps' cannot have negative values or zero, given {n_steps}")
        if sigma_min < 0.  or sigma_max < sigma_min or sigma_max > 1. :
            raise ValueError(f"sigmas cannot be negative, sigma_min < sigma_max and maximum value for sigma is 1.0")

        super(Policy, self).__init__()

        # parameters
        self.n_steps = n_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max


        # Define network
        self.sig = nn.Sigmoid()
        self.fc_mu = nn.Linear(1, self.n_steps, bias=False)        
        self.fc_sigma = nn.Linear(1, self.n_steps, bias=False)
        
    
    def forward(self
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute mean and standard deviation for the action space.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Means and standard deviations of the action space.
        """

        out = torch.ones((1,1))
        self.mu = self.sig(self.fc_mu(out)).squeeze()

        # limit sigma to be in range (sigma_min; sigma_max)
        self.sigma = self.sig(self.fc_sigma(out)).squeeze() * (self.sigma_max - self.sigma_min) + self.sigma_min
        
        return self.mu, self.sigma


class PolicyISO(nn.Module):

    def __init__(self, 
            n_steps: int,
            sigma_min: float = .0, 
            sigma_max: float = .1
        ) -> None:
        """
        Constructor for PolicyISO torch Module.

        Parameters
        ----------
        n_steps: int
            Number of steps for piece-wise constant solvent strength program.
        sigma_min: float
            Minimal standard deviation of the solvent strength search space.
            Default value .0. (max value < 1.0)
        sigma_max: float
            Maximal standard deviation of the solvent strength search space.
            Default value .2. (max value is 1.0)
        """


        if n_steps < 1:
            raise ValueError(f"'n_steps' cannot have negative values or zero, given {n_steps}")
        if sigma_min < 0.  or sigma_max < sigma_min or sigma_max > 1. :
            raise ValueError(f"sigmas cannot be negative, sigma_min < sigma_max and maximum value for sigma is 1.0")

        super(PolicyISO, self).__init__()

        # parameters
        self.n_steps = n_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max


        # Define network
        self.sig = nn.Sigmoid()
        self.fc_mu = nn.Linear(1, n_steps, bias=False)        
        self.fc_sigma = nn.Linear(1, n_steps, bias=False)
        
    
    def forward(self,
            up_lim: float,
            low_lim: float
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute mean and standard deviation for the action space.

        Parameters
        ----------
        up_lim: float
            Upper limit for mean.
        low_lim: float
            Lower limit for mean
        
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Means and standard deviations of the action space.
        """
        
        out = torch.ones((1,1))
        self.mu = self.sig(self.fc_mu(out)).squeeze() * (up_lim - low_lim) + low_lim
        # limit sigma to be in range (sigma_min; sigma_max)
        self.sigma = self.sig(self.fc_sigma(out)).squeeze() * (self.sigma_max - self.sigma_min) + self.sigma_min

        # retain grad for both mu and sigma
        self.mu.retain_grad()
        self.sigma.retain_grad()
        
        return self.mu, self.sigma


class PolicyTime(nn.Module):

    def __init__(self, 
            n_steps: int,
            sigma_min: float = .0, 
            sigma_max: float = .1
        ) -> None:
        """
        Constructor for PolicyTime torch Module.

        Parameters
        ----------
        n_steps: int
            Number of steps for piece-wise constant solvent strength program.
        sigma_min: float
            Minimal standard deviation of the solvent strength search space.
            Default value .0. (max value < 1.0)
        sigma_max: float
            Maximal standard deviation of the solvent strength search space.
            Default value .2. (max value is 1.0)
        """


        if n_steps < 1:
            raise ValueError(f"'n_steps' cannot have negative values or zero, given {n_steps}")
        if sigma_min < 0.  or sigma_max < sigma_min or sigma_max > 1. :
            raise ValueError(f"sigmas cannot be negative, sigma_min < sigma_max and maximum value for sigma is 1.0")

        super(PolicyTime, self).__init__()

        # parameters
        self.n_steps = n_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        
        # Define network
        self.fc_mu = nn.Linear(1, n_steps * 2 - 1, bias=False)
        self.sig = nn.Sigmoid()
        self.fc_sigma = nn.Linear(1, n_steps * 2 - 1, bias=False)
        self.softplus = nn.Softplus()
        
        
    def forward(self
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute mean and standard deviation for the action space.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Means and standard deviations of the action space.
        """
        out = torch.ones((1,1))
        self.mu = self.sig(self.fc_mu(out)).squeeze() 
        self.sigma = self.softplus(self.fc_sigma(out)).squeeze() * (self.sigma_max - self.sigma_min) + self.sigma_min

        return self.mu, self.sigma



def reinforce_one_set(
        exp: ExperimentAnalytes, 
        policy: Policy, 
        delta_taus: Iterable[float], 
        num_episodes = 1000, 
        batch_size = 10, 
        lr: float = 1., 
        optim: = torch.optim.SGD,
        print_every: int = 100,
        lr_decay = None,
        weights: list = [1., 1.],
        baseline: float = 0.,
        max_norm: float = None,
        beta:float = .0
    ):
    """
    Run Reinforcement Learning for a single set learning.

    exp: ExperimentAnalytes
        The experiment that is used to be optimized. 
    policy: Policy
        The policy that learns the optimal values for the solvent
        strength program.
    delta_taus: Iterable[float]
        Iterable list with the points of solvent strength change.
        MUST be the same length as policy.n_steps
    num_episodes = 1000
        Number of learning steps.
    batch_size = 10
        Number of samples taken from the action distribution to perform 
        Expected loss for the distribution of actions.
    lr = 1.
        Learning rate.
    optim = torch.optim.SGD
        Optimizer that performs weight update using gradients.
    print_every = 100,
        Number of episodes to print the average loss on.
    lr_decay = None
        Learning rate scheduler.
    weights = [1., 1.]
        Weigths of the errors to consider, first one is for the Placement Error,
        second one is for Overlap Error, By default both have the same wights.
    baseline = 0.
        Baseline value for the REINFORCE algorithm.
    max_norm = None
        Maximal value for the Neural Network Norm2.
    beta = .0
        Entropy Regularization term, is used for more exploration.
        By defauld is disabled.
    """

    losses = []
    loss = 0
    loss_best = 2 
    epoch_mus = []
    epoch_sigmas = []
    epoch_n_taus = []
    samples = []
    mu_grads = []
    sigma_grads = []

    
    for n in range(num_episodes):           
        
        # Optimizer
        optimizer = optim(policy.parameters(), lr)
        
        # compute distribution parameters (Normal)
        mu, sigma = policy.forward()

        # Save parameters
        epoch_mus.append(policy.mu.detach().numpy())
        epoch_sigmas.append(policy.sigma.detach().numpy())

        # Sample some values from the actions distributions
        programs = sample(mu, sigma, batch_size)
        
        # Fit the sampled data to the constraint [0,1]
        constr_programs = programs.clone()
        constr_programs[constr_programs > 1] = 1
        constr_programs[constr_programs < 0] = 0

        samples.append(constr_programs)
        
        J = 0
        for i in range(batch_size):
            exp.reset()            
            exp.run_all(constr_programs[i].data.numpy(), delta_taus)
            epoch_n_taus.append(len(exp.delta_taus))

            error = exp.loss(weights)
            loss += error
            log_prob_ = log_prob(programs[i], policy.mu, policy.sigma)
            J += (error - baseline) * log_prob_ - beta * torch.exp(log_prob_) * log_prob_
            if error < loss_best:
                loss_best = error
                best_program = constr_programs[i]
                        
        if (n + 1) % print_every == 0:
            losses.append(loss/(batch_size * print_every))
            print(f"Loss: {losses[-1]}, epoch: {n+1}/{num_episodes}")
            #print(f"Means: {pol.mu.data}, Sigmas: {pol.sigma.data}")
            loss = 0

        J /= batch_size  
        optimizer.zero_grad()
        
        # Calculate gradients
        J.backward()

        if max_norm:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm)

        # gradients with respect tu mu and sigma
        mu_grads.append(policy.mu.grad)
        sigma_grads.append(policy.sigma.grad)

        # Apply gradients
        optimizer.step()

        # learning rate decay
        if lr_decay:
            lr = lr_decay(lr, (n + 1), num_episodes) 
        
    return np.array(losses), best_program,  np.array(epoch_mus), np.array(epoch_sigmas), \
        np.array(epoch_n_taus), np.vstack(samples), np.vstack(mu_grads), np.vstack(sigma_grads)


def reinforce_delta_tau(
        exp, 
        policy,
        num_episodes = 1000, 
        batch_size = 10, 
        lr = 1., 
        optim = torch.optim.SGD,
        print_every = 100,
        lr_decay = None,
        weights = [1., 1.],
        baseline = 0.,
        max_norm = 1.
    ):

    losses = []
    loss = 0
    loss_best = 2
    epoch_n_par = []
    epoch_mus = []
    epoch_sigmas = []
    
    for n in range(num_episodes):          
        
        # Optimizer
        optimizer = optim(policy.parameters(), lr)
        
        # compute distribution parameters (Normal)
        mu, sigma = policy.forward()
        
        # Save parameters
        epoch_mus.append(list(policy.mu))
        epoch_sigmas.append(list(policy.sigma))
        
        # Sample some values from the actions distributions
        values = sample(mu, sigma, batch_size)

        # Add tau for the last solvent strength (
        # runs until an analyte reaches th end of the columns
        new_values = np.ones((values.shape[0], values.shape[1] + 1)) * 1e5
        new_values[:, :-1] = values
        
        # Fit the sampled data to the constraint [0,1] or (0, +inf)
        programs, delta_taus = np.split(new_values, 2, 1)
        programs[programs > 1] = 1
        programs[programs < 0] = 0
        delta_taus[delta_taus < 0] = 1e-7
        J = 0
        
        for i in range(batch_size):
            exp.reset()            
            exp.run_all(programs[i].data, delta_taus[i])
            epoch_n_par.append(len(exp.delta_taus))            
            
            error = exp.loss(weights)
            loss += error
            J += (error - baseline) * log_prob(values[i], policy.mu, policy.sigma)
            if error < loss_best:
                loss_best = error
                best_program = [programs[i], delta_taus[i]]
                        
        if (n + 1) % print_every == 0:
            losses.append(loss/(batch_size * print_every))
            print(f"Loss: {losses[-1]}, epoch: {n+1}/{num_episodes}")
            #print(f"Means: {pol.mu.data}, Sigmas: {pol.sigma.data}")
            loss = 0
        J /= batch_size  
        optimizer.zero_grad()
        
        # Calculate gradients
        J.backward()

        if max_norm:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm)
                
        # Apply gradients
        optimizer.step()

        # learning rate decay
        if lr_decay:
            lr = lr_decay(lr, (n + 1), num_episodes) 
        
    return losses, best_program, np.array(epoch_mus), np.array(epoch_sigmas), np.array(epoch_n_par)


def reinforce_best_iso(
        exp, 
        policy, 
        delta_taus, 
        num_episodes = 1000, 
        batch_size = 10, 
        lr = 1., 
        optim = torch.optim.SGD,
        print_every = 100,
        lr_decay = None,
        weights = [1., 1.],
        baseline = 0.,
        max_norm = None,
        beta = .0,
        lim = 0.1
    ):

    losses = []
    loss = 3
    loss_best = 2 
    epoch_mus = []
    epoch_sigmas = []
    epoch_n_taus = []
    samples = []
    mu_grads = []
    sigma_grads = []


    for phi in np.linspace(0, 1, 1000):
        exp.reset()
        exp.step(phi, 1.)
        if exp.loss() < loss:
            phi_iso = phi
            loss = exp.loss()

    loss = 0

    low_lim = max(0, phi_iso - lim)
    up_lim = min(1, phi_iso + lim)
    print(phi_iso, low_lim, up_lim)
    
    for n in range(num_episodes):           
        
        # Optimizer
        optimizer = optim(policy.parameters(), lr)
        
        # compute distribution parameters (Normal)
        mu, sigma = policy.forward(up_lim, low_lim)

        # Save parameters
        epoch_mus.append(policy.mu.detach().numpy())
        epoch_sigmas.append(policy.sigma.detach().numpy())

        # Sample some values from the actions distributions
        programs = sample(mu, sigma, batch_size)
        
        # Fit the sampled data to the constraint [0,1]
        constr_programs = programs.clone()
        constr_programs[constr_programs > up_lim] = up_lim
        constr_programs[constr_programs < low_lim] = low_lim

        samples.append(constr_programs)
        
        J = 0
        for i in range(batch_size):
            exp.reset()            
            exp.run_all(constr_programs[i].data.numpy(), delta_taus)
            epoch_n_taus.append(len(exp.delta_taus))

            error = exp.loss(weights)
            loss += error
            log_prob_ = log_prob(programs[i], policy.mu, policy.sigma)
            J += (error - baseline) * log_prob_ - beta * torch.exp(log_prob_) * log_prob_
            if error < loss_best:
                loss_best = error
                best_program = constr_programs[i]
                        
        if (n + 1) % print_every == 0:
            losses.append(loss/(batch_size * print_every))
            print(f"Loss: {losses[-1]}, epoch: {n+1}/{num_episodes}")
            #print(f"Means: {pol.mu.data}, Sigmas: {pol.sigma.data}")
            loss = 0

        J /= batch_size  
        optimizer.zero_grad()
        
        # Calculate gradients
        J.backward()

        if max_norm:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm)

        # gradients with respect tu mu and sigma
        mu_grads.append(policy.mu.grad)
        sigma_grads.append(policy.sigma.grad)

        # Apply gradients
        optimizer.step()

        # learning rate decay
        if lr_decay:
            lr = lr_decay(lr, (n + 1), num_episodes) 
        
    return np.array(losses), best_program,  np.array(epoch_mus), np.array(epoch_sigmas), \
        np.array(epoch_n_taus), np.vstack(samples), np.vstack(mu_grads), np.vstack(sigma_grads)




def reinforce_best_sample(
        exp, 
        policy, 
        delta_taus, 
        num_episodes = 1000, 
        batch_size = 10, 
        lr = 1., 
        optim = torch.optim.SGD,
        print_every = 100,
        lr_decay = None,
        weights = [1., 1.],
        baseline = 0.,
        max_norm = None,
        beta = .0
    ):

    losses = []
    loss = 0
    loss_best = 2 
    epoch_mus = []
    epoch_sigmas = []
    epoch_n_taus = []
    samples = []
    mu_grads = []
    sigma_grads = []

    
    for n in range(num_episodes):           
        
        # Optimizer
        optimizer = optim(policy.parameters(), lr)
        
        # compute distribution parameters (Normal)
        mu, sigma = policy.forward()

        # Save parameters
        epoch_mus.append(policy.mu.detach().numpy())
        epoch_sigmas.append(policy.sigma.detach().numpy())

        # Sample some values from the actions distributions
        programs = sample(mu, sigma, batch_size)
        
        # Fit the sampled data to the constraint [0,1]
        constr_programs = programs.clone()
        constr_programs[constr_programs > 1] = 1
        constr_programs[constr_programs < 0] = 0

        samples.append(constr_programs)
        
        J = 0
        best = 3.
        for i in range(batch_size):
            exp.reset()            
            exp.run_all(constr_programs[i].data.numpy(), delta_taus)
            epoch_n_taus.append(len(exp.delta_taus))

            error = exp.loss(weights)
            if error < best:
                log_prob_ = log_prob(programs[i], policy.mu, policy.sigma)
                J = -(error - baseline) * log_prob_ - beta * torch.exp(log_prob_) * log_prob_
                if error < loss_best:
                    loss_best = error
                    best_program = constr_programs[i]
                best = error.copy()
        loss += best
            
                        
        if (n + 1) % print_every == 0:
            losses.append(loss/(print_every))
            print(f"Loss: {losses[-1]}, epoch: {n+1}/{num_episodes}")
            #print(f"Means: {pol.mu.data}, Sigmas: {pol.sigma.data}")
            loss = 0
 
        optimizer.zero_grad()
        
        # Calculate gradients
        J.backward()

        if max_norm:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm)

        # gradients with respect tu mu and sigma
        mu_grads.append(policy.mu.grad)
        sigma_grads.append(policy.sigma.grad)

        # Apply gradients
        optimizer.step()

        # learning rate decay
        if lr_decay:
            lr = lr_decay(lr, (n + 1), num_episodes) 
        
    return np.array(losses), best_program,  np.array(epoch_mus), np.array(epoch_sigmas), \
        np.array(epoch_n_taus), np.vstack(samples), np.vstack(mu_grads), np.vstack(sigma_grads)

def reinforce_gen(
        alists, 
        policy, 
        delta_taus, 
        num_episodes = 1000, 
        batch_size = 10, 
        lr = 1., 
        optim = torch.optim.SGD,
        print_every = 100,
        lr_decay = None,
        weights = [1., 1.],
        baseline = 0.,
        max_norm = None,
        beta = .0,
        rand_prob = .2,
        max_rand_analytes = 30,
        min_rand_analytes = 10
    ):

    losses = []
    loss = 0
    samples = []
    exps = []
    mus, sigmas = [], []
    grads_mu_1 = []
    grads_sig_1 = []
    grads_mu_2 = []
    grads_sig_2 = []

    for alist in alists:
        exps.append(ExperimentAnalytes(k0 = alist.k0.values, S = alist.S.values, h=0.001, run_time=10.0))

    num_exps = len(alists)

    all_analytes = pd.concat(alists, sort=True)[['k0', 'S', 'lnk0']]

    
    for n in range(num_episodes):  
        if random() < rand_prob:
            dataframe = all_analytes.sample(randint(min_rand_analytes, max_rand_analytes))
            input_data = torch.tensor(dataframe[['S', 'lnk0']].values, dtype=torch.float32)
            exp = ExperimentAnalytes(k0 = dataframe.k0.values, S = dataframe.S.values, h=0.001, run_time=10.0)

        else:
            # Choose a random set
            set_index = randint(0, num_exps - 1) 
            #print(set_index)
            exp = exps[set_index]
            input_data = torch.tensor(alists[set_index][['S', 'lnk0']].values, dtype=torch.float32)
            #print(input_data)

        # Optimizer
        optimizer = optim(policy.parameters(), lr)
        
        # compute distribution parameters (Normal)
        mu, sigma = policy.forward(input_data)
        #mus.append(mu.detach().numpy())
        #sigmas.append(sigma.detach().numpy())
        #print("mu:", mu, "sigma:", sigma)


        # Sample some values from the actions distributions
        programs = sample(mu, sigma, batch_size)
        
        # Fit the sampled data to the constraint [0,1]
        constr_programs = programs.clone()
        constr_programs[constr_programs > 1] = 1
        constr_programs[constr_programs < 0] = 0

        #samples.append(constr_programs)
        
        J = 0
        for i in range(batch_size):
            exp.reset()            
            exp.run_all(constr_programs[i].data.numpy(), delta_taus)

            error = exp.loss(weights)
            loss += error
            log_prob_ = log_prob(programs[i], mu, sigma)
            J += (error - baseline) * log_prob_ - beta * torch.exp(log_prob_) * log_prob_
                        
        if (n + 1) % print_every == 0:
            losses.append(loss/(batch_size * print_every))
            print(f"Loss: {losses[-1]}, epoch: {n+1}/{num_episodes}")
            #print(f"Means: {pol.mu.data}, Sigmas: {pol.sigma.data}")
            loss = 0

        J /= batch_size
        optimizer.zero_grad()
        # Calculate gradients
        J.backward()
        #grads_mu_1.append(policy.fc_mu_1.weight.grad.clone())
        #grads_sig_1.append(policy.fc_sig_1.weight.grad.clone())
        #grads_mu_2.append(policy.fc_mu_2.weight.grad.clone())
        #grads_sig_2.append(policy.fc_sig_2.weight.grad.clone())


        if max_norm:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm)

        # Apply gradients
        optimizer.step()

        # learning rate decay
        if lr_decay:
            lr = lr_decay(lr, (n + 1), num_episodes) 
        
    return np.array(losses)#, np.vstack(samples), np.vstack(mus), np.vstack(sigmas), grads_mu_1, grads_sig_1, grads_sig_1, grads_sig_2