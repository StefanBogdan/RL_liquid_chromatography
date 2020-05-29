"""
Liquid Chromatography Separation Module
"""
from typing import Tuple, Iterable, Union
import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch import optim, tensor
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from random import randint, random
from chromatography import ExperimentAnalytes   
from copy import deepcopy


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


class PolicySingle(nn.Module):

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

        super(PolicySingle, self).__init__()

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
        mu = self.sig(self.fc_mu(out)).squeeze()

        # limit sigma to be in range (sigma_min; sigma_max)
        sigma = self.sig(self.fc_sigma(out)).squeeze() * (self.sigma_max - self.sigma_min) + self.sigma_min
        
        return mu, sigma


class PolicySingleISO(nn.Module):

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

        super(PolicySingleISO, self).__init__()

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
        mu = self.sig(self.fc_mu(out)).squeeze() * (up_lim - low_lim) + low_lim
        # limit sigma to be in range (sigma_min; sigma_max)
        sigma = self.sig(self.fc_sigma(out)).squeeze() * (self.sigma_max - self.sigma_min) + self.sigma_min
        
        return mu, sigma


class PolicySingleTime(nn.Module):

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

        super(PolicySingleTime, self).__init__()

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
        mu = self.sig(self.fc_mu(out)).squeeze() 
        sigma = self.softplus(self.fc_sigma(out)).squeeze() * (self.sigma_max - self.sigma_min) + self.sigma_min

        return mu, sigma

###########################################################################
############# Some extra Modules for the Generalized Policy ###############
###########################################################################

class Rho(nn.Module):
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
        
        mu = self.sig(self.fc_mu_2(mu)).squeeze()
        # limit sigma to be in range (sigma_min; sigma_max)
        sigma = self.sig(self.fc_sig_2(sigma)).squeeze() * (self.sigma_max - self.sigma_min) + self.sigma_min
        return mu, sigma
 

class RhoTime(nn.Module):
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
        self.fc_mu_2 = nn.Linear(hidden, 2 * n_steps - 1)
        self.fc_sig_1 = nn.Linear(in_dim, hidden)
        self.fc_sig_2 = nn.Linear(hidden, 2 * n_steps - 1)
          
    def forward(self, x):
        mu = F.relu(self.fc_mu_1(x))
        sigma = F.relu(self.fc_sig_1(x))
        
        mu = self.sig(self.fc_mu_2(mu)).squeeze()
        # limit sigma to be in range (sigma_min; sigma_max)
        sigma = self.sig(self.fc_sig_2(sigma)).squeeze() * (self.sigma_max - self.sigma_min) + self.sigma_min
        return mu, sigma
    
class PermEqui1_max(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PermEqui1_max, self).__init__()
        self.Gamma = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        xm, _ = x.max(0, keepdim=True)
        x = self.Gamma(x-xm)
        return x

class PermEqui2_max(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PermEqui2_max, self).__init__()
        self.Gamma = nn.Linear(in_dim, out_dim)
        self.Lambda = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        xm, _ = x.max(0, keepdim=True)
        xm = self.Lambda(xm) 
        x = self.Gamma(x)
        x = x - xm
        return x

class PermEqui1_mean(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PermEqui1_mean, self).__init__()
        self.Gamma = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        xm = x.mean(0, keepdim=True)
        x = self.Gamma(x-xm)
        return x

class PermEqui2_mean(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PermEqui2_mean, self).__init__()
        self.Gamma = nn.Linear(in_dim, out_dim)
        self.Lambda = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        xm = x.mean(0, keepdim=True)
        xm = self.Lambda(xm) 
        x = self.Gamma(x)
        x = x - xm
        return x

###########################################################################


class PolicyGeneral(nn.Module):
    def __init__(self, 
            phi: nn.Module,
            rho: nn.Module
        ) -> None:
        """
        Constructor for PolicyTime torch Module.

        Parameters
        ----------
        phi: nn.Module
            The network that encodes the analyte set to a single 
            vector (embedding)
        rho: nn.Module
            The network that outputs the programe for separation
            returns mean and standard deviation of the action space

        Ex:
        For a 4 step solvent gradient programe the generalized policy 
        with 3 elements embedding for the analyte set and intermediate
        layers of 5 neurons.
        policy = PolicyGeneral(
            phi = nn.Sequential(
                PermEqui1_max(2, 5),
                nn.ELU(inplace=True),
                PermEqui1_max(5, 5),
                nn.ELU(inplace=True),
                PermEqui1_max(5, 3),
                nn.ELU(inplace=True),
            ),
            rho = Rho(4, 5, 3, .3, .05)
        )
        """
        super().__init__()

        self.phi = phi
        self.rho = rho
        
    def forward(self, x):
        phi_output = self.phi(x)
        sum_output = phi_output.sum(0, keepdim=True)
        mu, sigma = self.rho(sum_output)
        return mu, sigma


################################################################
########################## REINFORCE ###########################
################################################################


def reinforce_one_set(
        exp: ExperimentAnalytes, 
        policy: PolicySingle, 
        delta_taus: Iterable[float], 
        num_episodes: int = 1000, 
        sample_size: int = 10, 
        lr: float = 1., 
        optim = torch.optim.SGD,
        lr_decay_factor: float = 1.,
        lr_milestones: Union[int, Iterable[int]] = 1000,
        print_every: int = 100,
        baseline: float = 0.,
        max_norm: float = None,
        beta:float = .0,
        weights: list = [1., 1.],
    ):
    """
    Run Reinforcement Learning for a single set learning.

    exp: ExperimentAnalytes
        The experiment that is used to be optimized. 
    policy: PolicySingle
        The policy that learns the optimal values for the solvent
        strength program.
    delta_taus: Iterable[float]
        Iterable list with the points of solvent strength change.
        MUST be the same length as policy.n_steps
    num_episodes = 1000
        Number of learning steps.
    sample_size = 10
        Number of samples taken from the action distribution to perform 
        Expected loss for the distribution of actions.
    lr = 1.
        Learning rate.
    optim = torch.optim.SGD
        Optimizer that performs weight update using gradients.
        By defauld is Stochastic Gradient Descent.
    lr_decay_factor: float
        Learning rate decay factor used for the LRScheduler.
        lr is updated according to lr = lr ** lr_decay_factor.
    lr_milestones: Union[int, Iterable[int]]
        Milestone episode/s to update the learning rate.
        If it is int StepLR is used where lr is changed every lr_milestones.
        If it is a list of ints then at that specific episode the lr
        will be changed.
    print_every = 100,
        Number of episodes to print the average loss on.
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

    Returns
    -------
    (losses, best_program, mus, sigmas)
    losses: np.ndarray
        Expected loss of the action distribution over the whole learning
        process.
    best_program: np.ndarray
        list of the phis that had the lowest loss (based from the samples).
        NOTE: It might not be the global minima of the loss filed because
        the samples are not drawn from the whole loss space.
    mus: np.ndarray
        Mus change over the learning process. its shape is (num_episodes, policy.n_split)
    sigmas: np.ndarray
        Sigmas change over the learning process. its shape is (num_episodes, policy.n_split)
    """

    losses = []
    loss_best = 2 
    epoch_mus = []
    epoch_sigmas = []

    # Optimizer
    optimizer = optim(policy.parameters(), lr)
    # LR sheduler
    if isinstance(lr_milestones, list) or isinstance(lr_milestones, np.ndarray):
        scheduler = MultiStepLR(optimizer, lr_milestones, gamma=lr_decay_factor)
    else:
        scheduler = StepLR(optimizer, lr_milestones, gamma=lr_decay_factor)

    
    for n in range(num_episodes):           
        
        # compute distribution parameters (Normal)
        mu, sigma = policy.forward()

        # Save parameters
        epoch_mus.append(mu.detach().numpy())
        epoch_sigmas.append(sigma.detach().numpy())

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
            if error < loss_best:
                loss_best = error
                best_program = constr_programs[i]
        losses.append(expected_loss/sample_size)
                        
        if (n + 1) % print_every == 0:
            print(f"Loss: {losses[-1]}, epoch: {n+1}/{num_episodes}")

        J /= sample_size  
        optimizer.zero_grad()
        
        # Calculate gradients
        J.backward()

        if max_norm:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm)

        # Apply gradients
        optimizer.step()

        # Adjust learning rate
        scheduler.step()
        
    return np.array(losses), best_program.numpy(),  np.array(epoch_mus), np.array(epoch_sigmas)


def reinforce_delta_tau(
        exp: ExperimentAnalytes, 
        policy: PolicySingleTime, 
        delta_taus: Iterable[float], 
        num_episodes: int = 1000, 
        sample_size: int = 10, 
        lr: float = 1., 
        optim = torch.optim.SGD,
        lr_decay_factor: float = 1.,
        lr_milestones: Union[int, Iterable[int]] = 1000,
        print_every: int = 100,
        baseline: float = 0.,
        max_norm: float = None,
        beta:float = .0,
        weights: list = [1., 1.]
    ):
    """
    Run Reinforcement Learning for a single set learning.

    exp: ExperimentAnalytes
        The experiment that is used to be optimized. 
    policy: PolicySingleTime
        The policy that learns the optimal values for the solvent
        strength program.
    delta_taus: Iterable[float]
        Iterable list with the points of solvent strength change.
        MUST be the same length as policy.n_steps
    num_episodes = 1000
        Number of learning steps.
    sample_size = 10
        Number of samples taken from the action distribution to perform 
        Expected loss for the distribution of actions.
    lr = 1.
        Learning rate.
    optim = torch.optim.SGD
        Optimizer that performs weight update using gradients.
        By defauld is Stochastic Gradient Descent.
    lr_decay_factor: float
        Learning rate decay factor used for the LRScheduler.
        lr is updated according to lr = lr ** lr_decay_factor.
    lr_milestones: Union[int, Iterable[int]]
        Milestone episode/s to update the learning rate.
        If it is int StepLR is used where lr is changed every lr_milestones.
        If it is a list of ints then at that specific episode the lr
        will be changed.
    print_every = 100,
        Number of episodes to print the average loss on.
    weights = [1., 1.]
        Weigths of the errors to consider, first one is for the Placement Error,
        second one is for Overlap Error, By default both have the same wights.
    baseline = 0.
        Baseline value for the REINFORCE algorithm.
    max_norm = None
        Maximal value for the Neural Network Norm2.

    Returns
    -------
    (losses, best_program, mus, sigmas)
    losses: np.ndarray
        Expected loss of the action distribution over the whole learning
        process.
    best_program: list
        A list of 2 np.ndarrays first one is the phis and second is delta taus.
        list of the phis and delta taus that had the lowest loss 
        (based from the samples).
        NOTE: It might not be the global minima of the loss filed because
        the samples are not drawn from the whole loss space.
    mus: np.ndarray
        Mus change over the learning process. its shape is (num_episodes, policy.n_split)
    sigmas: np.ndarray
        Sigmas change over the learning process. its shape is (num_episodes, policy.n_split)
    """


    losses = []
    loss_best = 2 
    epoch_mus = []
    epoch_sigmas = []

    # Optimizer
    optimizer = optim(policy.parameters(), lr)
    # LR sheduler
    if isinstance(lr_milestones, list) or isinstance(lr_milestones, np.ndarray):
        scheduler = MultiStepLR(optimizer, lr_milestones, gamma=lr_decay_factor)
    else:
        scheduler = StepLR(optimizer, lr_milestones, gamma=lr_decay_factor)
    
    for n in range(num_episodes):                  
        # compute distribution parameters (Normal)
        mu, sigma = policy.forward()
        
        # Save parameters
        epoch_mus.append(mu.detach().numpy())
        epoch_sigmas.append(sigma.detach().numpy())
        
        # Sample some values from the actions distributions
        values = sample(mu, sigma, sample_size)

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
        expected_loss = 0
        for i in range(sample_size):
            exp.reset()            
            exp.run_all(programs[i].data, delta_taus[i])        
            
            error = exp.loss(weights)
            expected_loss += error
            J += (error - baseline) * log_prob(values[i], mu, sigma)
            if error < loss_best:
                loss_best = error
                best_program = [programs[i], delta_taus[i]]

        losses.append(expected_loss/sample_size)
                        
        if (n + 1) % print_every == 0:
            print(f"Loss: {losses[-1]}, epoch: {n+1}/{num_episodes}")

        J /= sample_size  
        optimizer.zero_grad()
        
        # Calculate gradients
        J.backward()

        if max_norm:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm)
                
        # Apply gradients
        optimizer.step()

        # learning rate update
        scheduler.step()
        
    return losses, best_program, np.array(epoch_mus), np.array(epoch_sigmas)


def reinforce_best_iso(
        exp: ExperimentAnalytes, 
        policy: PolicySingleISO, 
        delta_taus: Iterable[float], 
        num_episodes: int = 1000, 
        sample_size: int = 10, 
        lr: float = 1., 
        optim = torch.optim.SGD,
        lr_decay_factor: float = 1.,
        lr_milestones: Union[int, Iterable[int]] = 1000,
        print_every: int = 100,
        baseline: float = 0.,
        max_norm: float = None,
        beta:float = .0,
        lim: float = 0.1,
        weights: list = [1., 1.]
    ):
    """
    Run Reinforcement Learning for a single set learning.

    exp: ExperimentAnalytes
        The experiment that is used to be optimized. 
    policy: PolicySingleISO
        The policy that learns the optimal values for the solvent
        strength program.
    delta_taus: Iterable[float]
        Iterable list with the points of solvent strength change.
        MUST be the same length as policy.n_steps
    num_episodes = 1000
        Number of learning steps.
    sample_size = 10
        Number of samples taken from the action distribution to perform 
        Expected loss for the distribution of actions.
    lr = 1.
        Learning rate.
    optim = torch.optim.SGD
        Optimizer that performs weight update using gradients.
        By defauld is Stochastic Gradient Descent.
    lr_decay_factor: float
        Learning rate decay factor used for the LRScheduler.
        lr is updated according to lr = lr ** lr_decay_factor.
    lr_milestones: Union[int, Iterable[int]]
        Milestone episode/s to update the learning rate.
        If it is int StepLR is used where lr is changed every lr_milestones.
        If it is a list of ints then at that specific episode the lr
        will be changed.
    print_every = 100,
        Number of episodes to print the average loss on.
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
    lim: float
        The limit of the box around the ISO solution, i.e.
        Search space = ISO_solution +- lim for every new phi dimension.

    Returns
    -------
    (losses, best_program, mus, sigmas)
    losses: np.ndarray
        Expected loss of the action distribution over the whole learning
        process.
    best_program: np.ndarray
        list of the phis that had the lowest loss (based from the samples).
        NOTE: It might not be the global minima of the loss filed because
        the samples are not drawn from the whole loss space.
    mus: np.ndarray
        Mus change over the learning process. its shape is (num_episodes, policy.n_split)
    sigmas: np.ndarray
        Sigmas change over the learning process. its shape is (num_episodes, policy.n_split)
    """

    losses = []
    expected_loss = 3
    loss_best = 2 
    epoch_mus = []
    epoch_sigmas = []


    for phi in np.linspace(0, 1, 1000):
        exp.reset()
        exp.step(phi, 1.)
        if exp.loss() < expected_loss:
            phi_iso = phi
            expected_loss    = exp.loss()

    low_lim = max(0, phi_iso - lim)
    up_lim = min(1, phi_iso + lim)
    print(low_lim, phi_iso, up_lim)

    # Optimizer
    optimizer = optim(policy.parameters(), lr)
    # LR sheduler
    if isinstance(lr_milestones, list) or isinstance(lr_milestones, np.ndarray):
        scheduler = MultiStepLR(optimizer, lr_milestones, gamma=lr_decay_factor)
    else:
        scheduler = StepLR(optimizer, lr_milestones, gamma=lr_decay_factor)
    
    for n in range(num_episodes):           
        # compute distribution parameters (Normal)
        mu, sigma = policy.forward(up_lim, low_lim)

        # Save parameters
        epoch_mus.append(mu.detach().numpy())
        epoch_sigmas.append(sigma.detach().numpy())

        # Sample some values from the actions distributions
        programs = sample(mu, sigma, sample_size)
        
        # Fit the sampled data to the constraint [0,1]
        constr_programs = programs.clone()
        constr_programs[constr_programs > up_lim] = up_lim
        constr_programs[constr_programs < low_lim] = low_lim
        
        J = 0
        expected_loss = 0
        for i in range(sample_size):
            exp.reset()            
            exp.run_all(constr_programs[i].data.numpy(), delta_taus)

            error = exp.loss(weights)
            expected_loss += error
            log_prob_ = log_prob(programs[i], mu, sigma)
            J += (error - baseline) * log_prob_ - beta * torch.exp(log_prob_) * log_prob_
            if error < loss_best:
                loss_best = error
                best_program = constr_programs[i]
        
        losses.append(expected_loss/sample_size)

        if (n + 1) % print_every == 0:
            print(f"Loss: {losses[-1]}, epoch: {n+1}/{num_episodes}")

        J /= sample_size  
        optimizer.zero_grad()
        
        # Calculate gradients
        J.backward()

        if max_norm:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm)

        # Apply gradients
        optimizer.step()

        # Adjust learning rate
        scheduler.step()
        
    return np.array(losses), best_program,  np.array(epoch_mus), np.array(epoch_sigmas)


def reinforce_gen(
        alists: Iterable[pd.DataFrame],
        policy: PolicyGeneral, 
        delta_taus: Iterable[float], 
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
        weights: list = [1., 1.]
    ):
    """
    Run Reinforcement Learning for a single set learning.

    alists: Iterable[pd.DataFrame]
        A list with pd.Dataframes for each dataset used to train on. 
    policy: PolicyGeneral
        The policy that learns the optimal values for the solvent
        strength program.
    delta_taus: Iterable[float]
        Iterable list with the points of solvent strength change.
        MUST be the same length as policy.n_steps
    num_episodes = 1000
        Number of learning steps.
    sample_size = 10
        Number of samples taken from the action distribution to perform 
        Expected loss for the distribution of actions.
    batch_size:
        Number of experiments to run in order to aproximate the true gradient.
    lr = 1.
        Learning rate.
    optim = torch.optim.SGD
        Optimizer that performs weight update using gradients.
        By defauld is Stochastic Gradient Descent.
    lr_decay_factor: float
        Learning rate decay factor used for the LRScheduler.
        lr is updated according to lr = lr ** lr_decay_factor.
    lr_milestones: Union[int, Iterable[int]]
        Milestone episode/s to update the learning rate.
        If it is int StepLR is used where lr is changed every lr_milestones.
        If it is a list of ints then at that specific episode the lr
        will be changed.
    rand_prob: float = .2
        The probability to draw a random subset from all the analytes.
        1 - rand_prob is the probability to use a "real" set (provided in
        alists).
    max_rand_analytes: int = 30
        The maximum number of analytes in the randomly drawn set.
    min_rand_analytes: int = 10
        The minimum number of analytes in the randomly drawn set.
    print_every = 100,
        Number of episodes to print the average loss on.
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
    Returns
    -------
    (losses, best_program, mus, sigmas)
    losses: np.ndarray
        Expected loss of the action distribution over the whole learning
        process.
    """

    losses = []
    perfect_loss = []
    exps = []

    # Make ExperimentAnalytes object for the given analyte sets for time saving purpose
    for alist in alists:
        exps.append(ExperimentAnalytes(k0 = alist.k0.values, S = alist.S.values, h=0.001, run_time=10.0))

    num_exps = len(alists)

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
            exp = ExperimentAnalytes(k0 = dataframe.k0.values, S = dataframe.S.values, h=0.001, run_time=10.0)

        else:
            # Choose a random set
            set_index = randint(0, num_exps - 1) 
            exp = exps[set_index]
            input_data = torch.tensor(alists[set_index][['S', 'lnk0']].values, dtype=torch.float32)
            
        
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
        
        losses.append(expected_loss/sample_size)
        perfect_loss.append(exp.perfect_loss(weights))
        if (n + 1) % print_every == 0:
            print(f"Loss: {losses[-1]}, epoch: {n+1}/{num_episodes}")

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
        
    return np.array(losses), np.array(perfect_loss)


def reinforce_single_from_gen(
        alist: pd.DataFrame, 
        policy: PolicyGeneral, 
        delta_taus: Iterable[float], 
        num_episodes: int = 1000, 
        sample_size: int = 10, 
        lr: float = 1., 
        optim = torch.optim.SGD,
        lr_decay_factor: float = 1.,
        lr_milestones: Union[int, Iterable[int]] = 1000,
        print_every: int = 100,
        baseline: float = 0.,
        max_norm: float = None,
        beta:float = .0,
        weights: list = [1., 1.]
    ):
    """
    Run Reinforcement Learning for a single set learning.

    alist: od.DataFrame
        DataFrame with 'S', k0' and 'lnk0' information of the analyte set 
    policy: PolicyGeneral
        The policy that learns the optimal values for the solvent.
        strength program.
        This policy is the starting point of learning.
    delta_taus: Iterable[float]
        Iterable list with the points of solvent strength change.
        MUST be the same length as policy.n_steps
    num_episodes = 1000
        Number of learning steps.
    sample_size = 10
        Number of samples taken from the action distribution to perform 
        Expected loss for the distribution of actions.
    lr = 1.
        Learning rate.
    optim = torch.optim.SGD
        Optimizer that performs weight update using gradients.
        By defauld is Stochastic Gradient Descent.
    lr_decay_factor: float
        Learning rate decay factor used for the LRScheduler.
        lr is updated according to lr = lr ** lr_decay_factor.
    lr_milestones: Union[int, Iterable[int]]
        Milestone episode/s to update the learning rate.
        If it is int StepLR is used where lr is changed every lr_milestones.
        If it is a list of ints then at that specific episode the lr
        will be changed.
    print_every = 100,
        Number of episodes to print the average loss on.
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

    Returns
    -------
    (losses, best_program, mus, sigmas)
    losses: np.ndarray
        Expected loss of the action distribution over the whole learning
        process.
    best_program: np.ndarray
        list of the phis that had the lowest loss (based from the samples).
        NOTE: It might not be the global minima of the loss filed because
        the samples are not drawn from the whole loss space.
    mus: np.ndarray
        Mus change over the learning process. its shape is (num_episodes, policy.n_split)
    sigmas: np.ndarray
        Sigmas change over the learning process. its shape is (num_episodes, policy.n_split)
    """

    encoding = (policy.phi(torch.tensor(alist[['S', 'lnk0']].values, dtype=torch.float32))).mean(0, keepdim=True).detach()
    policy = deepcopy(policy.rho)

    exp = ExperimentAnalytes(k0=alist.k0.values, S=alist.S.values, h=0.001, run_time=10.0)


    losses = []
    loss_best = 2 
    epoch_mus = []
    epoch_sigmas = []

    # Optimizer
    optimizer = optim(policy.parameters(), lr)
    # LR sheduler
    if isinstance(lr_milestones, list) or isinstance(lr_milestones, np.ndarray):
        scheduler = MultiStepLR(optimizer, lr_milestones, gamma=lr_decay_factor)
    else:
        scheduler = StepLR(optimizer, lr_milestones, gamma=lr_decay_factor)

    
    for n in range(num_episodes):       
        # compute distribution parameters (Normal)
        mu, sigma = policy.forward(encoding)

        # Save parameters
        epoch_mus.append(mu.detach().numpy())
        epoch_sigmas.append(sigma.detach().numpy())

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
            if error < loss_best:
                loss_best = error
                best_program = constr_programs[i]
        losses.append(expected_loss/sample_size)
                        
        if (n + 1) % print_every == 0:
            print(f"Loss: {losses[-1]}, epoch: {n+1}/{num_episodes}")

        J /= sample_size  
        optimizer.zero_grad()

        # Calculate gradients
        J.backward()
        
        # Calculate gradients
        if max_norm:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm)

        # Apply gradients
        optimizer.step()

        # Adjust learning rate
        scheduler.step()
        
    return np.array(losses), best_program.numpy(),  np.array(epoch_mus), np.array(epoch_sigmas)

def reinforce_delta_tau_gen(
        alists: Iterable[pd.DataFrame],
        policy: PolicyGeneral,
        num_episodes: int = 1000, 
        sample_size: int = 10, 
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
        weights: list = [1., 1.]
    ):
    """
    Run Reinforcement Learning for a single set learning.

    alists: Iterable[pd.DataFrame]
        A list with pd.Dataframes for each dataset used to train on. 
    policy: PolicyGeneral
        The policy that learns the optimal values for the solvent
        strength program.
    num_episodes = 1000
        Number of learning steps.
    sample_size = 10
        Number of samples taken from the action distribution to perform 
        Expected loss for the distribution of actions.
    lr = 1.
        Learning rate.
    optim = torch.optim.SGD
        Optimizer that performs weight update using gradients.
        By defauld is Stochastic Gradient Descent.
    lr_decay_factor: float
        Learning rate decay factor used for the LRScheduler.
        lr is updated according to lr = lr ** lr_decay_factor.
    lr_milestones: Union[int, Iterable[int]]
        Milestone episode/s to update the learning rate.
        If it is int StepLR is used where lr is changed every lr_milestones.
        If it is a list of ints then at that specific episode the lr
        will be changed.
    rand_prob: float = .2
        The probability to draw a random subset from all the analytes.
        1 - rand_prob is the probability to use a "real" set (provided in
        alists).
    max_rand_analytes: int = 30
        The maximum number of analytes in the randomly drawn set.
    min_rand_analytes: int = 10
        The minimum number of analytes in the randomly drawn set.
    print_every = 100,
        Number of episodes to print the average loss on.
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
    lim: float
        The limit of the box around the ISO solution, i.e.
        Search space = ISO_solution +- lim for every new phi dimension.

    Returns
    -------
    (losses, best_program, mus, sigmas)
    losses: np.ndarray
        Expected loss of the action distribution over the whole learning
        process.
    best_program: np.ndarray
        list of the phis and delta taus that had the lowest loss (based from the samples).
        NOTE: It might not be the global minima of the loss filed because
        the samples are not drawn from the whole loss space.
    mus: np.ndarray
        Mus change over the learning process. its shape is (num_episodes, policy.n_split)
    sigmas: np.ndarray
        Sigmas change over the learning process. its shape is (num_episodes, policy.n_split)
    """

    losses = []
    exps = []

    # Make ExperimentAnalytes object for the given analyte sets for time saving purpose
    for alist in alists:
        exps.append(ExperimentAnalytes(k0 = alist.k0.values, S = alist.S.values, h=0.001, run_time=10.0))

    num_exps = len(alists)

    all_analytes = pd.concat(alists, sort=True)[['k0', 'S', 'lnk0']]

    # Optimizer
    optimizer = optim(policy.parameters(), lr)

    # LR sheduler
    if isinstance(lr_milestones, list) or isinstance(lr_milestones, np.ndarray):
        scheduler = MultiStepLR(optimizer, lr_milestones, gamma=lr_decay_factor)
    else:
        scheduler = StepLR(optimizer, lr_milestones, gamma=lr_decay_factor)

    for n in range(num_episodes):
        # the set to use for the experiment.
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
        
        # compute distribution parameters (Normal)
        mu, sigma = policy.forward(input_data)

        # Sample some values from the actions distributions
        values = sample(mu, sigma, sample_size)

        # Add tau for the last solvent strength (
        # runs until an analyte reaches th end of the columns
        new_values = np.ones((values.shape[0], values.shape[1] + 1)) * (exp.run_time if exp.run_time else 1e5)
        new_values[:, :-1] = values
        
        # Fit the sampled data to the constraint [0,1] or (0, +inf)
        constr_programs, delta_taus = np.split(new_values, 2, 1)
        constr_programs[constr_programs > 1] = 1
        constr_programs[constr_programs < 0] = 0
        delta_taus[delta_taus < 0] = 1e-7
        
        J = 0
        expected_loss = 0
        for i in range(sample_size):
            exp.reset()            
            exp.run_all(constr_programs[i], delta_taus[i])

            error = exp.loss(weights)
            expected_loss += error
            log_prob_ = log_prob(values[i], mu, sigma)
            J += (error - baseline) * log_prob_ - beta * torch.exp(log_prob_) * log_prob_
        
        losses.append(expected_loss/sample_size)

        if (n + 1) % print_every == 0:
            print(f"Loss: {losses[-1]}, epoch: {n+1}/{num_episodes}")

        J /= sample_size
        optimizer.zero_grad()
        # Calculate gradients
        J.backward()

        if max_norm:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm)

        # Apply gradients
        optimizer.step()

        # learning rate decay
        scheduler.step()
        
    return np.array(losses)


def reinforce_single_from_delta_tau_gen(
        alist: pd.DataFrame, 
        policy: PolicyGeneral,
        num_episodes: int = 1000, 
        sample_size: int = 10, 
        lr: float = 1., 
        optim = torch.optim.SGD,
        lr_decay_factor: float = 1.,
        lr_milestones: Union[int, Iterable[int]] = 1000,
        print_every: int = 100,
        baseline: float = 0.,
        max_norm: float = None,
        beta:float = .0,
        weights: list = [1., 1.],
    ):
    """
    Run Reinforcement Learning for a single set learning.

    alist: od.DataFrame
        DataFrame with 'S', k0' and 'lnk0' information of the analyte set 
    policy: PolicyGeneral
        The policy that learns the optimal values for the solvent.
        strength program.
        This policy is the starting point of learning.
    num_episodes = 1000
        Number of learning steps.
    sample_size = 10
        Number of samples taken from the action distribution to perform 
        Expected loss for the distribution of actions.
    lr = 1.
        Learning rate.
    optim = torch.optim.SGD
        Optimizer that performs weight update using gradients.
        By defauld is Stochastic Gradient Descent.
    lr_decay_factor: float
        Learning rate decay factor used for the LRScheduler.
        lr is updated according to lr = lr ** lr_decay_factor.
    lr_milestones: Union[int, Iterable[int]]
        Milestone episode/s to update the learning rate.
        If it is int StepLR is used where lr is changed every lr_milestones.
        If it is a list of ints then at that specific episode the lr
        will be changed.
    print_every = 100,
        Number of episodes to print the average loss on.
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

    Returns
    -------
    (losses, best_program, mus, sigmas)
    losses: np.ndarray
        Expected loss of the action distribution over the whole learning
        process.
    best_program: np.ndarray
        list of the phis that had the lowest loss (based from the samples).
        NOTE: It might not be the global minima of the loss filed because
        the samples are not drawn from the whole loss space.
    mus: np.ndarray
        Mus change over the learning process. its shape is (num_episodes, policy.n_split)
    sigmas: np.ndarray
        Sigmas change over the learning process. its shape is (num_episodes, policy.n_split)
    """

    encoding = (policy.phi(torch.tensor(alist[['S', 'lnk0']].values, dtype=torch.float32))).mean(0, keepdim=True).detach()
    policy = deepcopy(policy.rho)

    exp = ExperimentAnalytes(k0=alist.k0.values, S=alist.S.values, h=0.001, run_time=10.0)


    losses = []
    loss_best = 2 
    epoch_mus = []
    epoch_sigmas = []

    # Optimizer
    optimizer = optim(policy.parameters(), lr)
    # LR sheduler
    if isinstance(lr_milestones, list) or isinstance(lr_milestones, np.ndarray):
        scheduler = MultiStepLR(optimizer, lr_milestones, gamma=lr_decay_factor)
    else:
        scheduler = StepLR(optimizer, lr_milestones, gamma=lr_decay_factor)

    run_time_lim = exp.run_time if exp.run_time else 1e5
    for n in range(num_episodes):         
        
        # compute distribution parameters (Normal)
        mu, sigma = policy.forward(encoding)

        # Save parameters
        epoch_mus.append(mu.detach().numpy())
        epoch_sigmas.append(sigma.detach().numpy())

        # Sample some values from the actions distributions
        values = sample(mu, sigma, sample_size)

        # Add tau for the last solvent strength (
        # runs until an analyte reaches th end of the columns
        new_values = np.ones((values.shape[0], values.shape[1] + 1)) * run_time_lim
        new_values[:, :-1] = values
        
        # Fit the sampled data to the constraint [0,1] or (0, +inf)
        constr_programs, delta_taus = np.split(new_values, 2, 1)
        constr_programs[constr_programs > 1] = 1
        constr_programs[constr_programs < 0] = 0
        delta_taus[delta_taus < 0] = 1e-7
        
        J = 0
        expected_loss = 0
        for i in range(sample_size):
            exp.reset()            
            exp.run_all(constr_programs[i], delta_taus[i])

            error = exp.loss(weights)
            expected_loss += error
            log_prob_ = log_prob(values[i], mu, sigma)
            J += (error - baseline) * log_prob_ - beta * torch.exp(log_prob_) * log_prob_
            if error < loss_best:
                loss_best = error
                best_program = [constr_programs[i], delta_taus[i]]
        losses.append(expected_loss/sample_size)
                        
        if (n + 1) % print_every == 0:
            print(f"Loss: {losses[-1]}, epoch: {n+1}/{num_episodes}")

        J /= sample_size  
        optimizer.zero_grad()
        
        # Calculate gradients
        J.backward()

        if max_norm:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm)

        # Apply gradients
        optimizer.step()

        # Adjust learning rate
        scheduler.step()
        
    return np.array(losses), best_program,  np.array(epoch_mus), np.array(epoch_sigmas)