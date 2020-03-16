"""
    Module for all relavant functions for performing 
    liquid chromatography.
"""
import numpy as np
import torch 
import torch.nn as nn
from torch import optim, tensor
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from scipy.special import erf
import matplotlib.pyplot as plt


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
            MultivariateNormal(mu, torch.diag(sigma))
            .log_prob(value)
            )


def placement_error(
        analytes_pl: np.ndarray, 
        true_pl: np.ndarray
    ) -> np.float64:
    """
    Conpute placement error for a given analyte list result.
    
    Parameters
    ----------
    analytes_pl: np.ndarray
        list with analyte placements.
    true_pl: np.ndarray
        the true placements that the analytes should have.
        
    Returns
    -------
    np.ndarray
        returns the placement error
    """
    
    return 2 * np.abs(np.sort(analytes_pl) - true_pl).sum() / analytes_pl.shape[0]

def cdf(
        x: np.ndarray, 
        mu: np.ndarray, 
        sigma: np.ndarray
    ) -> np.ndarray:
    """
    Cumulative distribution function for Normal Distribution.
    
    Parameters
    ----------
    x: np.ndarray
        position to calculate P(X < x)
    mu: np.ndarray
        mean of the Normal Distribution.
    sigma: np.ndarray
        standard deviation of the Normal Distribution.
        
    Returns
    -------
    np.ndarray
        cdf at x.
    """
    
    
    return 0.5 * (1.0 + erf((x - mu) / (sigma * np.sqrt(2.0))))

def overlap_error(
        mu: np.ndarray, 
        sigma: np.ndarray
    ) -> np.float64:
    """
    Compute the overlaping error of the analytes.
    
    Parameters
    ----------
    mu: np.ndarray
        mean of the peak per analyte
    sigma: np.ndarray
        standard deviation of the peak per analyte.
        
    Returns
    -------
    np.float64
        The overlaping error of all the analytes.
    """


    indexes = np.argsort(sigma)
    mu_1 = (mu[indexes])[:-1]
    sigma_1 = (sigma[indexes])[:-1]
    var_1 = sigma_1 ** 2
    mu_2 = (mu[indexes])[1:]
    sigma_2 = (sigma[indexes])[1:]
    var_2 = sigma_2 ** 2

    d_var = var_2 - var_1
    d_mu = np.fabs(mu_2 - mu_1)
    
    # Split into two cases: When d_var is zero and when not
    # Case d_var is zero
    zero_d_var = (d_var == 0)
    overlap_area = (1.0 - erf(d_mu[zero_d_var] / (2.0 * \
            sigma_1[zero_d_var] * np.sqrt(2.0)))).sum()
    
    # Case d_var is not zero
    if (~zero_d_var).sum() == 0:
        return overlap_area/zero_d_var.sum()
    
    # reassigne values
    mu_1 = mu_1[~zero_d_var]
    mu_2 = mu_2[~zero_d_var]
    sigma_1 = sigma_1[~zero_d_var]
    sigma_2 = sigma_2[~zero_d_var]
    var_1 = var_1[~zero_d_var]
    var_2 = var_2[~zero_d_var]
    d_var = d_var[~zero_d_var]
    d_mu = d_mu[~zero_d_var]
    
    a = mu_1 * var_2 - mu_2 * var_1
    b = (
        sigma_1 * sigma_2 * 
            np.sqrt(d_mu**2.0 + d_var * np.log(var_2 / var_1))
    )
    x1 = (a + b) / d_var
    x2 = (a - b) / d_var
    overlap_area += (
        (1.0 - (np.fabs(cdf(x1, mu_2, sigma_2) - cdf(x1, mu_1, sigma_1))\
        + np.fabs(cdf(x2, mu_2, sigma_2) - cdf(x2, mu_1, sigma_1)))).sum()
    )
    
    return overlap_area/(len(mu) - 1)

class ExperimentAnalytes(object):
    """
    Encapsulation of the Experiment data.
    """
    
    
    def __init__(
            self, 
            k0: np.ndarray, 
            S: np.ndarray, 
            h: float = 0.001, 
            run_time: float  = None, 
            grad: str = 'iso'
        ) -> None:
        """
        Initialize the class properties.

        Parameters
        ----------
        k0: np.ndarray
        S: np.ndarray, 
        h = 0.001: float, 
        run_time = None: float, 
        grad = 'iso': str
        """
        
        
        # 'k0' var datatype check
        if not isinstance(k0, np.ndarray):
            raise TypeError(f"'k0' is of a wrong data type (given '{type(k0)}', needed 'numpy.ndarray')")
            
        # 'S' var datatype check
        if not isinstance(S, np.ndarray):
            raise TypeError(f"'S' is of a wrong data type (given '{type(S)}', needed 'numpy.ndarray')")
        
        # 'k0' and 'S' shape match
        if not k0.shape == S.shape:
            raise ValueError(f"Shape mismatch: k0 -> {k0.shape}, S -> {S.shape}")

        # 'run_time' should be positive.
        if run_time and run_time <= 0:
            raise ValueError(f"'run_time' should be >0, given: {run_time}")
        
        ### Set initial values ###
        self.n_analytes = len(k0)
        self.k0 = k0
        self.S = S
        self.h = float(h)
        self.run_time_init = run_time
        if run_time:
            self.run_time = float(run_time)
        else:
            self.run_time = None
            
        self.time_travel = [np.zeros(k0.shape)]
        self.positions = [np.zeros(k0.shape)]
        self.final_position = 1/(1 + 2 * h ** 0.5)
        self.phis = []
        self.delta_taus = []
        self.done = False
        
        if grad == 'iso':
            self.x = self.x_iso
            self.intersection_time = self.intersection_time_iso
            self.step = self.step_iso
        elif grad == 'linear':
            self.x = self.x_linear
            self.intersection_time = self.intersection_time_linear
            self.step = self.step_linear
        else:
            raise ValueError(f"'grad' can have only these values ['iso', 'linear'], given '{grad}'")
            
        self.grad = grad

    def reset(
            self
        ) -> None:
        """
            Reset to default settings.
        """

        if self.run_time_init:
            self.run_time = float(self.run_time_init)
        else:
            self.run_time = None

        self.time_travel = [np.zeros(self.k0.shape)]
        self.positions = [np.zeros(self.k0.shape)]
        self.phis = []
        self.delta_taus = []
        self.done = False           
    
    def __str__(self):
        return (
            f"n_analytes: {self.n_analytes}\n" +
            f"k0: {self.k0}\n" +
            f"S: {self.S}\n" +
            f"h: {self.h}\n" + 
            f"run_time: {self.run_time}\n" +
            f"Last time_travel: {self.time_travel[-1]}\n" +
            f"Last positions: {self.positions[-1]}\n" +
            f"final_position: {self.final_position}\n" +
            f"grad: {self.grad}\n" +
            f"phis: {self.phis}\n" +
            f"delta_taus: {self.delta_taus}\n" +
            f"done: {self.done}"
        )
    
        
    def k(self, phi):
        """
        Compute retention factor k for a given phi.
        """
        
        return (10 ** (-phi * self.S)) * self.k0
    
    @property
    def even_space_positions(self):
        return np.linspace(0, self.final_position, self.n_analytes + 1)[1:]

    @property
    def sig(self):
        """
        Compute second Gaussian moment for the analytes.
        """
        
        return self.positions[-1] * np.sqrt(self.h)
    
    
    def intersection_time_iso(self, phi, delta_tau_phi):
        """
        Compute time needed for the analytes to reach 
        the new solvent for isocratic gradient function.
        """
        return delta_tau_phi * (1 + self.k(phi)) / self.k(phi)
    
    
    def x_iso(self, phi, delta_tau):
        """
        Compute distance traveled by the analytes for a 
        given solvent and time interval,
        for isocratic gradient function.
        """
        
        return delta_tau /(self.k(phi) + 1)
    
    
    def step_iso(self, phi, delta_tau_phi):
        """
        Compute Time and Distance traveled by each analyte.
        Update Distance and Time traveled by each analyte.
        Isocratic gradient function.
        """


        # If all analytes traveled the run_time there is nothing else to do
        if self.done:
            return True

        # Add gradient and delta tau to the to the list of gradients
        # and  delta taus respectively
        self.phis.append(phi)
        self.delta_taus.append(delta_tau_phi)
        # Compute the interval of time needed to intersect with the next phi.
        intersect_time = self.intersection_time(phi, delta_tau_phi)
        # Compute the distance Traveled until the intersection.
        delta_x = self.x(phi, intersect_time)

        over_x_lim = (self.positions[-1] + delta_x > self.final_position)
        
        if over_x_lim.any():
            max_time = (self.time_travel[-1][over_x_lim] + \
                        (self.final_position - self.positions[-1][over_x_lim]) * \
                        (self.k(phi)[over_x_lim] + 1)).min()
            if self.run_time:
                self.run_time = np.min([self.run_time, max_time])
            else:
                self.run_time = max_time

        if self.run_time:
            over_time = (self.time_travel[-1] + intersect_time > self.run_time)
            if over_time.any():
                intersect_time[over_time] = self.run_time - self.time_travel[-1][over_time]
            # if all analytes are over time then all are done:
            if over_time.all():
                self.done = True

        # Append new time travel and position.
        self.time_travel.append(self.time_travel[-1] + intersect_time) 
        self.positions.append(self.positions[-1] + self.x_iso(phi, intersect_time))

        
        
        return self.done

    
    def intersection_time_linear(self, phi_init, phi_final, delta_tau_phi):
        """
        Compute time needed for the analytes to reach 
        the new solvent for linear gradient function.
        """
        
        pass
    
    
    def x_linear(self, phi_init, phi_final, delta_tau):
        """
        Compute distance traveled by the analytes for a 
        given solvent and time interval,
        for linear gradient function.
        """
        
        pass
        

    def step_linear(self, phi_init, phi_final, delta_tau_phi):
        """
        Compute Time and Distance traveled by each analyte.
        Update Distance and Time traveled by each analyte.
        Isocratic gradient function.
        """

        
        pass


    def print_analytes(self, title="Solvent Gradient Function", angle=50, rc=(13, 10)):
        """
        Plot the dinamics of the analytes for the experiment.
        """

        plt.rcParams['figure.figsize'] = rc
        plt.rcParams['axes.facecolor'] = 'whitesmoke'

        # print limit of x
        plt.axhline(y=self.final_position, color='r', linestyle='--', label='x_lim')

        # coordinated for the gradient velocity
        u = np.linspace(0.0, self.final_position + 0.2, 100)

        # Start time of the first gradient
        tau = 0

        for i in range(len(self.time_travel) - 1):
            # This is for each segment in the solvent gradient to be different
            # Good for debuging
            if i % 2:
                col = 'r'
            else:
                col = 'b'
            # Plot the velocity of the solvent
            plt.plot(u + tau, u, linestyle='--', c='k', label='change of solvent strength'*i)
            
            # Add the solvent gradient on the line
            l2 = np.array(((u+tau)[-15] - 0.02, u[-15]))
            th2 = plt.text(
                l2[0],
                l2[1],
                f"phi = {str(np.round(self.phis[i], 3))}",
                fontsize=12,
                rotation=angle,
                rotation_mode='anchor'
                )
            tau += self.delta_taus[i]

            for a in range(len(self.time_travel[i])):
                plt.plot(
                    np.linspace(self.time_travel[i][a], self.time_travel[i+1][a], 100),
                    np.linspace(self.positions[i][a], self.positions[i+1][a], 100),  
                    c=col
                    )

            # Add Title and axes
            plt.title(title)
            plt.xlabel('Tau, [dimensionless]')
            plt.ylabel('Position, [dimensionless]')
            plt.grid(True)


    def run_all(self, phis, delta_taus):
        """
            
        """
        
        if len(phis) != len(delta_taus):
            raise ValueError(f"Input parametrs does not match length!{len(phis)}:{len(delta_taus)}")
        
        for phi, delta_tau in zip(phis, delta_taus):
            self.step(phi, delta_tau)

    def loss(
            self,
            weights=[1., 1.]
        ):
        """
        Compute loss of the Experiment based on placement error and overlap error.
        """

        return (
            weights[0] * placement_error(self.positions[-1], self.even_space_positions) + 
            weights[1] * overlap_error(self.positions[-1], self.sig)
        )


class Policy(nn.Module):

    def __init__(self, n_param, sigma_min = .0):

        super(Policy, self).__init__()
        self.n_param = n_param
        self.sigma_min = sigma_min

        # Define network
        self.fc_mu = nn.Linear(1, n_param, bias=False)
        self.sig = nn.Sigmoid()
        self.fc_sigma = nn.Linear(1, n_param, bias=False)
        self.softplus = nn.Softplus()
        
        
    def forward(self):
        out = torch.ones((1,1))
        self.mu = self.sig(self.fc_mu(out)).squeeze()
        self.sigma = self.sig(self.fc_sigma(out)).squeeze()/10.0 + self.sigma_min
        
        return self.mu, self.sigma
    
    def sample(self, n_samples):
        
        if self.n_param == 1:
            return (
                Normal(
                    self.mu, self.sigma
                )
                .sample((n_samples,))
            )
        else:
            return (
                MultivariateNormal(
                    self.mu, torch.diag(self.sigma)
                )
                .sample((n_samples,))
            )

class PolicyTime(nn.Module):

    def __init__(self, n_param, sigma_min = .0):
        super(PolicyTime, self).__init__()
        self.n_param = 2 * n_param
        self.sigma_min = sigma_min
        
        # Define network
        self.fc_mu = nn.Linear(1, n_param * 2 - 1, bias=False)
        self.sig = nn.Sigmoid()
        self.fc_sigma = nn.Linear(1, n_param * 2 - 1, bias=False)
        self.softplus = nn.Softplus()
        
        
    def forward(self):
        out = torch.ones((1,1))
        self.mu = self.sig(self.fc_mu(out)).squeeze() 
        self.sigma = self.softplus(self.fc_sigma(out)).squeeze() + self.sigma_min

        
        return self.mu, self.sigma
    
    def sample(self, n_samples):
        
        if self.n_param == 1:
            return (
                Normal(
                    self.mu, self.sigma
                )
                .sample((n_samples,))
            )
        else:
            return (
                MultivariateNormal(
                    self.mu, torch.diag(self.sigma)
                )
                .sample((n_samples,))
            )


def reinforce(
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
        baseline = 0.
    ):

    losses = []
    loss = 0
    loss_best = 2 
    epoch_mus = []
    epoch_sigmas = []
    epoch_n_taus = []

    
    for n in range(num_episodes):           
        
        # Optimizer
        optimizer = optim(policy.parameters(), lr)
        
        # compute distribution parameters (Normal)
        policy.forward()

        # Save parameters
        epoch_mus.append(policy.mu.detach().numpy())
        epoch_sigmas.append(policy.sigma.detach().numpy())

        # Sample some values from the actions distributions
        programs = policy.sample(batch_size)
        
        # Fit the sampled data to the constraint [0,1]
        constr_programs = programs.clone()
        constr_programs[constr_programs > 1] = 1
        constr_programs[constr_programs < 0] = 0
        
        J = 0
        for i in range(batch_size):
            exp.reset()            
            exp.run_all(constr_programs[i].data.numpy(), delta_taus)
            epoch_n_taus.append(len(exp.delta_taus))

            error = exp.loss(weights)
            loss += error
            J += (error - baseline) * log_prob(programs[i], policy.mu, policy.sigma)
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
        
        # Apply gradients
        optimizer.step()

        # learning rate decay
        if lr_decay:
            lr = lr_decay(lr, (n + 1), num_episodes) 
        
    return np.array(losses), best_program,  np.array(epoch_mus), np.array(epoch_sigmas), np.array(epoch_n_taus)

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
        baseline = 0.
    ):

    losses = []
    loss = 0
    loss_best = 2
    epoch_n_par = []
    epoch_mus = []
    epoch_sigmas = []
    
    for n in range(num_episodes):
        # learning rate decay
        if lr_decay:
            lr = lr_decay(lr, (n + 1), num_episodes)            
        
        # Optimizer
        optimizer = optim(policy.parameters(), lr=lr)
        
        # compute distribution parameters (Normal)
        policy.forward()
        
        # Save parameters
        epoch_mus.append(list(policy.mu))
        epoch_sigmas.append(list(policy.sigma))
        
        # Sample some values from the actions distributions
        values = policy.sample(batch_size)

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
                
        # Apply gradients
        optimizer.step()
        
    return losses, best_program, np.array(epoch_mus), np.array(epoch_sigmas), np.array(epoch_n_par)