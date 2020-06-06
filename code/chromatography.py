"""
    Module for all relavant functions for performing 
    liquid chromatography simulations.
"""
import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt


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
            run_time: float  = None
        ) -> None:
        """
        Initialize the class properties.

        Parameters
        ----------
        k0: np.ndarray
        S: np.ndarray, 
        h = 0.001: float, 
        run_time = None: float
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
        self.at_the_end = False
        

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
        self.at_the_end = False

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
    
    
    def intersection_time(self, phi, delta_tau_phi):
        """
        Compute time needed for the analytes to reach 
        the new solvent for isocratic gradient function.
        """
        return delta_tau_phi * (1 + self.k(phi)) / self.k(phi)
    
    
    def x(self, phi, delta_tau):
        """
        Compute distance traveled by the analytes for a 
        given solvent and time interval,
        for isocratic gradient function.
        """
        
        return delta_tau /(self.k(phi) + 1)
    
    
    def step(self, phi, delta_tau_phi):
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
        
        if over_x_lim.any() and not self.at_the_end:
            self.at_the_end = True
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
        self.positions.append(self.positions[-1] + self.x(phi, intersect_time))

        
        
        return self.done


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
            plt.plot(u + tau, u, linestyle='--', c='k', label='change of solvent strength'*(i == 0))
            
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
            weights=[1., 1.],
        ):
        """
        Compute loss of the Experiment based on placement error and overlap error.

        Parameters
        ----------
        weights: list
            Weigths of the errors to consider, first one is for the Placement Error,
            second one is for Overlap Error, By default both have the same wights.
        """

        return (
            weights[0] * placement_error(self.positions[-1], self.even_space_positions) + 
            weights[1] * overlap_error(self.positions[-1], self.sig)
        )


    def perfect_loss(
            self,
            weights=[1., 1.],
        ):
        """
        Compute loss of the Experiment based on placement error and overlap error.

        Parameters
        ----------
        weights: list
            Weigths of the errors to consider, first one is for the Placement Error,
            second one is for Overlap Error, By default both have the same wights.
        """

        return weights[1] * overlap_error(self.even_space_positions, 
                            self.even_space_positions * np.sqrt(self.h))
