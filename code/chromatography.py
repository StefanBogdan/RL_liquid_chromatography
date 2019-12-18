"""
    Module for all relavant functions for performing 
    liquid chromatography.
"""
import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt


def placement_error(analytes_pl, true_pl=None):
    """
    Conpute placement error for a given analyte list result.
    
    Parameters
    ----------
    analytes_pl: 
        list with analyte placements.
    true_pl:
        the true placements that the analytes should have.
        
    Returns
    -------
        returns the placement error
    """
    
    list_len = analytes_pl.shape[0]
    
    if not true_pl:
        true_pl = np.arange(1, list_len + 1)/list_len
       
    return 2 * np.abs(analytes_pl - true_pl).sum() / (list_len - 1)

def cdf(x, mu, sigma):
    """
    Cumulative distribution function for Normal Distribution.
    
    Parameters
    ----------
    x:
        position to calculate P(X < x)
    mu:
        mean of the Normal Distribution.
    sigma:
        standard deviation of the Normal Distribution.
        
    Return
    ------
        cdf at x.
    """
    
    
    return 0.5 * (1.0 + erf((x - mu) / (sigma * np.sqrt(2.0))))

def overlap_error(mu, sigma):
    """
    Compute the overlaping error of the analytes.
    
    Parameters
    ----------
    mu:
        mean of the peak per analyte
    sigma:
        standard deviation of the peak per analyte.
        
    Returns
    -------
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
    
    
    def __init__(self, k0, S, h=0.001, run_time=None, grad='iso'):
        """
        Initialize the class properties.
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
        
        ### Set initial values ###
        
        self.k0 = k0
        self.S = S
        self.h = float(h)
        if run_time:
            self.run_time = float(run_time)
        else:
            self.run_time = run_time
            
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
            
    
    def __str__(self):
        return (
            f"k0: {self.k0}\nS: {self.S}\nh: {self.h}\n" + 
            f"run_time: {self.run_time}\nLast time_travel: {self.time_travel[-1]}\n" +
            f"Last positions: {self.positions[-1]}\nfinal_position: {self.final_position}\n" +
            f"grad: {self.grad}\nphis: {self.phis}\ndelta_taus: {self.delta_taus}\n" +
            f"done: {self.done}"
        )
    
        
    def k(self, phi):
        """
        Compute retention factor k a given phi.
        """
        
        return (10 ** (-phi * self.S)) * self.k0
    
    
    @property
    def sig(self):
        """
        Compute second Gaussian moment for the analytes.
        """
        
        return self.position * np.sqrt(self.h)
    
    
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


    def print_analytes(self, title="Solvent Gradient Function", angle=50,rc=(13, 10)):
        """
        Plot the dinamics of the analytes for the experiment.
        """

        plt.rcParams['figure.figsize'] = rc
        plt.rcParams['axes.facecolor'] = 'whitesmoke'

        # print limit of x
        plt.axhline(y=self.final_position, color='r', linestyle='--')

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
            # Plot the volocity of the solvent
            plt.plot(u + tau, u, linestyle='--', c='k')
            
            # Add the solvent gradient on the line
            l2 = np.array(((u+tau)[-15] - 0.02, u[-15]))
            th2 = plt.text(
                l2[0],
                l2[1],
                f"phi = {self.phis[i]}",
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