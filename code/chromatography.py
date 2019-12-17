"""
    Module for all relavant functions for performing 
    liquid chromatography.
"""
import numpy as np
from scipy.special import erf


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
            
        self.time_travel = np.zeros(k0.shape)
        self.positions = np.zeros(k0.shape)
        self.final_position = 1/(1 + 2 * h ** 0.5)
        
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
        
        return np.sqrt((self.position**2) * self.h)
    
    
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
        # Compute the interval of time needed to intersect with the next phi.
        intersection_tau = self.intersection_time(phi, delta_tau_phi)
        # Compute the distance Traveled until the intersection.
        delta_x = self.x(phi, intersection_tau)

        over_x_lim = (self.positions + delta_x > self.final_position)
        #print(f'Over X:{over_x_lim.any()}')
        if over_x_lim.any():
            max_time = (self.time_travel[over_x_lim] + \
                        (self.final_position - self.positions[over_x_lim]) * \
                        (self.k(phi)[over_x_lim] + 1)).min()
            
            if self.run_time:
                self.run_time = np.min([self.run_time, max_time])
            else:
                self.run_time = max_time
        #print(f'Run Time:{self.run_time}')
        if self.run_time:
            over_time = (self.time_travel + intersection_tau > self.run_time)
            if over_time.any():
                intersection_tau[over_time] = self.run_time - self.time_travel[over_time]
        #print(f'Time:{intersection_tau}\n')
        # Update time travel and position.
        self.time_travel += intersection_tau
        self.positions += self.x_iso(phi, intersection_tau)
        
        return self.time_travel, self.positions

    
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