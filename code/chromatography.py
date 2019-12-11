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