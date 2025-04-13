# distributions.py
# Contains distribution functions (PMF, SF) for any used distributions. 

import numpy as np
from scipy.special import zeta

## DISCRETE POWER LAW DISTS ##

def dpl_pmf(x, xmin, alpha, log=False):
    """
    Computes the PMF for a discrete power law.
    
    Args:
        x (int or array-like): Value(s) for evaluation (requires x >= xmin).
        xmin (int): Lower cutoff for the distribution support.
        alpha (float): Scaling exponent.
        log (bool): If True, returns the log PMF.
    
    Returns:
        float or numpy.ndarray: PMF (or log PMF).
        
    """
    # Ensure input is a NumPy array for vectorized operations.
    x = np.asarray(x, dtype=np.float64)
    # Normalization constant using the Hurwitz zeta function: ζ(alpha, xmin)
    norm = zeta(alpha, xmin)
    pmf_val = x ** (-alpha) / norm
    return np.log(pmf_val) if log else pmf_val

def dpl_sf(x, xmin, alpha):
    """
    Computes the survival function (SF) for the discrete power law distribution.
    
    Args:
        x (int or array-like): Value(s) at which the SF is evaluated (x >= xmin).
        xmin (int): Lower cutoff for the distribution support.
        alpha (float): Scaling exponent.
    
    Returns:
        float or numpy.ndarray: Survival probability P(X >= x).
        
    """
    x = np.asarray(x, dtype=np.float64)
    norm = zeta(alpha, xmin)
    return zeta(alpha, x) / norm

## END DISCRETE POWER LAW DISTS ##
