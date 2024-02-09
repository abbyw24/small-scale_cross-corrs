import numpy as np
import astropy.units as u
from astropy.cosmology import Planck15 as cosmo
from colossus.cosmology import cosmology
from scipy import special, integrate
import os
import sys

from illustris_sim import TNGSim
import corrfuncs
import tools

def powerspec_to_wlin(theta, ell, Cell):
    """
    Transform an input matter power spectrum Cell to the real-space linear angular correlation function w(theta).
    """

    theta = theta.to(u.rad) if isinstance(theta, u.Quantity) else theta << u.rad
    
    assert ell.ndim == Cell.ndim == 1, "input ell and Cell must be 1D"

    # function of ell that we want to integrate
    def integrand(ell_, Cell_):
        return (ell_ / (2 * np.pi)) * Cell_ * (special.jv(0, ell_ * theta.value))

    # construct our array, and integrate using trapezoid rule
    ell_func_arr = np.array([
        integrand(ell[i], Cell[i]) for i in range(len(ell))
    ])
    trapz = integrate.trapz(ell_func_arr, x=ell)

    return trapz, ell_func_arr