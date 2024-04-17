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


def get_linear_bias(gal_pos, sim, nx=500,
                r_edges=np.logspace(np.log10(1), np.log10(100.), 21),
                bias_range=(8,-5), return_ratio=False):
    """
    Returns the linear bias given a galaxy sample and an input TNG `sim` object.

    Parameters
    ----------
    gal_pos : 2darray
        A (N,3) array of galaxy positions (x,y,z).
    sim : TNGSim object

    nx : int, optional
        Subsample parameter for dark matter, to reduce computation time: take every `nx`th particle.
        The default is 500.
    r_edges : 1darray, optional
        The separation bins to use when computing the pair counts.
    bias_range : len(2) tuple, optional
        The lower and upper indices to use to compute the bias from the ratio of the 3D c.f.s,
        gal. x DM to linear theory (DM x DM).
        The default is (8,-5), based on a few trials with the default `r_edges`.
    return_ratio : bool, optional
        Whether to return the `len(r_edges)-1` ratio of xi(gal x DM)) to xi(lin)—if we don't trust
        the input `bias_range` without looking at this curve by eye.

    Returns
    -------
    bias : float
        The linear galaxy bias: the mean ratio of xi(gal x DM) to xi(lin) over `bias_range`.

    """

    # bias: Gal x DM / linear theory
    dm_pos = tools.get_subsample(sim.dm_pos(), nx=nx).value  # underlying dark matter
    L = sim.boxsize.value
    # corresponding random set
    rand_pos = np.random.uniform(0, L, (len(dm_pos),3))

    # format galaxy sample
    gal_pos = gal_pos.value if isinstance(gal_pos, u.Quantity) else gal_pos
    if np.amin(gal_pos) < 0:
        gal_pos += L/2
    for i, x in enumerate([dm_pos, rand_pos, gal_pos]):
        assert 0 < np.all(x) < L 
    
    # Gal x DM cross correlation
    ravg, xix_spec = corrfuncs.xi_cross(gal_pos, dm_pos, rand_pos, r_edges, boxsize=L, dtype=float)

    # linear theory from Colossus
    xi_lin = tools.linear_2pcf(sim.redshift, ravg)

    # ratio
    ratio = xix_spec / xi_lin

    if bias_range is None:
        bias = np.nanmean(ratio)
    else:
        bias = np.nanmean(ratio[bias_range[0]:bias_range[1]])
    
    if return_ratio == True:
        return bias, ratio
    else:
        return bias
        

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