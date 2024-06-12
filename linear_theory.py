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


def get_linear_bias(gal_pos, sim, method='cross_dm',
                    rmin=1, rmax=100, nbins=20, logbins=True,
                    nx=500, randmult=3, periodic=False):
    """
    Returns the linear bias as a function of comoving separation b(r),
    given a galaxy sample and an input TNG `sim` object.

    Parameters
    ----------
    gal_pos : 2darray
        A (N,3) array of galaxy positions (x,y,z).
    sim : TNGSim object

    method : str ('cross_dm' or 'auto_gal'), optional
        Which formula to use in the calculation. 'cross_dm' loads the dark matter particle positions
        from the TNG snapshot, computes the cross-correlation between the galaxy and dark matter
        (subsampled with `nx`), and returns the ratio between the cross-correlation and the linear
        matter correlation function. 'auto_gal' computes the galaxy auto-correlation and returns the
        square root of the ratio between the auto-correlation and the linear matter correlation function.
        The default is 'cross_dm'.
    rmin : float, optional

    rmax : float, optional

    nbins : int, optional

    logbins : bool, optional

    nx : int, optional
        Subsample parameter for dark matter (if 'cross_dm'), to reduce computation time:
        take every `nx`th particle. The default is 500.
    randmult : int, optional
        Multiplication factor for number of random particles used in the pair counts IF `method`
        is 'auto_gal', otherwise not used (this is effectively determined by `nx` for 'cross_dm').
        The default is 3.
    periodic : bool, optional
        Whether to compute the pair counts on a periodic box, passed to Corrfunc.

    Returns
    -------
    bias : 1D array of length nbins+1
        The linear galaxy bias as a function of comoving separation b(r).

    """

    if method.lower() == 'cross_dm':
        # load dark matter positions and randomly subsample
        dm_pos = tools.get_subsample(sim.dm_pos(), nx=nx)
        # Gal x DM cross correlation
        ravg, galxdm = corrfuncs.compute_xi_cross(dm_pos, gal_pos, 1, rmin, rmax, nbins,
                                                    boxsize=sim.boxsize, periodic=periodic)
        # denominator is linear matter correlation function from Colossus
        ratio = galxdm / tools.linear_2pcf(sim.redshift, ravg)

    elif method.lower() == 'auto_gal':
        # Gal x Gal auto correlation
        ravg, galxgal = corrfuncs.compute_xi_auto(gal_pos, randmult, rmin, rmax, nbins,
                                                    boxsize=sim.boxsize, periodic=periodic)
        # denominator is linear matter correlation function from Colossus
        ratio = np.sqrt(galxgal / tools.linear_2pcf(sim.redshift, ravg))
        
    else:
        assert False, "input method must be 'cross_dm' or 'auto_gal'"

    return ratio
        

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