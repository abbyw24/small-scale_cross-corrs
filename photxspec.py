"""
Functions to construct a 2D photometric sample from a spectroscopic sample constructed from TNG,
compute their cross correlations, and compare to the linear theory prediction.
"""

import numpy as np
import astropy.units as u
import astropy.cosmology.units as cu
from astropy.cosmology import Planck15 as cosmo
from colossus.cosmology import cosmology
from scipy import special, integrate
import os
import sys

from illustris_sim import TNGSim
import corrfuncs
import tools


def construct_photometric_sample(gal_pos_spec, dx, boxsize, mean=0.):
    gal_pos_phot = []
    for i, pos in enumerate(gal_pos_spec):
        draw = tools.eval_Gaussian(pos[2], dx, mean=mean) * dx.unit
        if draw > np.random.uniform():
            gal_pos_phot.append(pos)
        else:
            continue
    return np.array(gal_pos_phot) << dx.unit


def compute_photxspec(gal_pos_spec, redshift, sigma_z, boxsize,
                nslices=11, nbins=10, thetamin=.001, thetamax=5.):
    
    # prep inputs
    gal_pos_spec = gal_pos_spec.to(u.Mpc/cu.littleh) if isinstance(gal_pos_spec, u.Quantity) \
                    else gal_pos_spec * u.Mpc/cu.littleh
    boxsize = boxsize.to(u.Mpc/cu.littleh) if isinstance(boxsize, u.Quantity) else boxsize * u.Mpc/cu.littleh
    # width of Gaussian for constructing the photometric sample
    dx = tools.get_dx(redshift, sigma_z).to(boxsize.unit)
    # comoving distance to the box center
    r = tools.redshift_to_comov(redshift).to(dx.unit)
    # separation bins
    bins = np.logspace(np.log10(thetamin), np.log10(thetamax), nbins+1)
    
    # divide into narrow 2D slices along the LOS
    losbins = np.linspace(-boxsize.value/2, boxsize.value/2, nslices+1)
    slicewidth = (losbins[1]-losbins[0])
    slice_centers = np.array([(losbins[i]+losbins[i+1])/2 for i in range(nslices)])
    
    # create nslices samples by shifting and wrapping the LOS positions around the box
    gal_pos_specs = np.empty((nslices, len(gal_pos_spec), 3))
    for i in range(nslices):
        # first, shift everything by i * slicewidth
        gal_pos_spec_ = np.copy(gal_pos_spec)
        gal_pos_spec_[:,2] += i * slicewidth << gal_pos_spec_.unit
        # wrap any galaxies outside the box
        idx_to_wrap = (gal_pos_spec_[:,2] >= boxsize / 2)
        gal_pos_spec_[:,2][idx_to_wrap] -= boxsize
        gal_pos_specs[i] = gal_pos_spec_
        
    gal_pos_phots = [
        construct_photometric_sample(gal_pos_spec_ << dx.unit, dx, boxsize) for gal_pos_spec_ in gal_pos_specs
    ]

    # compute the cross correlation in each slice in each sample and take the average to reduce noise
    xcorrs = np.empty((nslices, nslices, nbins))
    for i, spec_sample in enumerate(gal_pos_specs):
        # divide the spectroscopic sample and corresponding photometric sample into slices
        slices_spec = [
            spec_sample[(losbins[i] <= spec_sample[:,2]) & (spec_sample[:,2] < losbins[i+1])] \
            for i in range(nslices)
        ]
        # convert photometric sample to (RA,Dec), setting LOS positions to box center
        phot_sample = np.copy(gal_pos_phots[i].value)
        phot_sample[:,2] = 0
        ra_phot, dec_phot = tools.CartesiantoEquatorial(phot_sample, observer=[0.,0.,r.value])
        # random sample
        nd1 = len(ra_phot)
        ra_rand_phot = np.random.uniform(min(ra_phot), max(ra_phot), nd1)
        dec_rand_phot = np.random.uniform(min(dec_phot), max(dec_phot), nd1)

        for j, spec_slice in enumerate(slices_spec):
            # prep spectroscopic slice and corresponding random set
            nd2 = len(spec_slice)
            gal_pos_ = np.copy(spec_slice)
            gal_pos_[:,2] = 0
            ra_spec_, dec_spec_ = tools.CartesiantoEquatorial(gal_pos_, observer=[0.,0.,r.value])

            # compute xcorr
            thetaavg, xcorrs[i,j] = corrfuncs.wtheta_cross_PH(ra_phot, dec_phot, ra_spec_, dec_spec_,
                                              ra_rand_phot, dec_rand_phot, bins)
            
    res = {
        'dx' : dx,
        'thetaavg' : thetaavg,
        'xcorrs': np.mean(xcorrs, axis=0),
        'slice_centers' : slice_centers,
        'losbins' : losbins,
        'gal_pos_specs' : gal_pos_specs,
        'gal_pos_phots' : gal_pos_phots
    }
    return res


def wlin_theory(gal_pos_phot, gal_pos_spec, sim, thetaavg, ell=np.logspace(0, 6, 1000), nx=500,
                r_edges=np.logspace(np.log10(1), np.log10(100.), 21), bias_range=(8,-5), nphotslices=11):
    """
    Returns the theory prediction for the linear angular correlation function, given a spectroscopic galaxy sample
    and an input TNG `sim` object.
    """

    # comoving distance to the box center
    r = tools.redshift_to_comov(sim.redshift)

    def ell_to_k(ell):
        return (ell + 0.5) / r.value
    def k_to_ell(k):
        return (k * r.value) - 0.5

    # linear angular power spectrum
    colcosmo = cosmology.setCosmology('planck15')
    P = colcosmo.matterPowerSpectrum(ell_to_k(ell), sim.redshift)

    # linear bias from spectroscopic sample
    b_spec = get_linear_bias(gal_pos_spec, sim, nx=nx, r_edges=r_edges, bias_range=bias_range)

    # photometric weights dN/dchi
    W_phot = get_photometric_weights(gal_pos_phot, boxsize=sim.boxsize.value, nslices=nphotslices)

    # angular power spectrum in each LOS bin
    prefactor = 1 / r**2 * b_spec**2 * P
    C_ells = np.array([
        prefactor * W_phot[i] for i in range(nphotslices)
    ])

    # integrate the power spectra to get the angular correlation function in each LOS bin
    wlin_pred = np.zeros((nphotslices,len(thetaavg)))
    for i, Cell in enumerate(C_ells):
        wlin_pred[i] = np.array([
            powerspec_to_wlin(theta_, ell, Cell)[0] for theta_ in np.deg2rad(thetaavg)
        ])
    return wlin_pred


def get_linear_bias(gal_pos, sim, nx=500,
                r_edges=np.logspace(np.log10(1), np.log10(100.), 21), bias_range=(8,-5)):
    """
    Returns the linear bias given a galaxy sample and an input TNG `sim` object.
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
    
    return bias



def get_photometric_weights(gal_pos_phot, boxsize, nslices=11):
    """
    Returns the weights dN/dchi for an input photometric sample, used to calculate the predicted
    cross correlation function from linear theory.
    """
    L = boxsize.value if isinstance(boxsize, u.Quantity) else boxsize
    gal_pos_phot = gal_pos_phot.value if isinstance(gal_pos_phot, u.Quantity) else gal_pos_phot
    # divide sample into bins along the LOS
    losbins = np.linspace(-L/2, L/2, nslices+1)
    slicewidth = (losbins[1]-losbins[0])
    slice_centers = np.array([
        (losbins[i]+losbins[i+1])/2 for i in range(nslices)
    ]) << u.Mpc / cu.littleh
    
    # photometric
    slices_phot = [
        gal_pos_phot[(losbins[i] <= gal_pos_phot[:,2]) & (gal_pos_phot[:,2] < losbins[i+1])] << u.Mpc / cu.littleh \
        for i in range(nslices)
    ]
    Nphot = len(gal_pos_phot)
    dNdchi = np.array([len(x) for x in slices_phot]) / slicewidth
    W_phot = 1 / Nphot * dNdchi
    
    return W_phot


def powerspec_to_wlin(theta, ell, Cell):
    """
    Transform an input matter power spectrum Cell to the real-space linear angular correlation function w(theta).
    """
    
    assert ell.ndim == Cell.ndim == 1, "input ell and Cell must be 1D"

    # function of ell that we want to integrate
    def ell_func(ell_, Cell_):
        return (ell_ / (2 * np.pi)) * Cell_ * (special.jv(0, ell_ * theta))

    # construct our array, and integrate using trapezoid rule
    ell_func_arr = np.array([
        ell / (2 * np.pi) * Cell * special.jv(0, ell * theta)
    ]).flatten()
    trapz = integrate.trapz(ell_func_arr, x=ell)

    return trapz, ell_func_arr