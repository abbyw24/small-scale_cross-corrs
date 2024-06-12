"""
Functions to construct a 2D photometric sample from a spectroscopic sample constructed from TNG,
compute their cross correlations, and compare to the linear theory prediction.
"""

import numpy as np
import astropy.units as u
from astropy.cosmology import Planck15 as cosmo
from colossus.cosmology import cosmology
from scipy import special, integrate
import os
import sys

from illustris_sim import TNGSim
from linear_theory import get_linear_bias, powerspec_to_wlin
import corrfuncs
import tools


def construct_photometric_sample(gal_pos_spec, dx, mean=None):
    """
    Construct a mock photometric galaxy sample with a Gaussian LOS distribution,
    given input spectroscopic galaxy positions and width of the Gaussian `dx` in comoving units.
    If `mean == None`, takes the mean position of `gal_pos_spec` along the LOS (assumed z component).
    """
    gal_pos_phot = []
    if mean is None:
        mean = np.nanmean(gal_pos_spec[:,2])
    for i, pos in enumerate(gal_pos_spec):
        draw = tools.eval_Gaussian(pos[2], dx, mean=mean)
        # normalize to get fraction of the Gaussian's peak
        draw *= (dx * np.sqrt(2 * np.pi))
        if draw > np.random.uniform():
            gal_pos_phot.append(pos)
        else:
            continue
    return np.array(gal_pos_phot) << dx.unit


def compute_wtheta_photxspec_single_snapshot(gal_pos_spec, redshift, sigma_z, boxsize, losbins, theta_edges):
    
    # prep inputs
    gal_pos_spec = gal_pos_spec.to(u.Mpc/u.littleh) if isinstance(gal_pos_spec, u.Quantity) \
                    else gal_pos_spec * u.Mpc/u.littleh
    boxsize = boxsize.to(u.Mpc/u.littleh) if isinstance(boxsize, u.Quantity) else boxsize * u.Mpc/u.littleh
    losbins = losbins.to(u.Mpc/u.littleh) if isinstance(losbins, u.Quantity) else losbins * u.Mpc/u.littleh
    theta_edges = theta_edges.to(u.deg) if isinstance(theta_edges, u.Quantity) else theta_edges * u.deg

    # width of Gaussian for constructing the photometric sample
    dx = tools.get_dx(redshift, sigma_z).to(boxsize.unit)
    # comoving distance to the box center
    chi = tools.redshift_to_comov(redshift).to(dx.unit)
    
    # divide into narrow 2D slices along the LOS
    nslices = len(losbins)-1
    slicewidth = (losbins[1]-losbins[0])
    slice_centers = np.array([(losbins[i]+losbins[i+1]).value/2 for i in range(nslices)])
    
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
        construct_photometric_sample(gal_pos_spec_ << dx.unit, dx) for gal_pos_spec_ in gal_pos_specs
    ]

    # compute the cross correlation in each slice in each sample and take the average to reduce noise
    xcorrs = np.empty((nslices, nslices, len(theta_edges)-1))
    for i, spec_sample in enumerate(gal_pos_specs):

        # divide the spectroscopic sample and corresponding photometric sample into slices
        slices_spec = [
            spec_sample[(losbins.value[i] <= spec_sample[:,2]) & (spec_sample[:,2] < losbins.value[i+1])] \
            for i in range(nslices)
        ]
            
        # convert photometric sample to (RA,Dec), setting LOS positions to box center
        phot_sample = np.copy(gal_pos_phots[i])
        phot_sample[:,2] = 0
        ra_phot, dec_phot = tools.CartesiantoEquatorial(phot_sample, observer=[0.,0.,chi.value])
        # random sample
        nd1 = len(ra_phot)
        ra_rand_phot = np.random.uniform(min(ra_phot.to(u.deg).value), max(ra_phot.to(u.deg).value), nd1)
        dec_rand_phot = np.random.uniform(min(dec_phot.to(u.deg).value), max(dec_phot.to(u.deg).value), nd1)

        for j, spec_slice in enumerate(slices_spec):
            # prep spectroscopic slice and corresponding random set
            nd2 = len(spec_slice)
            gal_pos_ = np.copy(spec_slice)
            gal_pos_[:,2] = 0
            ra_spec_, dec_spec_ = tools.CartesiantoEquatorial(gal_pos_, observer=[0.,0.,chi.value])

            # compute xcorr
            thetaavg, xcorrs[i,j] = corrfuncs.wtheta_cross_PH(ra_phot, dec_phot, ra_spec_, dec_spec_,
                                              ra_rand_phot, dec_rand_phot, theta_edges)
            
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


def construct_spherex_galaxy_samples(snapshots, sigma_z, ns=None, verbose=True):

    # check inputs
    if ns is not None:
        assert len(ns) == len(snapshots), "number of input target number densities must match number of snapshots"

    # load the snapshots and construct the spectroscopic galaxy samples
    redshifts = np.empty(len(snapshots)) # redshift of each snapshot
    chis = np.empty(len(snapshots)) # comoving distance to the center of each snapshot
    gal_pos_specs = [] # where to store galaxy positions
    for i, snapshot in enumerate(snapshots):
        sim = TNGSim('TNG300-3', snapshot=snapshot)
        chi = tools.redshift_to_comov(sim.redshift)
        redshifts[i] = sim.redshift
        chis[i] = chi.value
        n = ns[i] if ns is not None else None
        
        gal_pos_spec = sim.subhalo_pos()[sim.gal_idx('','SPHEREx', sigma_z=sigma_z, n=n, verbose=verbose)]
        gal_pos_spec = tools.remove_values(gal_pos_spec, minimum=0, maximum=sim.boxsize)
        gal_pos_spec -= sim.boxsize / 2  # center at zero
        assert np.all(gal_pos_spec >= -sim.boxsize / 2) and np.all(gal_pos_spec <= sim.boxsize / 2), \
            f"galaxy positions out of bounds! min = {np.nanmin(gal_pos_spec):.3f}, max = {np.nanmax(gal_pos_spec):.3f}"
        gal_pos_specs.append(gal_pos_spec)
    chis *= chi.unit  # give unit back to the chis
    if verbose:
        print(f"mean redshift of these {len(snapshots)} snapshots is {np.mean(redshifts):.2f}")
    
    res = dict(redshifts=redshifts, chis=chis, gal_pos_specs=gal_pos_specs, boxsize=sim.boxsize)
    return res





def Cellx_theory(gal_pos_photometric, gal_pos_spectroscopic, sim, thetaavg, losbins, ell=np.logspace(0, 6, 1000), nx=500,
                r_edges=np.logspace(np.log10(1), np.log10(100.), 21), b_spec=None, bias_range=(8,-5)):
    """
    Returns the theory prediction for the linear angular cross power spectrum, given photometric and spectroscopic
    galaxy samples and an input TNG `sim` object.

    """

    gal_pos_phot = np.copy(gal_pos_photometric)
    gal_pos_spec = np.copy(gal_pos_spectroscopic)
    nphotslices = len(losbins)-1

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
    if b_spec is None:
        b_spec = get_linear_bias(gal_pos_spec, sim, nx=nx, r_edges=r_edges, bias_range=bias_range)
        print(f"bias: ", b_spec)

    # photometric weights dN/dchi
    W_phot = get_photometric_weights(gal_pos_phot, boxsize=sim.boxsize, losbins=losbins)

    # angular power spectrum in each LOS bin
    prefactor = 1 / r**2 * b_spec**2 * P
    C_ells = np.array([
        prefactor * W_phot[i] for i in range(nphotslices)
    ])

    return C_ells


def wlin_photxspec_theory(gal_pos_photometric, gal_pos_spectroscopic, sim, thetaavg, losbins, ell=np.logspace(0, 6, 1000), nx=500,
                r_edges=np.logspace(np.log10(1), np.log10(100.), 21), b_spec=None, bias_range=(8,-5)):
    """
    Wrapper for Cellx_theory() that returns the real-space correlation function.

    """

    C_ells = Cellx_theory(gal_pos_photometric, gal_pos_spectroscopic, sim, thetaavg, losbins, ell=ell, nx=nx,
                r_edges=r_edges, b_spec=b_spec, bias_range=bias_range)

    # integrate the power spectra to get the angular correlation function in each LOS bin
    wlin_pred = np.zeros((len(C_ells),len(thetaavg)))
    for i, Cell in enumerate(C_ells):
        wlin_pred[i] = np.array([
            powerspec_to_wlin(theta_, ell, Cell)[0] for theta_ in thetaavg
        ])
    return wlin_pred, C_ells


def get_photometric_weights(gal_pos_phot, boxsize, losbins):
    """
    Returns the weights dN/dchi for an input photometric sample, used to calculate the predicted
    cross correlation function from linear theory.
    """
    gal_pos_phot = gal_pos_phot.value if isinstance(gal_pos_phot, u.Quantity) else gal_pos_phot
    L = boxsize.value if isinstance(boxsize, u.Quantity) else boxsize
    losbins = losbins.value if isinstance(losbins, u.Quantity) else losbins
    # divide sample into bins along the LOS
    nslices = len(losbins)-1
    slicewidth = (losbins[1]-losbins[0])
    slice_centers = np.array([
        (losbins[i]+losbins[i+1])/2 for i in range(nslices)
    ]) << u.Mpc / u.littleh
    
    # photometric
    slices_phot = [
        gal_pos_phot[(losbins[i] <= gal_pos_phot[:,2]) & (gal_pos_phot[:,2] < losbins[i+1])] << u.Mpc / u.littleh \
        for i in range(nslices)
    ]
    Nphot = len(gal_pos_phot)
    dNdchi = np.array([len(x) for x in slices_phot]) / slicewidth
    W_phot = 1 / Nphot * dNdchi
    
    return W_phot