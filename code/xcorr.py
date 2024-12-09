import numpy as np
from scipy import integrate
import time
import os
import sys
import astropy.units as u

from illustris_sim import TNGSim
import corrfuncs
import linear_theory
import tools


class Xcorr():
    """
    Class to manage galaxy and dark matter auto- and cross-correlations
    from IllustrisTNG and linear theory.
    * Does not compute spectroscopic galaxy samples but involves methods which take
    them as input.
    """
    def __init__(self, snapshots, dNdz,
                    rpmin=0.1, rpmax=60, nrpbins=10,    # bins for projected c.f.
                    pimax_frac=0.45,                    # fraction of the boxsize for pimax
                    rmin=0.1, rmax=100, nbins=10,       # bins for full 3D c.f.
                    nrepeats=10, periodic=True, randmult=3,   # other inputs for both projected and full c.f.
                    sim='TNG300-3',
                    scratch='/scratch1/08811/aew492'):

        # leaving flexibility for input dNdz to be None, because some child classes need to instantiate
        #   this class first in order to calculate the correct dNdz (e.g. as a function of redshift)

        # check inputs
        assert len(snapshots) > 1, "must input multiple snapshots"
        assert type(nrpbins) == int
        assert type(nbins) == int
        assert type(nrepeats) == int
        assert type(periodic) == bool
        assert type(randmult) == int
        assert type(sim) == str
        assert type(scratch) == str

        # inputs
        self.snapshots = snapshots.astype(int)
        self.rpmin = rpmin
        self.rpmax = rpmax 
        self.nrpbins = nrpbins
        self.pimax_frac = pimax_frac
        self.rmin = rmin
        self.rmax = rmax
        self.nbins = nbins
        self.nrepeats = nrepeats
        self.periodic = periodic
        self.randmult = randmult
        self.sim = sim
        self.scratch = scratch

        if dNdz is not None:
            self.set_dNdz(dNdz)

        # separation bins
        self.rp_edges = np.logspace(np.log10(self.rpmin), np.log10(self.rpmax), self.nrpbins+1)
        self.rp_avg = 0.5 * (self.rp_edges[1:] + self.rp_edges[:-1])
        self.r_edges = np.logspace(np.log10(self.rmin), np.log10(self.rmax), self.nbins+1)
        self.r_avg = 0.5 * (self.r_edges[1:] + self.r_edges[:-1])

        # get the redshift and comoving distance (chi) to the center of each snapshot
        self._get_snapshot_info()

        # sort snapshots in decreasing order and redshifts in increasing order,
        #   to deal with interpolation + integration schemes in other functions
        self._sort_snapshots()

    
    def set_dNdz(self, dNdz):
        assert len(self.snapshots) == len(dNdz), "length of dNdz must match number of snapshots"
        self.dNdz = dNdz

    def _get_snapshot_info(self):
        """
        Get information associated with each TNG snapshot.
        """
        redshifts = np.empty(len(self.snapshots))
        chis = np.empty(len(self.snapshots))
        for i, snapshot in enumerate(self.snapshots):
            sim = TNGSim(self.sim, snapshot=snapshot)
            redshifts[i] = sim.redshift
            chis[i] = tools.redshift_to_comov(sim.redshift).value
        self.redshifts = redshifts
        self.chis = chis * tools.redshift_to_comov(sim.redshift).unit # and give units back to chis
        self.central_chi = np.nanmean(self.chis)
        # also save other simulation info
        self.boxsize = sim.boxsize
        self.sim_basepath = sim.basepath

    def _sort_snapshots(self):

        # check that the input snapshots are monotonically increasing or decreasing
        if strictly_increasing(self.snapshots):
            assert strictly_decreasing(self.redshifts), "redshifts should be strictly decreasing if snapshots are strictly increasing"
        else:
            assert strictly_decreasing(self.snapshots), "input snapshots must be strictly increasing or strictly decreasing"
            assert strictly_increasing(self.redshifts), "redshifts should be strictly increasing if snapshots are strictly decreasing"

        self.snapshots = self.snapshots[::-1]
        self.redshifts = self.redshifts[::-1]
        self.chis = self.chis[::-1]
    
    def _fetch_gal_pos_specs(self, gal_pos_specs):
        if gal_pos_specs is None:
            if hasattr(self, 'gal_pos_specs'):
                return self.gal_pos_specs
            else:
                raise AttributeError("no gal_pos_specs attribute, and no gal_pos_specs input")
        else:
            return gal_pos_specs


    """
    Correlation functions
    """
    ###
    # PAIR COUNTS: require input spectroscopic galaxy samples
    ###
    def compute_wps_pair_counts(self, gal_pos_specs=None, verbose=False):
        """
        Computes the projected auto-correlation from input spectroscopic galaxies in each snapshot.
        """
        gal_pos_specs = self._fetch_gal_pos_specs(gal_pos_specs)

        assert len(gal_pos_specs) == len(self.snapshots), \
            "length input gal_pos_specs must equal the number of snapshots"

        # pimax, now that we have the boxsize attribute
        self.pimax = int(self.pimax_frac * self.boxsize.value)

        # compute and store wp(rp) from each reference galaxy sample
        wps = np.full((len(self.snapshots), self.nrpbins), np.nan)
        for i, gal_pos_spec in enumerate(gal_pos_specs):
            if verbose == True:
                end = '\n' if i == len(gal_pos_specs)-1 else '\r'
                print(f"computing projected autocorr. from pair counts:\t{i+1} of {len(self.snapshots)} (z={self.redshifts[i]:.2f})",
                        end=end)
            rp_avg, wps[i] = corrfuncs.compute_wp_auto(gal_pos_spec,
                                    self.rpmin, self.rpmax, self.nrpbins, self.pimax,
                                    randmult=self.randmult, boxsize=self.boxsize, logbins=True,
                                    nrepeats=self.nrepeats, periodic=self.periodic)
        assert np.all(rp_avg == self.rp_avg)
        self.theta_avg = self.r_comov_to_theta(self.rp_avg)
        self.wps = wps

    def compute_xis_pair_counts(self, gal_pos_specs=None, verbose=False):
        """
        Computes the 3D auto-correlation from the spectroscopic galaxies in each snapshot.
        """
        gal_pos_specs = self._fetch_gal_pos_specs(gal_pos_specs)

        # compute the pair counts
        xis = np.full((len(self.snapshots), self.nbins), np.nan)
        for i, gal_pos_spec in enumerate(gal_pos_specs):
            if verbose == True:
                end = '\n' if i == len(gal_pos_specs)-1 else '\r'
                print(f"computing 3D autocorr. from pair counts:\t{i+1} of {len(self.snapshots)} (z={redshift:.2f})",
                        end=end)
            r_avg, xis[i] = corrfuncs.compute_xi_auto(gal_pos_spec,
                                    self.rmin, self.rmax, self.nbins,
                                    randmult=self.randmult, boxsize=self.boxsize, logbins=True,
                                    nrepeats=self.nrepeats, periodic=self.periodic)
        assert np.all(r_avg == self.r_avg)
        self.xis = xis
    
    def compute_xis_dark_matter(self, subsample=1, verbose=False):

        xi_dms = np.full((len(self.snapshots), self.nbins), np.nan)
        for i, snapshot in enumerate(self.snapshots):
            sim = TNGSim(self.sim, snapshot=snapshot)
            dm_pos = tools.get_subsample(sim.dm_pos(verbose=verbose), verbose=verbose)
            r_avg, xi_dms[i] = corrfuncs.compute_xi_auto(dm_pos,
                                    self.rmin, self.rmax, self.nbins,
                                    randmult=self.randmult, boxsize=self.boxsize, logbins=True,
                                    nrepeats=self.nrepeats, periodic=self.periodic)
        assert np.all(r_avg == self.r_avg)
        self.xi_dms = xi_dms

    def compute_angular_xcorrs_pair_counts(self, gal_pos_specs=None, verbose=True):

        assert hasattr(self, 'dNdz'), \
            "must set a photometric distribution dNdz before computing cross-correlation"

        gal_pos_specs = self._fetch_gal_pos_specs(gal_pos_specs)

        # compute necessary things if they haven't been already!
        if not hasattr(self, 'wps'):
            self.compute_wps_pair_counts(gal_pos_specs, verbose=verbose)
        
        self.wthetax = np.array([
            self.dNdz[i] * self.wps[i] for i in range(len(self.snapshots))
        ])

    ###
    # LINEAR THEORY
    ###
    def compute_xis_linear_theory(self, bias_fn=None, gal_pos_specs=None, matter_cf_type='linear',
                                    dm_subsample=1000, verbose=False):
        """
        Computes the 3D auto-correlation from linear theory.
        """

        self.get_bias(bias_fn, gal_pos_specs, verbose=verbose)

        # galaxy c.f. is the matter c.f. times the bias squared
        if matter_cf_type.lower() == 'linear':
            self.matter_cf_type = 'linear'
            self.xi_lins = np.array([
                self.biases[i] * tools.linear_2pcf(redshift, self.r_avg) \
                    for i, redshift in enumerate(self.redshifts)
            ])
        else:
            assert matter_cf_type.lower() == 'dm', \
                "matter_cf_type must be 'linear' or 'dm'"
            self.matter_cf_type = 'dm'
            self.dm_subsample = int(dm_subsample)
            if verbose == True:
                print("computing autocorr. from dark matter particles")
            self.compute_xis_dark_matter()
            self.xi_lins = np.array([
                self.biases[i] * self.xi_dms[i] for i in range(len(self.snapshots))
            ])
    
    def compute_wps_linear_theory(self, gal_pos_specs=None, matter_cf_type='linear', verbose=False):
        """
        Computes the projected auto-correlation from the linear theory 3D auto-correlation
        (hence assumes isotropy).
        """
        if verbose == True:
            print("computing projected autocorr. from linear theory")

        self.compute_xis_linear_theory(gal_pos_specs=gal_pos_specs, matter_cf_type=matter_cf_type, verbose=verbose)
        
        # populate 2D array of separation r from each (r_p, pi) pair
        rp = self.rp_avg
        pi = self.rp_avg
        r_arr = tools.r_from_rppi(rp, pi)

        # linear wp(rp) from linear xi(r)
        wp_lins = np.full((len(self.snapshots), self.nrpbins), np.nan)
        for i, xi_lin in enumerate(self.xi_lins):
            # interpolate xi(r) on this r_arr grid
            xi_lin_arr = np.exp(np.interp(np.log(r_arr), np.log(self.r_avg), np.log(xi_lin)))
            # sum over pi / r_parallel to get 1D wp(rp)
            wp_lins[i] = 2.0 * integrate.trapz(xi_lin_arr, x=pi, axis=0) 
        self.theta_avg = self.r_comov_to_theta(self.rp_avg)
        self.wp_lins = wp_lins

    def compute_angular_xcorrs_linear_theory(self, gal_pos_specs=None, matter_cf_type='linear', verbose=True):

        assert hasattr(self, 'dNdz'), \
            "must set a photometric distribution dNdz before computing cross-correlation"

        # compute necessary things if they haven't been already!
        if not hasattr(self, 'wp_lins') or gal_pos_specs is not None:
            self.compute_wps_linear_theory(gal_pos_specs=gal_pos_specs, matter_cf_type=matter_cf_type, verbose=verbose)
        
        self.wthetax_lin = np.array([
            self.dNdz[i] * self.wp_lins[i] for i in range(len(self.snapshots))
        ])

    ###
    # ANGULAR CROSS-CORRELATIONS from both pair counts and linear theory
    ###
    def compute_angular_xcorrs(self, gal_pos_specs=None, verbose=False):

        gal_pos_specs = self._fetch_gal_pos_specs(gal_pos_specs)

        print(f"computing angular xcorrs from pair counts")
        self.compute_angular_xcorrs_pair_counts(gal_pos_specs=gal_pos_specs, verbose=verbose)
        print(f"computing angular xcorrs from linear theory")
        self.compute_angular_xcorrs_linear_theory(gal_pos_specs=gal_pos_specs, verbose=verbose)
    

    """
    Galaxy bias
    """
    def get_bias(self, bias_fn=None, gal_pos_specs=None, method='auto_gal', rmin=1, rmax=100, nbins=10,
                        r_range=(5, 30), verbose=False):

        # if file is input, try to load the bias dictionary
        if bias_fn is not None:
            self.bias_dict = np.load(bias_fn, allow_pickle=True).item()
            self.bias_fn = bias_fn
        # otherwise calculate the bias from the input spectroscopic galaxies
        else:
            assert gal_pos_specs is not None, \
                "must input gal_pos_specs if input bias_fn is None"
            self.bias_dict = self.calculate_bias(gal_pos_specs, method=method,
                                                    rmin=rmin, rmax=rmax, nbins=nbins,
                                                    r_range=r_range, verbose=verbose)
        self.biases = self.bias_dict['biases']

    
    def calculate_bias(self, gal_pos_specs=None, method='auto_gal', rmin=1, rmax=100, nbins=10,
                        r_range=(5, 30), verbose=False):

        gal_pos_specs = self._fetch_gal_pos_specs(gal_pos_specs)

        assert method in ['auto_gal', 'cross_dm']

        # get linear bias from spectroscopic sample
        biases_r = np.empty((len(self.redshifts), self.nbins))
        for i, redshift in enumerate(self.redshifts):
            if verbose == True:
                end = '\n' if i == len(self.redshifts)-1 else '\r'
                print(f"calculating bias:\t{i+1} of {len(self.redshifts)} (z={redshift:.2f})", end=end)
            r_avg, biases_r[i] = linear_theory.get_linear_bias(gal_pos_specs[i],
                                        redshift=redshift, boxsize=self.boxsize, method=method,
                                        rmin=rmin, rmax=rmax, nbins=nbins,
                                        periodic=self.periodic)

        # get mean across a relatively flat range of scales
        r_idx = (r_avg > r_range[0]) & (r_avg < r_range[1])
        biases = np.array([
            np.mean(x[r_idx]) for x in biases_r
        ])

        # store parameters used in the bias calculation
        self.bias_method = method
        self.bias_rmin = rmin
        self.bias_rmax = rmax
        self.bias_nbins = nbins

        bias_dict = dict(redshifts = self.redshifts,
                            ns = self.ns,
                            r_avg = r_avg,
                            biases_r = biases_r,
                            r_range = r_range,
                            biases = biases)

        return bias_dict


    """
    Coordinate transformations
    """
    def theta_to_r_comov(self, theta):  # theta in DEGREES
        return tools.theta_to_r_comov(theta, np.mean(self.redshifts)).value
    def r_comov_to_theta(self, r):
        return tools.r_comov_to_theta(r, np.mean(self.redshifts)).value


# to check monotonicity:

def strictly_increasing(L):
    return all(x<y for x, y in zip(L, L[1:]))

def strictly_decreasing(L):
    return all(x>y for x, y in zip(L, L[1:]))