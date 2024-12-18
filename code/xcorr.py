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
        
        # get the redshift and comoving distance (chi) to the center of each snapshot
        self._get_snapshot_info()

        # sort snapshots in decreasing order and redshifts in increasing order,
        #   to deal with interpolation + integration schemes in other functions
        self._sort_snapshots()

        # separation bins
        self.rp_edges = np.logspace(np.log10(self.rpmin), np.log10(self.rpmax), self.nrpbins+1)
        self.rp_avg = 0.5 * (self.rp_edges[1:] + self.rp_edges[:-1])
        self.r_edges = np.logspace(np.log10(self.rmin), np.log10(self.rmax), self.nbins+1)
        self.r_avg = 0.5 * (self.r_edges[1:] + self.r_edges[:-1])
        # pimax, now that we have the boxsize attribute
        self.pimax = int(self.pimax_frac * self.boxsize.value)

    
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

    def _fetch_gal_pos_phots(self, gal_pos_phots):
        if gal_pos_phots is None:
            if hasattr(self, 'gal_pos_phots'):
                return self.gal_pos_phots
            else:
                raise AttributeError("no gal_pos_phots attribute, and no gal_pos_phots input")
        else:
            return gal_pos_phots

    """
    Correlation functions
    """
    ###
    # PAIR COUNTS: require input spectroscopic (and/or photometric) galaxy samples
    ###
    def compute_auto_xis(self, gal_pos=None, verbose=False):
        """
        Computes the 3D auto-correlation from a set of galaxy positions in each snapshot.
        """

        # compute the pair counts
        xis = np.full((len(self.snapshots), self.nbins), np.nan)
        for i, gal_pos_arr in enumerate(gal_pos):
            if verbose == True:
                end = '\n' if i == len(gal_pos)-1 else '\r'
                print(f"computing 3D autocorr. from pair counts:\t{i+1} of {len(self.snapshots)} (z={redshift:.2f})",
                        end=end)
            r_avg, xis[i] = corrfuncs.compute_xi_auto(gal_pos_arr,
                                    self.rmin, self.rmax, self.nbins,
                                    randmult=self.randmult, boxsize=self.boxsize, logbins=True,
                                    nrepeats=self.nrepeats, periodic=self.periodic)
        assert np.all(r_avg == self.r_avg)

        return xis

    def compute_auto_wps(self, gal_pos=None, verbose=False):
        """
        Computes the projected auto-correlation from a set of galaxy positions in each snapshot.
        """

        if gal_pos is not None:
            assert len(gal_pos) == len(self.snapshots), \
                "length input gal_pos must equal the number of snapshots"

        # compute and store wp(rp) from each reference galaxy sample
        wps = np.full((len(self.snapshots), self.nrpbins), np.nan)
        for i, gal_pos_arr in enumerate(gal_pos):
            if verbose == True:
                end = '\n' if i == len(gal_pos)-1 else '\r'
                print(f"computing projected autocorr. from pair counts:\t{i+1} of {len(self.snapshots)} (z={self.redshifts[i]:.2f})",
                        end=end)
            rp_avg, wps[i] = corrfuncs.compute_wp_auto(gal_pos_arr,
                                    self.rpmin, self.rpmax, self.nrpbins, self.pimax,
                                    randmult=self.randmult, boxsize=self.boxsize, logbins=True,
                                    nrepeats=self.nrepeats, periodic=self.periodic)
        assert np.all(rp_avg == self.rp_avg)

        return wps

    def compute_wpx(self, gal_pos_phots=None, gal_pos_specs=None, verbose=False):
        """
        Computes the projected cross-correlation between the photometric and spectroscopic galaxy
        positions in each snapshot.
        """

        gal_pos_phots = self._fetch_gal_pos_phots(gal_pos_phots)
        gal_pos_specs = self._fetch_gal_pos_specs(gal_pos_specs)

        # compute nad store wp(rp) in each snapshot
        wpxs = np.full((len(self.snapshots), self.nrpbins), np.nan)
        for i in range(len(self.snapshots)):
            if verbose == True:
                end = '\n' if i == len(gal_pos_phots)-1 else '\r'
                print(f"computing projected cross-corr. from pair counts:\t{i+1} of {len(self.snapshots)} (z={self.redshifts[i]:.2f})",
                        end=end)
            rp_avg, wpxs[i] = corrfuncs.compute_wp_cross(gal_pos_phots[i], gal_pos_specs[i],
                                                        self.rpmin, self.rpmax, self.nrpbins, self.pimax,
                                                        randmult=self.randmult, boxsize=self.boxsize, logbins=True,
                                                        nrepeats=self.nrepeats, periodic=self.periodic)
        assert np.all(rp_avg == self.rp_avg)

        return wpxs

    # specific functions to add attributes for spectroscopic and photometric samples

    def compute_xis_specs(self, gal_pos_specs=None, verbose=False):
        gal_pos_specs = self._fetch_gal_pos_specs(gal_pos_specs)
        
        self.xis_specs = self.compute_auto_xis(gal_pos_specs, verbose=verbose)

    def compute_xis_phots(self, gal_pos_phots=None, verbose=False):
        gal_pos_phots = self._fetch_gal_pos_phots(gal_pos_phots)

        self.xis_phots = self.compute_auto_xis(gal_pos_phots, verbose=verbose)

    def compute_wps_specs(self, gal_pos_specs=None, verbose=False):
        gal_pos_specs = self._fetch_gal_pos_specs(gal_pos_specs)
        
        self.wps_specs = self.compute_auto_xis(gal_pos_specs, verbose=verbose)

    def compute_wps_phots(self, gal_pos_phots=None, verbose=False):
        gal_pos_phots = self._fetch_gal_pos_phots(gal_pos_phots)

        self.wps_phots = self.compute_auto_xis(gal_pos_phots, verbose=verbose)

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

    def compute_wthetax_from_wps_pair_counts(self, gal_pos_phots=None, gal_pos_specs=None, cross=False, verbose=True):

        assert hasattr(self, 'dNdz'), \
            "must set a photometric distribution dNdz before computing cross-correlation"

        gal_pos_specs = self._fetch_gal_pos_specs(gal_pos_specs)
        # if we want to compute the projected CROSS-correlation, also get the photometric sample
        if cross == True:
            gal_pos_phots = self._fetch_gal_pos_phots(gal_pos_phots)
            wps = self.compute_wpx(gal_pos_phots, gal_pos_specs, verbose=verbose)
        # otherwise, compute the projected autocorrelation from the spectroscopic sample
        else:
            wps = self.compute_auto_xis(gal_pos_specs, verbose=verbose)
        
        self.wthetax = np.array([
            self.dNdz[i] * self.wps[i] for i in range(len(self.snapshots))
        ])

    # def compute_wthetax_from_pair_counts(self, gal_pos_phots=None, gal_pos_specs=None, verbose=True):

    #     gal_pos_phots = self._fetch_gal_pos_phots(gal_pos_phots)
    #     gal_pos_specs = self._fetch_gal_pos_specs(gal_pos_specs)

    #     theta_avg = np.full((len(self.snapshots), self.nbins), np.nan)
    #     wthetax = np.copy(theta_avg)
    #     for i, snapshot in enumerate(self.snapshots):

    #         if verbose:
    #             print(f"snapshot {i+1} of {len(self.snapshots)}: prepping data for corrfunc computation", flush=True)
    #         # set up for c.f. computation (mainly to get random set)
    #         dataforcf = corrfuncs.set_up_cf_data(gal_pos_phots[i], self.randmult, self.rpmin, self.rpmax, self.nbins,
    #                                 data2=gal_pos_specs[i], boxsize=self.boxsize, logbins=True)
    #         rp_edges, rp_avg, nd1, nd2, boxsize, nr, rand_set, d1_set, d2_set = dataforcf.values()
    #         assert np.allclose(rp_edges, self.rp_edges)

    #         # convert bin edges to angles
    #         theta_edges = tools.r_comov_to_theta(rp_edges, self.redshifts[i]).value

    #         # convert (x,y,z) -> (RA, Dec)
    #         ra_phot, dec_phot = tools.get_ra_dec(gal_pos_phots[i], self.chis[i])
    #         ra_spec, dec_spec = tools.get_ra_dec(gal_pos_specs[i], self.chis[i])
    #         # convert random (x,y,z) -> (RA, Dec)
    #         ra_rand, dec_rand = tools.get_ra_dec(rand_set, self.chis[i])

    #         if verbose:
    #             print(f"snapshot {i+1} of {len(self.snapshots)}: computing angular xcorr from pair counts", flush=True)
    #         theta_avg[i], wthetax[i] = corrfuncs.wtheta_cross_PH(ra_phot, dec_phot,
    #                                      ra_spec, dec_spec, ra_rand, dec_rand, theta_edges)
    #     self.theta_avg = theta_avg
    #     self.wthetax = wthetax


    ###
    # LINEAR THEORY
    ###
    def compute_xis_linear_theory(self, gal_pos_phots=None, gal_pos_specs=None, cross=False,
                                    matter_cf_type='linear', dm_subsample=1000, verbose=False):
        """
        Computes the 3D auto-correlation from linear theory.
        """

        # fetch the spectroscopic galaxies, since we need them no matter what
        gal_pos_specs = self._fetch_gal_pos_specs(gal_pos_specs)
        # compute the bias in the spectroscopic sample
        bias_spec = self.compute_bias(gal_pos_specs)['biases']

        # if we want to compute the cross-correlation between the photometric and spectroscopic samples,
        #   then we need the bias in the spectroscopic _and_ photometric samples
        if cross == True:
            gal_pos_phots = self._fetch_gal_pos_phots(gal_pos_phots)
            bias_phot = self.compute_bias(gal_pos_phots)['biases']
            # the bias term is the photometric bias * the spectroscopic bias
            bias_term = bias_phot * bias_spec
        else:
            # if we're just using the spectroscopic sample, the bias term is the spectroscopic bias squared
            bias_term = bias_spec**2

        # galaxy c.f. is the matter c.f. times the bias squared
        if matter_cf_type.lower() == 'linear':
            self.matter_cf_type = 'linear'
            self.xi_lins = np.array([
                bias_term[i] * tools.linear_2pcf(redshift, self.r_avg) \
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
                bias_term[i] * self.xi_dms[i] for i in range(len(self.snapshots))
            ])
    
    def compute_wps_linear_theory(self, gal_pos_phots=None, gal_pos_specs=None, cross=False,
                                    matter_cf_type='linear', verbose=False):
        """
        Computes the projected auto-correlation from the linear theory 3D auto-correlation
        (hence assumes isotropy).
        """

        xis = self.compute_xis_linear_theory(gal_pos_phots, gal_pos_specs, cross,
                                                matter_cf_type=matter_cf_type, verbose=verbose)
        
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
        self.wp_lins = wp_lins

    def compute_wthetax_linear_theory(self, gal_pos_phots=None, gal_pos_specs=None, cross=False,
                                        matter_cf_type='linear', verbose=True):

        assert hasattr(self, 'dNdz'), \
            "must set a photometric distribution dNdz before computing cross-correlation"

        # compute the projected correlation function from linear theory
        self.compute_wps_linear_theory(gal_pos_phots=gal_pos_phots, gal_pos_specs=gal_pos_specs, cross=cross,
                                        matter_cf_type=matter_cf_type, verbose=verbose)
        
        # weight the result in each snapshot by the corresponding dNdz
        self.wthetax_lin = np.array([
            self.dNdz[i] * self.wp_lins[i] for i in range(len(self.snapshots))
        ])
    

    """
    Galaxy bias
    """
    def calculate_bias(self, gal_pos, method='auto_gal', rmin=1, rmax=100, nbins=10,
                        r_range=(5, 30), verbose=False):

        assert len(gal_pos) == len(self.snapshots)
        assert method in ['auto_gal', 'cross_dm']

        # get linear bias from spectroscopic sample
        biases_r = np.empty((len(self.redshifts), self.nbins))
        for i, redshift in enumerate(self.redshifts):
            if verbose == True:
                end = '\n' if i == len(self.redshifts)-1 else '\r'
                print(f"calculating bias:\t{i+1} of {len(self.redshifts)} (z={redshift:.2f})", end=end)
            r_avg, biases_r[i] = linear_theory.get_linear_bias(gal_pos[i],
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
    return all(x < y for x, y in zip(L, L[1:]))

def strictly_decreasing(L):
    return all(x > y for x, y in zip(L, L[1:]))