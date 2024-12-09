"""
Lookup table to convert redshifts <--> Illustris snapshot numbers.
"""

import numpy as np 
import illustris_python as il 
from astropy import table
import astropy.units as u
import astropy.cosmology.units as cu
from colossus.cosmology import cosmology
import os
import sys

from corrfuncs import compute_xi_cross
from survey_params_gal import eBOSS_param, DESI_param, SPHEREx_param
import tools

class TNGSim():
    """
    Class to manage Illustris-TNG simulations and data most relevant for testing small-scale cross-correlations.
    Works with a single snapshot/redshift at a time.
    """

    def __init__(self, sim, snapshot=None, redshift=None, scratch='/work2/08811/aew492/frontera'):

        # make sure we have passed either a snapshot or a redshift
        assert (snapshot is not None and redshift is None) or (snapshot is None and redshift is not None), \
            "must pass either snapshot or redshift"
        
        self.sim = str(sim)
        self.basepath = os.path.join(scratch, f'{sim}/output')
        if self.sim[:-2] == 'TNG300':
            self.snapshots = [0, 4, 17, # z to 3.0
                                25, 26, 27, 28, 29, 30, 31, 32, # 3.0 to 2.0
                                33, 34, 35, 36, 37, 38, 39, # 2.0 to 1.5
                                40, 41, 42, 43, 44, 45, 46, 47, 48, 49, # 1.5 to 1.0
                                50, 51, 52, 53, 54, 55, 56, 57, # 1.0 to 0.75
                                58, 59, 60, 61, 62, 63, 64, 65, 66, # 0.75 to 0.5
                                67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, # 0.5 to 0.25
                                81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 99]  # 0.25 to 0.0
            self.redshifts = [20.05, 10.00, 5.00,
                                3.01, 2.90, 2.73, 2.58, 2.44, 2.32, 2.21, 2.10,
                                2.00, 1.90, 1.82, 1.74, 1.67, 1.60, 1.53,
                                1.50, 1.41, 1.36, 1.30, 1.25, 1.21, 1.15, 1.11, 1.07, 1.04,
                                1.00, 0.95, 0.92, 0.89, 0.85, 0.82, 0.79, 0.76,
                                0.73, 0.70, 0.68, 0.64, 0.62, 0.60, 0.58, 0.55, 0.52,
                                0.50, 0.48, 0.46, 0.44, 0.42, 0.40, 0.38, 0.36, 0.35, 0.33, 0.31, 0.3, 0.27, 0.26,
                                0.24, 0.23, 0.21, 0.20, 0.18, 0.17, 0.15, 0.14, 0.13, 0.11, 0.10, 0.]
            self.boxsize = 205 * (u.Mpc / cu.littleh)
            assert len(self.snapshots) == len(self.redshifts)
        else:
            assert False, "unknown simulation"
        
        self._set_snapshot(snapshot, redshift)
        
        self.sim_tag = self.sim+f', z={self.redshift:.2f}'

    
    """ SNAPSHOTS <--> REDSHIFTS """

    def _set_snapshot(self, snapshot=None, redshift=None):
        """
        Set the simulation snapshot by either snapshot number or corresponding redshift.
        If nothing is passed, check that the simulation already has a `snapshot` attribute.
        """
        if snapshot:
            assert redshift is None
            assert snapshot in self.snapshots, f"snapshot {snapshot} not found"
            self.snapshot = snapshot
            self.redshift = self.redshifts[self.snapshots.index(snapshot)]
        else:
            assert snapshot is None
            assert redshift in self.redshifts, f"snapshot {snapshot} not found"
            self.redshift = redshift
            self.snapshot = self.snapshots[self.redshifts.index(redshift)]
    
    
    """
    GROUP CATALOGS:

        _load_subfind_subhalos() : loads the raw requested field data from the group catalog.

        [SOME_FIELD]() : returns the requested field with proper units.
    
    """

    def _load_subfind_subhalos(self, fields=['SubhaloPos','SubhaloMassType','SubhaloLenType','SubhaloSFR'],
                                remove_flagged=True, verbose=False):
        """
        Load the requested fields of this sim's Subfind subhalos (by default removing any flagged as non-cosmological in origin).
        """

        # get existing subhalo table or create new table
        subhalo_info = self.subhalo_info if hasattr(self, 'subhalo_info') else {}
        # load in any new fields (if we created a new dictionary this will be all the input fields),
        #   checking for flagged subhalos along the way
        fields_ = ['SubhaloFlag']
        for field in fields:
            if field not in subhalo_info.keys():
                fields_.append(field)
        # continue if we have any new fields to load:
        if len(fields_) > 1:
            subhalos = il.groupcat.loadSubhalos(self.basepath, self.snapshot, fields=list(fields_))
            # optionally, remove flagged subhalos:
            if remove_flagged:
                subhalo_idx = subhalos['SubhaloFlag']
                self.nsubhalos = np.sum(subhalo_idx)
                if verbose:
                    print(f"removed {np.sum(~subhalo_idx)} flagged subhalos")
            else:
                self.nsubhalos = len(subhalos['SubhaloFlag'])
                subhalo_idx = np.full(self.nsubhalos, True)
            if verbose:
                print(f"loaded the following fields for {self.sim_tag} ({self.nsubhalos} subhalos): \n\t", fields_[1:])
            # we need to remove the 'count' entry before we store data as a table, so each column has the same length
            del subhalos['count']
            if 'SubhaloFlag' in subhalo_info.keys():
                del subhalos['SubhaloFlag']
            subhalos = table.Table(subhalos)
            # append this new info to the table
            self.subhalo_info = table.hstack([subhalo_info, subhalos]) if hasattr(self, 'subhalo_info') else table.Table(subhalos)
        else:
            if verbose:
                print("requested fields already loaded!")
    
    def idx_nonzero(self):
        """
        Boolean array of subhalos (as returned in `_load_subfind_subhalos()`) with nonzero SFR and nonzero stellar mass.
        """
        return np.where((self.SFR() > 0) & (self.stellar_mass() > 0))[0]
    
    def subhalo_pos(self, unit=u.Mpc/cu.littleh):
        self._load_subfind_subhalos(fields=['SubhaloPos'])
        return (self.subhalo_info['SubhaloPos'] * u.kpc / cu.littleh).to(unit)
    
    def subhalo_mass(self):
        self._load_subfind_subhalos(fields=['SubhaloMass'])
        return self.subhalo_info['SubhaloMass'] * 1e10 * u.M_sun / cu.littleh # 4th col corresponds to star particles

    def stellar_mass(self):
        self._load_subfind_subhalos(fields=['SubhaloMassType'])
        return self.subhalo_info['SubhaloMassType'][:,4] * 1e10 * u.M_sun / cu.littleh # 4th col corresponds to star particles
    
    def SFR(self):
        self._load_subfind_subhalos(fields=['SubhaloSFR'])
        return self.subhalo_info['SubhaloSFR'] * u.M_sun / u.year
    
    def sSFR(self):
        """
        Get specific star formation rates of subhalos (star formation rate per stellar mass) (as returned in `_load_subfind_subhalos()`). \
        Returns zero where stellar mass is zero.
        """
        return np.divide(self.SFR(), self.stellar_mass(), out=np.zeros_like(self.SFR()), where=self.stellar_mass()!=0)
    

    """ SNAPSHOTS """

    def dm_pos(self, unit=u.Mpc/cu.littleh, verbose=True):
        """
        Load the (x,y,z) comoving coordinates of all DM particles.
        """
        dm_pos = il.snapshot.loadSubset(self.basepath, self.snapshot, 'dm', ['Coordinates']) * u.kpc / cu.littleh
        if verbose:
            print(f"loaded {len(dm_pos)} dark matter particles")
        return dm_pos.to(unit)
    

    """ SURVEY EMULATION """

    def survey_params(self, survey_name, tracer_name, sigma_z=None):
        """
        Calculate the target galaxy number density (function of redshift) for a specific survey, using survey_params_gal.py.
        """
        if survey_name == 'eBOSS':
            params = eBOSS_param(z=self.redshift, tracer_name=tracer_name)
        elif survey_name == 'DESI':
            params = DESI_param(z=self.redshift, tracer_name=tracer_name)
        elif survey_name == 'SPHEREx':
            params = SPHEREx_param(z=self.redshift, sigma_z=sigma_z)
        else:
            raise ValueError(f"{survey_name} is not a valid survey name")
        return params


    def target_N(self, tracer_name, survey='DESI', sigma_z=None, n=None, verbose=False):
        """
        Compute the target number of tracers from TNG boxsize and target number density.
        Target number density is computed as a function of tracer type ('LRG' or 'ELG') and survey, \
        using `survey_params()`, or passed as an optional argument.
        """
        # target number of galaxies = volume * number density
        V = self.boxsize**3
        if n is not None:
            self.n = n.to((cu.littleh / u.Mpc)**3) if hasattr(n, 'unit') else n * (cu.littleh / u.Mpc)**3
            if verbose:
                print(f"input number density: {self.n.value:.2e} (h/Mpc)^3")
        else:
            self.n = self.survey_params(survey, tracer_name, sigma_z).n_Mpc3 * (cu.littleh / u.Mpc)**3
            if verbose:
                print(f"{tracer_name} number density for {survey} at z={self.redshift}: {self.n.value:.2e} (h/Mpc)^3")
        if verbose:
            print(f"target number of subhalos: {int(V * self.n)}")
        return int(V * self.n)
    

    def idx_sSFR_cut(self, tracer_name, sSFR_cutval):
        """
        Return the indices of the subhalos (as returned in `_load_subfind_subhalos()`) that meet the criterion \
        log10(sSFR) >/< `sSFR_cutval`, with (>/<) dependent on `tracer_name`.
        """
        logsSFR = np.ma.log10(self.sSFR().value).filled(False)  # mask any invalid values, then fill mask with False
        if tracer_name == 'LRG':
            sSFR_cut = (logsSFR < sSFR_cutval)
        else:
            assert tracer_name == 'ELG', "'tracer_name' must be 'LRG' or 'ELG'"
            sSFR_cut = (logsSFR > sSFR_cutval)
        assert len(sSFR_cut) == len(self.subhalo_pos())
        return np.where(sSFR_cut)[0]


    def gal_idx(self, tracer_name, survey='DESI', sigma_z=None, sSFR_cutval=-9.09, n=None, verbose=False):
        """
        Make cuts in subhalo sSFR and stellar mass to select for LRGs/ELGs and reach a target galaxy number density,
        as outlined in Sullivan, Prijon, & Seljak (2023).
        Option to manually input target number density, otherwise compute as a function of `tracer_name` and `survey` \
        using survey_params().

        Sullivan et al. (2023) uses two different cutoff values for sSFR, log10(sSFR) =
            1. -9.09 (https://arxiv.org/abs/2210.10068 using MilleniumTNG)
            2. -9.23 (https://arxiv.org/abs/2011.05331 using TNG300-1)
        """

        # specific star formation rate (sSFR) = star formation rate per stellar mass
        if survey.upper()=='SPHEREX':
            sSFR_cut = np.arange(len(self.subhalo_pos()))
        else:
            sSFR_cut = self.idx_sSFR_cut(tracer_name, sSFR_cutval)  # length == nsubhalos

        # target number of galaxies = volume * number density
        target_N = self.target_N(tracer_name, survey, sigma_z, n, verbose)  # int

        # get the indices of the subhalos that have nonzero SFR, nonzero stellar mass, and meet sSFR criteria
        idx = np.intersect1d(self.idx_nonzero(), sSFR_cut)  # length < nsubhalos
        # abundance matching -> sort subhalos that meet the criteria by decreasing stellar mass
        idx_mass_sorted = idx[np.argsort(self.stellar_mass()[idx])[::-1]]  # length < nsubhalos
        assert len(idx) == len(idx_mass_sorted)
        
        # and only return the first target_N
        return idx_mass_sorted[:target_N]


    """ CORRELATION FUNCTIONS """
    def galxdm(self, tracer_name, survey='DESI', n=None, dm_nx=100, verbose=False,
                randmult=3, rmin=0.1, rmax=50., nbins=20, nthreads=24, logbins=True, periodic=True):

        # dark matter particle coordinates -> underlying matter field
        dm_pos = self.dm_pos()
        if dm_nx:
            dm_subsample = tools.get_subsample(dm_pos, dm_nx, verbose=verbose)
        
        # galaxy coordinates -> tracers
        gal_pos = self.subhalo_pos()[self.gal_idx(tracer_name, survey, n=n, verbose=verbose)]

        # compute cross correlation
        ravg, xi = compute_xi_cross(gal_pos.value, dm_subsample.value, randmult, rmin, rmax, nbins,
                                        logbins=logbins, periodic=periodic, nthreads=nthreads, verbose=verbose)
        return ravg, xi