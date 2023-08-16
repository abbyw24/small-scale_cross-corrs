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

from corrfunc_ls import compute_3D_ls_auto
from survey_params_gal import eBOSS_param, DESI_param

class TNGSim():
    """
    Class to manage Illustris-TNG simulations and data most relevant for testing small-scale cross-correlations.
    Works with a single snapshot/redshift at a time.
    """

    def __init__(self, sim, snapshot=None, redshift=None, scratch='/scratch/08811/aew492'):

        # make sure we have passed either a snapshot or a redshift
        assert (snapshot is not None and redshift is None) or (snapshot is None and redshift is not None), \
            "must pass either snapshot or redshift"
        
        self.sim = str(sim)
        self.basepath = os.path.join(scratch, f'small-scale_cross-corrs/{sim}/output')
        if self.sim[:-2] == 'TNG300':
            self.snapshots = [0, 4, 17, 33, 40, 50, 67, 84, 91, 99]
            self.redshifts = [20.05, 10., 5., 2., 1.5, 1., 0.5, 0.2, 0.1, 0.]
            self.boxsize = 205 * (u.Mpc / cu.littleh)
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
            assert snapshot in self.snapshots, "snapshot not found"
            self.snapshot = snapshot
            self.redshift = self.redshifts[self.snapshots.index(snapshot)]
        else:
            assert snapshot is None
            assert redshift in self.redshifts, "snapshot not found"
            self.redshift = redshift
            self.snapshot = self.snapshots[self.redshifts.index(redshift)]
    
    
    """
    GROUP CATALOGS:

        _load_subfind_subhalos() : loads the raw requested field data from the group catalog.

        [SOME_FIELD]() : returns the requested field with proper units.
    
    """

    def _load_subfind_subhalos(self, fields=['SubhaloPos','SubhaloMassType','SubhaloLenType','SubhaloSFR'],
                                remove_flagged=True, prints=False):
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
                if prints:
                    print(f"removed {np.sum(~subhalo_idx)} flagged subhalos")
            else:
                self.nsubhalos = len(subhalos['SubhaloFlag'])
                subhalo_idx = np.full(self.nsubhalos, True)
            if prints:
                print(f"loaded the following fields for {self.sim_tag} ({self.nsubhalos} subhalos): \n\t", fields_[1:])
            # we need to remove the 'count' entry before we store data as a table, so each column has the same length
            del subhalos['count']
            if 'SubhaloFlag' in subhalo_info.keys():
                del subhalos['SubhaloFlag']
            subhalos = table.Table(subhalos)
            # append this new info to the table
            self.subhalo_info = table.hstack([subhalo_info, subhalos]) if hasattr(self, 'subhalo_info') else table.Table(subhalos)
        else:
            if prints:
                print("requested fields already loaded!")
    
    def subhalo_pos(self, unit=u.Mpc):
        self._load_subfind_subhalos(fields=['SubhaloPos'])
        return (self.subhalo_info['SubhaloPos'] * u.kpc).to(unit)

    def stellar_mass(self):
        self._load_subfind_subhalos(fields=['SubhaloMassType'])
        return self.subhalo_info['SubhaloMassType'][:,4] * 1e10 * u.M_sun / cu.littleh # 4th col corresponds to star particles
    
    def SFR(self):
        self._load_subfind_subhalos(fields=['SubhaloSFR'])
        return self.subhalo_info['SubhaloSFR'] * u.M_sun / u.year
    

    """ SNAPSHOTS """

    def dm_pos(self, unit=u.Mpc):
        """
        Load the (x,y,z) comoving coordinates of all DM particles.
        If a snapshot or redshift is passed, this overrides any previously set snapshot.
        """
        dm_pos = il.snapshot.loadSubset(self.basepath, self.snapshot, 'dm', ['Coordinates']) * u.kpc
        return dm_pos.to(unit)
    

    """ SURVEY EMULATION """

    def survey_params(self, survey_name, tracer_name):
        """
        Calculate the target galaxy number density (function of redshift) for a specific survey, using survey_params_gal.py.
        """
        if survey_name == 'eBOSS':
            return eBOSS_param(z=self.redshift, tracer_name=tracer_name)
        else:
            assert survey_name == 'DESI', "'survey_name' must be 'eBOSS' or 'DESI'"
            return DESI_param(z=self.redshift, tracer_name=tracer_name)
    
    def gal_idx(self, tracer_name, survey='DESI', sSFR_cutval=-9.09, n=None, prints=False):
        """
        Make cuts in subhalo sSFR and stellar mass to select for LRGs/ELGs and reach a target galaxy number density,
        as outlined in Sullivan, Prijon, & Seljak (2023).
        Option to manually input target number density, otherwise compute as a function of `tracer_name` and `survey` \
        using survey_params().

        S, P & S 2023 uses two different cutoff values for sSFR, log10(sSFR) =
            1. -9.09 (https://arxiv.org/abs/2210.10068 using MilleniumTNG)
            2. -9.23 (https://arxiv.org/abs/2011.05331 using TNG300-1)
        """
        # target number of galaxies = volume * number density
        V = self.boxsize**3
        if n:
            n = n.to((cu.littleh / u.Mpc)**3) if hasattr(n, 'unit') else n * (cu.littleh / u.Mpc)**3
            print(f"input number density: {n.value:.2e} (h/Mpc)^3")
        else:
            n = self.survey_params(survey, tracer_name).n_Mpc3 *  (cu.littleh / u.Mpc)**3
            print(f"{tracer_name} number density for {survey} at z={self.redshift}: {n.value:.2e} (h/Mpc)^3 ")
        target_N = int(V * n)
        if prints:
            print(f"target number of subhalos: {target_N}")
        
        # specific star formation rate (sSFR) = star formation rate per stellar mass; and make cut
        idx_nonzero = (self.SFR() > 0) & (self.stellar_mass() > 0)
        sSFR = self.SFR()[idx_nonzero] / self.stellar_mass()[idx_nonzero]
        if tracer_name == 'LRG':
            sSFR_cut = (np.log10(sSFR.value) < sSFR_cutval)
        elif tracer_name == 'ELG':
            sSFR_cut = (np.log10(sSFR.value) > sSFR_cutval)

        # sort subhalos by decreasing stellar mass, then take only the first target_N
        return np.argsort(self.stellar_mass()[idx_nonzero][sSFR_cut])[::-1][:target_N]
    
    def LRG_pos(self, survey='DESI', sSFR_cutval=-9.09, n=None, prints=False):
        return self.subhalo_pos()[self.gal_idx('LRG', survey, sSFR_cutval, n, prints)]