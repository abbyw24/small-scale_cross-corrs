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

class IllustrisSim():
    """
    Class to manage Illustris-TNG simulations and data most relevant for testing small-scale cross-correlations.
    Works with a single snapshot/redshift at a time.
    """

    def __init__(self, sim, snapshot=None, redshift=None, scratch='/scratch/08811/aew492'):
        
        self.sim = str(sim)
        self.basepath = os.path.join(scratch, f'small-scale_cross-corrs/{sim}/output')

        if sim=='Illustris-3':
            self.snapshots = []
            self.redshifts = []
        elif sim[:-2]=='TNG300':  # -2 to include all resolutions
            self.snapshots = [0, 4, 17, 33, 40, 50, 67, 84, 91, 99]
            self.redshifts = [20.05, 10., 5., 2., 1.5, 1., 0.5, 0.2, 0.1, 0.]
        else:
            print("not a known simulation")
        
        # if a snapshot has been passed, go ahead and set the snapshot:
        if (snapshot!=None) or (redshift!=None):
            self.set_snapshot(snapshot, redshift)
        
        self.sim_tag = self.sim+f', z={self.redshift:.2f}' if hasattr(self, 'snapshot') else self.sim

    
    """ SNAPSHOTS <--> REDSHIFTS """

    def get_redshift(self, snapshot):
        return self.redshifts[self.snapshots.index(snapshot)]
    
    def get_snapshot(self, redshift):
        return self.snapshots[self.redshifts.index(redshift)]

    def set_snapshot(self, snapshot=None, redshift=None):
        """
        Set the simulation snapshot by either snapshot number or corresponding redshift.
        If nothing is passed, check that the simulation already has a `snapshot` attribute.
        """
        if snapshot!=None and redshift==None:
            assert snapshot in self.snapshots, "snapshot not found"
            self.snapshot = snapshot
            self.redshift = self.get_redshift(snapshot)
        elif redshift!=None and snapshot==None:
            assert redshift in self.redshifts, "snapshot not found"
            self.redshift = redshift
            self.snapshot = self.get_snapshot(redshift)
        else:
            assert hasattr(self, 'snapshot'), "Must pass a snapshot if the simulation does not already have one"
    
    
    """
    GROUP CATALOGS:

        load_subfind_subhalos() : loads the raw requested field data from the group catalog.

        load_[SOME_FIELD]() : loads the requested field, stored with proper units.
    
    """

    def load_subfind_subhalos(self, fields=['SubhaloPos','SubhaloMassType','SubhaloLenType','SubhaloSFR'],
                                remove_flagged=True, prints=False):
        """
        Load the requested fields of this sim's Subfind subhalos,
        (by default removing any flagged as non-cosmological in origin).
        If a snapshot or redshift is passed, this overrides any previously set snapshot.
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

    def load_stellar_mass(self):
        self.load_subfind_subhalos(fields=['SubhaloMassType'])
        self.stellar_mass = self.subhalo_info['SubhaloMassType'][:,4] * 1e10 * u.M_sun / cu.littleh # 4 corresponds to star particles
    
    def load_SFR(self):
        self.load_subfind_subhalos(fields=['SubhaloSFR'])
        self.SFR = self.subhalo_info['SubhaloSFR'] * u.M_sun / u.year
    

    """ SNAPSHOTS """

    def load_dm_pos(self, unit=u.Mpc):
        """
        Load the (x,y,z) comoving coordinates of all DM particles.
        If a snapshot or redshift is passed, this overrides any previously set snapshot.
        """
        dm_pos = il.snapshot.loadSubset(self.basepath, self.snapshot, 'dm', ['Coordinates']) * u.kpc
        self.dm_pos = dm_pos.to(unit)


    def load_galaxies(self, minstars=100, minstarmass=0, prints=False, return_gals=False):
        self.load_subfind_subhalos()

        # unpack values and give proper units:
        subhalo_pos = (self.subhalo_info['SubhaloPos'] * u.kpc).to(u.Mpc)    # (x,y,z) coordinate of each subhalo
        # total_mass = self.subhalo_info['SubhaloMass'] * 1e10 * u.M_sun      # total mass of each subhalo
        mass_types = self.subhalo_info['SubhaloMassType'] * 1e10 * u.Msun   # total mass of each particle type in each subhalo
        len_types = self.subhalo_info['SubhaloLenType']                     # total number of each particle type in each subhalo

        # galaxies -> take only subhalos with non-zero star mass
        #   and with at least 100 star particles (following Barreira et al 2021)
        gal_idx = np.where((mass_types[:,4].value>minstarmass) & (len_types[:,4]>minstars))
        self.gal_pos = subhalo_pos[gal_idx]
        # self.gal_mass = total_mass[gal_idx]
        self.gal_mass_types = mass_types[gal_idx]
        self.gal_len_types = len_types[gal_idx]
        if prints:
            print(f"{self.sim_tag}: loaded {len(self.gal_pos)} subhalos with > {minstars:0d} stars and star mass > {minstarmass:.0f}")
        if return_gals:
            return dict(gal_pos=subhalo_pos[gal_idx], # gal_mass=total_mass[gal_idx],
                        gal_mass_types=mass_types[gal_idx], gal_len_types=len_types[gal_idx])