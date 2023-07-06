"""
Lookup table to convert redshifts <--> Illustris snapshot numbers.
"""

import numpy as np 
import illustris_python as il 
import astropy.units as u
import os
import sys

from corrfunc_ls import compute_3D_ls_auto

class IllustrisSim():

    def __init__(self, sim, scratch='/scratch/08811/aew492'):
        
        self.sim = str(sim)
        self.basepath = os.path.join(scratch, f'small-scale_cross-corrs/{sim}/output')

        if sim=='Illustris-3':
            self.snapshots = []
            self.redshifts = []
        elif sim[:-2]=='TNG300':  # -2 to include all resolutions
            self.snapshots = [0, 4, 17, 33, 40, 50, 67, 99]
            self.redshifts = [20.05, 10., 5., 2., 1.5, 1., 0.5, 0.]
        else:
            print("not a known simulation")
    

    def set_snapshot(self, snapshot=None, redshift=None):
        if snapshot!=None and redshift==None:
            assert snapshot in self.snapshots, "snapshot not found"
            self.snapshot = snapshot
            self.redshift = self.redshifts[self.snapshots.index(snapshot)]
        elif redshift!=None and snapshot==None:
            assert redshift in self.redshifts, "snapshot not found"
            self.redshift = redshift
            self.snapshot = self.snapshots[self.redshifts.index(redshift)]
        else:
            assert snapshot!=None and redshift!=None, "must pass snapshot or redshift"
            self.snapshot = snapshot
            self.redshift = redshift
        assert hasattr(self, 'snapshot') and hasattr(self, 'redshift')
        self.snapdir = os.path.join(self.basepath, f'snapdir_{self.snapshot:03d}')
        if not os.path.exists(self.snapdir):
            print(f"{self.sim}: snapshot ({self.snapshot:03d}) not found")
        self.groupdir = os.path.join(self.basepath, f'groups_{self.snapshot:03d}')
        if not os.path.exists(self.groupdir):
            print(f"{self.sim}: group catalog ({self.snapshot:03d}) not found")
    

    def load_dm_pos(self, snapshot=None, unit=u.Mpc):
        snapshot = snapshot if snapshot else self.snapshot
        dm_pos = il.snapshot.loadSubset(self.basepath, snapshot, 'dm', ['Coordinates']) * u.kpc
        self.dm_pos = dm_pos.to(unit)
    

    def load_galaxies(self, snapshot=None, minstars=100, minstarmass=0, prints=False):
        snapshot = snapshot if snapshot else self.snapshot
        fields = ['SubhaloFlag','SubhaloPos','SubhaloMass','SubhaloMassType', 'SubhaloLenType']
        subhalos = il.groupcat.loadSubhalos(self.basepath, snapshot, fields=fields)
        if prints:
            print(f"loaded {subhalos['count']} subhalos for {self.sim}")
        # remove any subhalos flagged as non-cosmological in origin, unpack, and give proper units
        subhalo_idx = subhalos['SubhaloFlag']
        # (x,y,z) coordinate of each subhalo:
        subhalo_pos = (subhalos['SubhaloPos'][subhalo_idx] * u.kpc).to(u.Mpc)
        # total mass of each subhalo:
        total_mass = subhalos['SubhaloMass'][subhalo_idx] * 1e10 * u.M_sun
        # total mass of each particle type in each subhalo:
        mass_types = subhalos['SubhaloMassType'][subhalo_idx] * 1e10 * u.Msun
        # total number of each particle type in each subhalo:
        len_types = subhalos['SubhaloLenType'][subhalo_idx]
        if prints:
            print(f"removed {np.sum(~subhalo_idx)} flagged subhalos")
        # galaxies -> take only subhalos with non-zero star mass
        #   and with at least 100 star particles (following Barreira et al 2021)
        gal_idx = np.where((mass_types[:,4].value>minstarmass) & (len_types[:,4]>minstars))
        self.gal_pos = subhalo_pos[gal_idx]
        self.gal_mass = total_mass[gal_idx]
        self.gal_mass_types = mass_types[gal_idx]
        self.gal_len_types = len_types[gal_idx]
        print(f"{self.sim}: loaded {len(self.gal_pos)} subhalos with > {minstars:0d} stars and star mass > {minstarmass:.0f}")