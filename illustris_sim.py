"""
Lookup table to convert redshifts <--> Illustris snapshot numbers.
"""

import numpy as np 
import illustris_python as il 
import astropy.units as u
import os
import sys

from corrfunc_ls import compute_3D_ls

class IllustrisSim():

    def __init__(self, sim, scratch='/scratch/08811/aew492'):
        
        self.sim = str(sim)
        self.basepath = os.path.join(scratch, f'small-scale_cross-corrs/{sim}/output')

        if sim=='Illustris-3':
            self.snapshots = []
            self.redshifts = []
        elif sim=='TNG300-3':
            self.snapshots = [0, 4, 17, 33, 40, 50, 67, 99]
            self.redshifts = [20.05, 10., 5., 2., 1.5, 1., 0.5, 0.]
        else:
            print("not a known simulation")
    

    def set_snapshot(self, snapshot=None, redshift=None, unit=u.Mpc):
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
            print("snapshot directory not found")
    

    def load_dm_pos(self, snapshot=None, unit=u.Mpc):
        snapshot = snapshot if snapshot else self.snapshot
        dm_pos = il.snapshot.loadSubset(self.basepath, snapshot, 'dm', ['Coordinates']) * u.kpc
        self.dm_pos = dm_pos.to(unit)
    

    def compute_xi(self, randmult, rmin, rmax, nbins, logbins=True, periodic=True, nthreads=12, prints=False):
        self.randmult = randmult
        self.rmin = rmin
        self.rmax = rmax
        self.nbins = nbins
        self.logbins = logbins
        self.periodic = periodic
        self.nthreads = nthreads

        if not hasattr(self, 'dm_pos'):
            self.load_dm_pos()
        self.ravg, self.xi = compute_3D_ls(self.dm_pos.value, randmult, rmin, rmax, nbins,
                                            logbins=logbins, periodic=periodic, nthreads=nthreads, prints=prints)
        