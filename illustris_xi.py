"""
GOAL: Compute the 3D 2pcf on an Illustris DM-only simulation.
"""

import numpy as np
import illustris_python as il
import time
import os
import sys

from illustris_sim import IllustrisSim
from corrfunc_ls import compute_3D_ls
import tools


def compute_xi_illustris(sim_name, redshift, randmult, rmin, rmax, nbins, subsample_nx=None,
                        logbins=True, periodic=True, nthreads=12, prints=False, save_fn=None, return_res=True):
    sim = IllustrisSim(sim_name)
    sim.set_snapshot(redshift=redshift)
    sim.load_dm_pos()

    if subsample_nx:
        dm_pos = tools.get_subsample(sim.dm_pos.value, subsample_nx)
    else:
        dm_pos = sim.dm_pos.value
    
    ravg, xi = compute_3D_ls(dm_pos, randmult, rmin, rmax, nbins, logbins=logbins, periodic=periodic, nthreads=nthreads, prints=prints)

    result = dict(ravg=ravg, xi=xi, rmin=rmin, rmax=rmax, nbins=nbins,
                    logbins=logbins, periodic=periodic, nthreads=nthreads,
                    redshift=redshift, dm_pos=dm_pos)
    
    if save_fn:
        np.save(save_fn, result)
        print(f"saved result to {save_fn}")
    if return_res:
        return result


def main():

    s = time.time()

    # params for Illustris simulation
    sim_name = 'TNG300-3'
    redshifts = [20.05]

    # params for Corrfunc
    randmult = 3
    rmin = 0.1
    rmax = 50.
    nbins = 20
    subsample_nx = 100
    nthreads = 24

    save_dir = '/scratch/08811/aew492/small-scale_cross-corrs/xi'

    for redshift in redshifts:
        save_fn = os.path.join(save_dir, f'xi_{sim_name}_z-{redshift:.2f}_nx-{subsample_nx}')
        compute_xi_illustris(sim_name, redshift, randmult, rmin, rmax, nbins,
                                subsample_nx=subsample_nx, nthreads=nthreads, save_fn=save_fn, return_res=False)
    
    total_time = time.time() - s
    print(total_time)


if __name__=='__main__':
    main()