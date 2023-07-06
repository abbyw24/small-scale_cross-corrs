"""
GOAL: Compute the 3D 2pcf on an Illustris DM-only simulation.
"""

import numpy as np
import illustris_python as il
import time
import datetime
import os
import sys

from illustris_sim import IllustrisSim
from corrfunc_ls import compute_3D_ls_auto
import tools


def compute_xi_auto_illustris(sim_name, redshift, data_type, randmult, rmin, rmax, nbins, subsample_nx=None,
                        logbins=True, periodic=True, nthreads=12, prints=False, save_fn=None, return_res=True):
    sim = IllustrisSim(sim_name)
    sim.set_snapshot(redshift=redshift)
    if data_type=='dm':
        sim.load_dm_pos()
        data = sim.dm_pos.value
    else:
        assert data_type=='gal', "unknown data_type!"
        sim.load_gal_pos()
        data = sim.gal_pos.value

    if subsample_nx:
        print(f"subsampling 1/{subsample_nx}th of the data", flush=True)
        data = tools.get_subsample(data, subsample_nx)
    
    ravg, xi = compute_3D_ls_auto(data, randmult, rmin, rmax, nbins, logbins=logbins, periodic=periodic, nthreads=nthreads, prints=prints)

    result = dict(ravg=ravg, xi=xi, rmin=rmin, rmax=rmax, nbins=nbins,
                    logbins=logbins, periodic=periodic, nthreads=nthreads,
                    redshift=redshift, data=data)
    
    if save_fn:
        np.save(save_fn, result)
        print(f"saved result to {save_fn}", flush=True)
    if return_res:
        return result


def main():

    s = time.time()

    # params for Illustris simulation
    sim_name = 'TNG300-2'
    redshifts = [0.]
    data_type = 'dm'

    # params for Corrfunc
    randmult = 3
    rmin = 0.1
    rmax = 50.  
    nbins = 20
    subsample_nx = 100
    nthreads = 24

    save_dir = '/scratch/08811/aew492/small-scale_cross-corrs/xi'
    subsample_tag = f'_nx-{subsample_nx}' if subsample_nx else ''

    for redshift in redshifts:
        save_fn = os.path.join(save_dir, f'xi_{data_type}_{sim_name}_z-{redshift:.2f}{subsample_tag}.npy')
        compute_xi_auto_illustris(sim_name, redshift, data_type, randmult, rmin, rmax, nbins,
                                subsample_nx=subsample_nx, nthreads=nthreads, save_fn=save_fn, return_res=False, prints=True)
    
    total_time = time.time() - s
    print(f"total time: {datetime.timedelta(seconds=total_time)}", flush=True)


if __name__=='__main__':
    main()