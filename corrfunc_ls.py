import numpy as np
import Corrfunc
from Corrfunc.theory import DD, DDrppi


def set_up_cf_data(data, randmult, rmin, rmax, nbins, logbins=True):
    """Helper function to set up data sets for Corrfunc."""
    if logbins:
        r_edges = np.logspace(np.log10(rmin), np.log10(rmax), nbins+1)
    else:
        r_edges = np.linspace(rmin, rmax, nbins+1)
    r_avg = 0.5*(r_edges[1:]+r_edges[:-1])
    nd = len(data)
    boxsize = round(np.amax(data))
    nr = randmult * nd
    rand_set = np.random.uniform(0, boxsize, (nr,3)).astype('float32')
    return dict(r_edges=r_edges, r_avg=r_avg, nd=nd, boxsize=boxsize, nr=nr, rand_set=rand_set)


def compute_3D_ls(data, randmult, rmin, rmax, nbins, logbins=True, periodic=True, nthreads=12, rr_fn=None, prints=False):
    """Estimate the 3D 2-pt. autocorrelation function."""

    # set up data
    dataforcf = set_up_cf_data(data, randmult, rmin, rmax, nbins, logbins=logbins)
    r_edges, r_avg, nd, boxsize, nr, rand_set = dataforcf.values()

    # unpack
    x, y, z = data.T
    x_rand, y_rand, z_rand = rand_set.T

    dd_res = DD(1, nthreads, r_edges, x, y, z, boxsize=boxsize, periodic=periodic, output_ravg=True)
    if prints:
        print("DD calculated")
    dr_res = DD(0, nthreads, r_edges, x, y, z, X2=x_rand, Y2=y_rand, Z2=z_rand, boxsize=boxsize, periodic=periodic)
    if prints:
        print("DR calculated")
    
    if rr_fn:
        rr_res = np.load(rr_fn, allow_pickle=True)
    else:
        rr_res = DD(1, nthreads, r_edges, x_rand, y_rand, z_rand, boxsize=boxsize, periodic=periodic)
    if prints:
        print("RR calculated")

    dd = np.array([x['npairs'] for x in dd_res], dtype=float)
    dr = np.array([x['npairs'] for x in dr_res], dtype=float)
    rr = np.array([x['npairs'] for x in rr_res], dtype=float)

    results_xi = Corrfunc.utils.convert_3d_counts_to_cf(nd, nd, nr, nr, dd, dr, dr, rr)
    if prints:
        print("3d counts converted to cf")

    return r_avg, results_xi


def compute_2D_ls(data, randmult, rpmin, rpmax, nrpbins, pimax, logbins=True, periodic=True, nthreads=12, rr_fn=None, prints=False):
    """Estimate the projected (2D) 2-pt. autocorrelation function."""

    # set up data
    dataforcf = set_up_cf_data(data, randmult, rpmin, rpmax, nrpbins, logbins=logbins)
    rp_edges, rp_avg, nd, boxsize, nr, rand_set = dataforcf.values()

    # unpack
    x, y, z = data.T
    x_rand, y_rand, z_rand = rand_set.T

    dd_res = DDrppi(1, nthreads, pimax, rp_edges, x, y, z, boxsize=boxsize, periodic=periodic)
    if prints:
        print("DD calculated")
    dr_res = DDrppi(0, nthreads, pimax, rp_edges, x, y, z, X2=x_rand, Y2=y_rand, Z2=z_rand, boxsize=boxsize, periodic=periodic)
    if prints:
        print("DR calculated")
    
    if rr_fn:
        rr_res = np.load(rr_fn, allow_pickle=True)
    else:
        rr_res = DDrppi(1, nthreads, pimax, rp_edges, x_rand, y_rand, z_rand, boxsize=boxsize, periodic=periodic)
    if prints:
        print("RR calculated")

    dd = np.array([x['npairs'] for x in dd_res], dtype=float)
    dr = np.array([x['npairs'] for x in dr_res], dtype=float)
    rr = np.array([x['npairs'] for x in rr_res], dtype=float)

    results_wp = Corrfunc.utils.convert_rp_pi_counts_to_wp(nd, nd, nr, nr, dd, dr, dr, rr, nrpbins, pimax)
    if prints:
        print("2d counts converted to cf")
    
    return rp_avg, results_wp