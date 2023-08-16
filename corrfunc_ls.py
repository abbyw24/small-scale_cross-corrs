import numpy as np
import Corrfunc
from Corrfunc.theory import DD, DDrppi


def set_up_cf_data(data, randmult, rmin, rmax, nbins, boxsize=None, logbins=True, dtype='float32'):
    """Helper function to set up data sets for Corrfunc."""
    if logbins:
        r_edges = np.logspace(np.log10(rmin), np.log10(rmax), nbins+1)
    else:
        r_edges = np.linspace(rmin, rmax, nbins+1)
    r_avg = 0.5*(r_edges[1:]+r_edges[:-1])
    nd = len(data)
    boxsize = round(np.amax(data)) if boxsize is None else boxsize
    nr = randmult * nd
    rand_set = np.random.uniform(0, boxsize, (nr,3)).astype(dtype)
    data_set = data.astype(dtype)
    return dict(r_edges=r_edges, r_avg=r_avg, nd=nd, boxsize=boxsize, nr=nr, rand_set=rand_set, data_set=data_set)


# AUTO-CORRELATIONS #
def compute_3D_ls_auto(data, randmult, rmin, rmax, nbins, logbins=True, periodic=True, nthreads=12, rr_fn=None, prints=False):
    """Estimate the 3D 2-pt. autocorrelation function."""

    # set up data
    dataforcf = set_up_cf_data(data, randmult, rmin, rmax, nbins, logbins=logbins, dtype='float32')
    r_edges, r_avg, nd, boxsize, nr, rand_set, data_set = dataforcf.values()

    # unpack
    x, y, z = data_set.T
    x_rand, y_rand, z_rand = rand_set.T

    if prints:
        print("starting computation", flush=True)
    dd_res = DD(1, nthreads, r_edges, x, y, z, boxsize=boxsize, periodic=periodic, output_ravg=True)
    if prints:
        print("DD calculated", flush=True)
    dr_res = DD(0, nthreads, r_edges, x, y, z, X2=x_rand, Y2=y_rand, Z2=z_rand, boxsize=boxsize, periodic=periodic)
    if prints:
        print("DR calculated", flush=True)
    
    if rr_fn:
        rr_res = np.load(rr_fn, allow_pickle=True)
    else:
        rr_res = DD(1, nthreads, r_edges, x_rand, y_rand, z_rand, boxsize=boxsize, periodic=periodic)
    if prints:
        print("RR calculated", flush=True)

    dd = np.array([x['npairs'] for x in dd_res], dtype=float)
    dr = np.array([x['npairs'] for x in dr_res], dtype=float)
    rr = np.array([x['npairs'] for x in rr_res], dtype=float)

    results_xi = Corrfunc.utils.convert_3d_counts_to_cf(nd, nd, nr, nr, dd, dr, dr, rr)
    if prints:
        print("3d counts converted to cf", flush=True)

    return r_avg, results_xi


def compute_2D_ls_auto(data, randmult, rpmin, rpmax, nrpbins, pimax, logbins=True, periodic=True, nthreads=12, rr_fn=None, prints=False):
    """Estimate the projected (2D) 2-pt. autocorrelation function."""

    # set up data
    dataforcf = set_up_cf_data(data, randmult, rpmin, rpmax, nrpbins, logbins=logbins)
    rp_edges, rp_avg, nd, boxsize, nr, rand_set, data_set = dataforcf.values()

    # unpack
    x, y, z = data_set.T
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


# CROSS-CORRELATIONS #
def compute_3D_ls_cross(d1, d2, randmult, rmin, rmax, nbins, boxsize=None,
                        logbins=True, periodic=True, nthreads=12, rr_fn=None, prints=False):
    """Estimate the 3D 2-pt. cross-correlation function."""

    # set up data: random set goes with tracers
    d1forcf = set_up_cf_data(d1, randmult, rmin, rmax, nbins, boxsize=boxsize, logbins=logbins, dtype='float32')
    r_edges, r_avg, nd1, boxsized1, _, _, d1_set = d1forcf.values()
    d2forcf = set_up_cf_data(d2, randmult, rmin, rmax, nbins, boxsize=boxsize, logbins=logbins, dtype='float32')
    r_edges, r_avg, nd2, boxsized2, nr, rand_set, d2_set = d2forcf.values()
    assert boxsized1==boxsized2, "data sets must have the same boxsize!"

    # unpack
    xd1, yd1, zd1 = d1_set.T
    xd2, yd2, zd2 = d2_set.T
    x_rand, y_rand, z_rand = rand_set.T

    d1d2_res = DD(0, nthreads, r_edges, xd1, yd1, zd1, X2=xd2, Y2=yd2, Z2=zd2, boxsize=boxsized1, periodic=periodic, output_ravg=True)
    if prints:
        print("D1D2 calculated")
    d1r_res = DD(0, nthreads, r_edges, xd1, yd1, zd1, X2=x_rand, Y2=y_rand, Z2=z_rand, boxsize=boxsized1, periodic=periodic)
    if prints:
        print("D1R calculated")
    d2r_res = DD(0, nthreads, r_edges, xd2, yd2, zd2, X2=x_rand, Y2=y_rand, Z2=z_rand, boxsize=boxsized1, periodic=periodic)
    if prints:
        print("D2R calculated")
    
    if rr_fn:
        rr_res = np.load(rr_fn, allow_pickle=True)
    else:
        rr_res = DD(1, nthreads, r_edges, x_rand, y_rand, z_rand, boxsize=boxsized1, periodic=periodic)
    if prints:
        print("RR calculated")

    d1d2 = np.array([x['npairs'] for x in d1d2_res], dtype=float)
    d1r = np.array([x['npairs'] for x in d1r_res], dtype=float)
    d2r = np.array([x['npairs'] for x in d2r_res], dtype=float)
    rr = np.array([x['npairs'] for x in rr_res], dtype=float)

    results_xi = Corrfunc.utils.convert_3d_counts_to_cf(nd1, nd2, nr, nr, d1d2, d1r, d2r, rr)
    if prints:
        print("3d counts converted to cf")

    return r_avg, results_xi