import numpy as np
import astropy.units as u
import astropy.cosmology.units as cu
import Corrfunc
from Corrfunc.theory import DDrppi
from Corrfunc.theory.DD import DD
from Corrfunc.mocks.DDtheta_mocks import DDtheta_mocks


def set_up_cf_data(data, randmult, rmin, rmax, nbins, boxsize=None, zrange=None, logbins=True, dtype='float32'):
    """Helper function to set up data sets for Corrfunc."""
    # remove units
    if isinstance(data, u.Quantity):
        data = data.value
    if isinstance(boxsize, u.Quantity):
        boxsize = boxsize.value
    if logbins:
        r_edges = np.logspace(np.log10(rmin), np.log10(rmax), nbins+1)
    else:
        r_edges = np.linspace(rmin, rmax, nbins+1)
    r_avg = 0.5*(r_edges[1:]+r_edges[:-1])
    nd = len(data)
    if np.amin(data) < 0:
        print("shifting data by L/2!")
        L = 2 * round(np.amax(data)) if boxsize is None else boxsize
        data += L/2
    boxsize = round(np.amax(data)) if boxsize is None else boxsize
    # construct random set: cubic if zrange is None else incorporate zrange
    nr = randmult * nd
    rand_set = np.random.uniform(0, boxsize, (nr,3)).astype(dtype)
    # zrange = [round(np.amin(data[:,2])), round(np.amax(data[:,2]))] if zrange is None else zrange
    if zrange is not None:
        rand_set[:,2] = np.random.uniform(zrange[0], zrange[1], nr).astype(dtype)
    # rand_set = np.array([
    #     np.random.uniform(0, boxsize, nr),
    #     np.random.uniform(0, boxsize, nr),
    #     np.random.uniform(zrange[0], zrange[1], nr)
    # ]).T.astype(dtype)
    data_set = data.astype(dtype)
    # # !!
    # print(f"zrange = {zrange}")
    # print("data:")
    # x, y, z = data_set.T
    # print(f"x range: {min(x):.1f} - {max(x):.1f}")
    # print(f"y range: {min(y):.1f} - {max(y):.1f}")
    # print(f"z range: {min(z):.1f} - {max(z):.1f}")
    # print("random:")
    # x, y, z = rand_set.T
    # print(f"x range: {min(x):.1f} - {max(x):.1f}")
    # print(f"y range: {min(y):.1f} - {max(y):.1f}")
    # print(f"z range: {min(z):.1f} - {max(z):.1f}")
    # # !!
    return dict(r_edges=r_edges, r_avg=r_avg, nd=nd, boxsize=boxsize, nr=nr, rand_set=rand_set, data_set=data_set)


# AUTO-CORRELATIONS #
def compute_3D_ls_auto(data, randmult, rmin, rmax, nbins, boxsize=None, zrange=None, logbins=True, periodic=True, nthreads=12,
                        nrepeats=1, rr_fn=None, prints=False):
    """Estimate the 3D 2-pt. autocorrelation function."""

    # set up data
    dataforcf = set_up_cf_data(data, randmult, rmin, rmax, nbins, boxsize=boxsize, zrange=zrange, logbins=logbins, dtype='float32')
    r_edges, r_avg, nd, boxsize, nr, rand_set, data_set = dataforcf.values()

    # unpack
    x, y, z = data_set.T
    x_rand, y_rand, z_rand = rand_set.T

    if prints:
        print("starting computation", flush=True)
    # repeat the calculation multiple times if desired (helps with noise)
    xis = np.empty((nrepeats, nbins))
    for i in range(nrepeats):
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

        xis[i] = Corrfunc.utils.convert_3d_counts_to_cf(nd, nd, nr, nr, dd, dr, dr, rr)
        if prints:
            print("3d counts converted to cf", flush=True)

    return r_avg, np.nanmean(xis, axis=0)


def compute_2D_ls_auto(data, randmult, rpmin, rpmax, nrpbins, pimax, boxsize=None, zrange=None,
                        logbins=True, periodic=True, nthreads=12, rr_fn=None, prints=False):
    """Estimate the projected (2D) 2-pt. autocorrelation function."""

    # set up data
    dataforcf = set_up_cf_data(data, randmult, rpmin, rpmax, nrpbins, boxsize=boxsize, zrange=None, logbins=logbins)
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
def compute_3D_ls_cross(d1, d2, randmult, rmin, rmax, nbins, boxsize=None, zrange=None,
                        logbins=True, periodic=True, nthreads=12, rr_fn=None, prints=False):
    """Estimate the 3D 2-pt. cross-correlation function."""

    # set up data: random set goes with tracers
    d1forcf = set_up_cf_data(d1, randmult, rmin, rmax, nbins, boxsize=boxsize, zrange=zrange, logbins=logbins, dtype='float32')
    r_edges, r_avg, nd1, boxsized1, _, _, d1_set = d1forcf.values()
    d2forcf = set_up_cf_data(d2, randmult, rmin, rmax, nbins, boxsize=boxsize, zrange=zrange, logbins=logbins, dtype='float32')
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

def compute_2D_ls_cross(d1, d2, randmult, rpmin, rpmax, nbins, pimax, boxsize=None, zrange=None,
                        logbins=True, periodic=True, nthreads=12, rr_fn=None, prints=False):
    """Estimate the 2D 2-pt. cross-correlation function."""

    # set up data: random set goes with tracers
    d1forcf = set_up_cf_data(d1, randmult, rpmin, rpmax, nbins, boxsize=boxsize, zrange=zrange, logbins=logbins, dtype='float32')
    rp_edges, rp_avg, nd1, boxsized1, _, _, d1_set = d1forcf.values()
    d2forcf = set_up_cf_data(d2, randmult, rpmin, rpmax, nbins, boxsize=boxsize, zrange=zrange, logbins=logbins, dtype='float32')
    rp_edges, rp_avg, nd2, boxsized2, nr, rand_set, d2_set = d2forcf.values()
    assert boxsized1==boxsized2, "data sets must have the same boxsize!"

    # unpack
    xd1, yd1, zd1 = d1_set.T
    xd2, yd2, zd2 = d2_set.T
    x_rand, y_rand, z_rand = rand_set.T

    d1d2_res = DDrppi(0, nthreads, pimax, rp_edges, xd1, yd1, zd1, X2=xd2, Y2=yd2, Z2=zd2, boxsize=boxsized1, periodic=periodic, output_rpavg=True)
    if prints:
        print("D1D2 calculated")
    d1r_res = DDrppi(0, nthreads, pimax, rp_edges, xd1, yd1, zd1, X2=x_rand, Y2=y_rand, Z2=z_rand, boxsize=boxsized1, periodic=periodic)
    if prints:
        print("D1R calculated")
    d2r_res = DDrppi(0, nthreads, pimax, rp_edges, xd2, yd2, zd2, X2=x_rand, Y2=y_rand, Z2=z_rand, boxsize=boxsized1, periodic=periodic)
    if prints:
        print("D2R calculated")
    
    if rr_fn:
        rr_res = np.load(rr_fn, allow_pickle=True)
    else:
        rr_res = DDrppi(1, nthreads, pimax, rp_edges, x_rand, y_rand, z_rand, boxsize=boxsized1, periodic=periodic)
    if prints:
        print("RR calculated")

    d1d2 = np.array([x['npairs'] for x in d1d2_res], dtype=float)
    d1r = np.array([x['npairs'] for x in d1r_res], dtype=float)
    d2r = np.array([x['npairs'] for x in d2r_res], dtype=float)
    rr = np.array([x['npairs'] for x in rr_res], dtype=float)

    results_wp = Corrfunc.utils.convert_rp_pi_counts_to_wp(nd1, nd2, nr, nr, d1d2, d1r, d2r, rr, nbins, pimax)
    if prints:
        print("3d counts converted to cf")

    return rp_avg, results_wp


def wtheta_cross_PH(ra1, dec1, ra2, dec2, ra_rand2, dec_rand2, bins, nthreads=12):
    nd1 = len(ra1)
    nd2 = len(ra2)
    nr2 = len(ra_rand2)
    binavg = 0.5 * (bins[1:] + bins[:-1])
    
    # D1D2
    D1D2_counts = DDtheta_mocks(0, nthreads, bins, ra1, dec1, RA2=ra2, DEC2=dec2)
    # D1R2
    D1R2_counts = DDtheta_mocks(0, nthreads, bins, ra1, dec1, RA2=ra_rand2, DEC2=dec_rand2)
    
    d1d2 = np.array([x['npairs'] for x in D1D2_counts]) / (nd1 * nd2)
    d1r2 = np.array([x['npairs'] for x in D1R2_counts]) / (nd1 * nr2)
    
    wtheta = np.empty(len(binavg))
    wtheta[:] = np.nan
    wtheta = np.divide(d1d2, d1r2, where=(d1r2!=0.), out=wtheta) - 1
    
    return binavg, wtheta


def xi_cross(data1, data2, rand2, bins, boxsize, nthreads=12, periodic=True, dtype='float32'):

    # get compatible units
    unit = u.Mpc / cu.littleh
    data1 = data1.to(unit).value if isinstance(data1, u.Quantity) else data1
    data2 = data2.to(unit).value if isinstance(data2, u.Quantity) else data2
    rand2 = rand2.to(unit).value if isinstance(rand2, u.Quantity) else rand2
    bins = bins.to(unit).value if isinstance(bins, u.Quantity) else bins
    boxsize = boxsize.to(unit).value if isinstance(boxsize, u.Quantity) else boxsize

    # params
    binavg = (0.5 * (bins[1:] + bins[:-1])).astype(dtype)
    xd1, yd1, zd1 = data1.T.astype(dtype)
    xd2, yd2, zd2 = data2.T.astype(dtype)
    xr, yr, zr = rand2.T.astype(dtype)

    assert 0 < np.all(data1) < boxsize
    assert 0 < np.all(data2) < boxsize
    assert 0 < np.all(rand2) < boxsize
    
    # compute pair counts
    d1d2_res = DD(0, nthreads, bins, xd1, yd1, zd1, X2=xd2, Y2=yd2, Z2=zd2,
                      boxsize=boxsize, periodic=periodic, output_ravg=True)
    d1r2_res = DD(0, nthreads, bins, xd1, yd1, zd1, X2=xr, Y2=yr, Z2=zr,
                      boxsize=boxsize, periodic=periodic, output_ravg=True)
    d1d2 = np.array([x['npairs'] for x in d1d2_res], dtype=dtype)
    d1r2 = np.array([x['npairs'] for x in d1r2_res], dtype=dtype)

    ndpairs = len(data1) * len(data2)
    nrpairs = len(data1) * len(rand2)

    counts = np.divide(d1d2, d1r2, where=(d1r2!=0.), out=np.zeros_like(d1d2), dtype=dtype)

    return binavg, nrpairs / ndpairs * counts - 1


def xi_auto(data, rand, bins, boxsize, nthreads=12, periodic=True, dtype='float32'):

    # get compatible units
    unit = u.Mpc / cu.littleh
    data = data.to(unit).value if isinstance(data, u.Quantity) else data
    rand = rand.to(unit).value if isinstance(rand, u.Quantity) else rand
    bins = bins.to(unit).value if isinstance(bins, u.Quantity) else bins
    boxsize = boxsize.to(unit).value if isinstance(boxsize, u.Quantity) else boxsize

    # params
    binavg = (0.5 * (bins[1:] + bins[:-1])).astype(dtype)
    xd, yd, zd = data.T.astype(dtype)
    xr, yr, zr = rand.T.astype(dtype)

    assert 0 < np.all(data) < boxsize
    assert 0 < np.all(rand) < boxsize
    
    # compute pair counts
    dd_res = DD(1, nthreads, bins, xd, yd, zd, boxsize=boxsize, periodic=periodic, output_ravg=True)
    rr_res = DD(1, nthreads, bins, xr, yr, zr, boxsize=boxsize, periodic=periodic, output_ravg=True)

    dd = np.array([x['npairs'] for x in dd_res], dtype=dtype)
    rr = np.array([x['npairs'] for x in rr_res], dtype=dtype)

    ndpairs = len(data) * (len(data)-1)
    nrpairs = len(rand) * (len(rand)-1)

    counts = np.divide(dd, rr, where=(rr!=0.), out=np.zeros_like(rr), dtype=dtype)

    return binavg, nrpairs / ndpairs * counts - 1