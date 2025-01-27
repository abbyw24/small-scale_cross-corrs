import numpy as np
from colossus.cosmology import cosmology
import astropy.units as u
import astropy.cosmology.units as cu
import astropy.constants as c
from astropy.cosmology import Planck15 as cosmo
from mcfit import P2xi
import symbolic_pofk.syrenhalofit as syrenhalofit

"""
CONVERSIONS
"""
def a(redshift):
    return 1 / (1 + redshift)
def perh():
    return (cosmo.H(0.) / 100 * u.Mpc / u.km * u.s) / cu.littleh

def get_dx(z, sigma_z):
    return sigma_z * (1 + z) * c.c.to(u.km/u.s) / cosmo.H(z) * perh()

def redshift_to_comov(z, cosmo=cosmo):
    r = cosmo.comoving_distance(z) * perh() # convert to Mpc/h
    return r

def theta_to_r_comov(theta, redshift):
    return (theta * u.deg * cosmo.kpc_comoving_per_arcmin(redshift).to(u.Mpc/u.deg) * perh())
def r_comov_to_theta(r, redshift):
    return (r / (cosmo.kpc_comoving_per_arcmin(redshift).to(u.Mpc/u.deg) * perh()))

def r_comov_to_r(r_comov, redshift):
    return r_comov * a(redshift)
def r_to_r_comov(r, redshift):
    return r / a(redshift)

def CartesiantoEquatorial(pos, observer=[0,0,0]):
    """
    Bug:
    - pretty sure this doesn't work if pos[:,2] ≠ 0, but I need to do more detailed trig debugging
    """

    pos_ = pos.value if isinstance(pos, u.Quantity) else pos
    if isinstance(observer, u.Quantity) and isinstance(pos, u.Quantity):
        observer = observer.to(pos.unit)
    observer_ = observer.value if isinstance(observer, u.Quantity) else observer
    x, y, z = (pos_ - np.array(observer_)).T
    s = np.hypot(z, y) 
    lon = np.arctan2(y, z)
    lat = np.arctan2(x, s)

    # convert to degrees
    lon = np.rad2deg(lon)
    lat = np.rad2deg(lat)
    # wrap lon to [0,360]
    lon = np.mod(lon-360., 360.)
    return lon << u.deg, lat << u.deg

def get_ra_dec(sample, chi):
    """
    Return the (RA,Dec) coordinates given (x,y,z) galaxy coordinates and comoving distance `chi`
    to the observer.
    """
    # strip potential units
    if isinstance(sample, u.Quantity):
        sample = sample.to(u.Mpc / cu.littleh).value
    if isinstance(chi, u.Quantity):
        chi = chi.to(u.Mpc / cu.littleh).value
    # convert photometric sample to (RA, Dec)
    s = np.copy(sample)
    observer = np.zeros_like(s)
    observer[:,2] = s[:,2] + chi    # take z positions into account for LOS distance
    s[:,2] = 0
    ra, dec = CartesiantoEquatorial(s, observer)
    return ra, dec


"""
2-PT CORRELATION FUNCTIONS & POWER SPECTRA
"""
def linear_2pcf(z, r, cosmo_model='planck15', runit=u.Mpc/cu.littleh): # , k=np.logspace(-5.0, 2.0, 500)):
    """
    Return the 2-pt. matter autocorrelation function at redshift `z` and scales `r` as predicted by linear theory from Colossus.
    """
    cosmo = cosmology.setCosmology(cosmo_model, persistence='r')  # persistence='r' sets this to read-only
    r = r.to(runit).value if isinstance(r, u.Quantity) else r       # ensures that any units agree
    return cosmo.correlationFunction(r, z=z) 

def nonlinear_2pcf(z, cosmo_model='planck15'):
    """
    Return the 2-pt. nonlinear matter autocorrelation function at redshift `z`. Uses the `symbolic_pofk` package (Bartlett et al. 2024)
    to get the nonlinear P(k)
    """
    cosmo = cosmology.setCosmology(cosmo_model, persistence='r')  # persistence='r' sets this to read-only

    # Define k range
    kmin = 1e-3
    kmax = 10
    nk = 400
    k = np.logspace(np.log10(kmin), np.log10(kmax), nk)

    pk_halofit = syrenhalofit.run_halofit(k, cosmo.sigma8, cosmo.Om(z), cosmo.Ob(z), cosmo.h, cosmo.ns, a(z), emulator='fiducial',
                                      extrapolate=True, which_params='Bartlett', add_correction=False)

    

    return k, pk_halofit

def Pk_to_xi(k, Pk):
    """
    Transform an input 3D power spectrum P(k) to its correlation function xi(r).
    """
    r, xi = mcfit.P2xi(k)(P)

    return r, xi


"""
DATA MANIPULATION + MISC HELPERS
"""
def get_subsample(data, nx=100, verbose=False):
    """
    Randomly sample 1/nxth entries of a data set.
    """
    n = len(data)//nx
    if verbose:
        print(f"subsampling {n} random particles...")
    idx = np.random.choice(len(data), size=n, replace=False)  # get random indices
    return data[idx]

def eval_Gaussian(loc, sigma, mean=0.):
    """
    Evaluate a Gaussian with `sigma` and `mean` at point `loc`.
    """
    pre = 1 / (sigma * np.sqrt(2 * np.pi))
    exp = -(loc-mean)**2 / (2 * sigma**2)
    return pre * np.e**exp

def remove_values(array, minimum=None, maximum=None, axis=0, verbose=True):
    """
    Remove any values in an input `array` outside of `minimum` and `maximum`, along `axis`.
    """
    idx = np.full(array.shape[axis], True)
    if minimum is not None:
        idx = idx & np.all(array >= minimum, axis=axis+1)
    if maximum is not None:
        idx = idx & np.all(array <= maximum, axis=axis+1)
    if verbose and np.sum(~idx):
        print(f"removing {np.sum(~idx)} values")
    array = np.delete(array, ~idx, axis=axis)
    return array

def r_from_rppi(rp, pi, max_ravg=None):
    """
    Return a 2D array of the total separation r at each point on a grid of (`rp`,`pi`),
    separations perpendicular to and parallel to the LOS.
    """
    r_arr = np.empty((len(rp), len(pi)))
    for i, rp_ in enumerate(rp):
        for j, pi_ in enumerate(pi):
            r_arr[i,j] = np.sqrt(rp_**2 + pi_**2)
            if max_ravg is not None and r_arr[i,j] > max_ravg:
                print(f"r to evaluate ({r_arr[i,j]:.3f}) is greater than max r_avg ({max_ravg:.3f})!")
                continue
    return r_arr

def interpolate_number_densities(redshifts, target_ns):
    """
    Linearly interpolate input number densities `target_ns` between the min and max `redshifts`.
    """
    assert len(redshifts) == len(target_ns), "input redshifts and target_ns must have the same length"
    if isinstance(target_ns, u.Quantity):
        target_ns = target_ns.value
    # np.interp() needs its x input to _increase_ monotonically, so we need to sort the arrays
    sorted_idx = np.argsort(redshifts)
    redshifts_sorted = redshifts[sorted_idx]
    target_ns_sorted = target_ns[sorted_idx]
    target_ns_interp = np.interp(redshifts_sorted,
                                [redshifts_sorted[0], redshifts_sorted[-1]], # min and max redshift
                                [target_ns_sorted[0], target_ns_sorted[-1]]) # min and max traget n
    return target_ns_interp[sorted_idx]