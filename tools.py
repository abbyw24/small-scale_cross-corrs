import numpy as np
from colossus.cosmology import cosmology
import astropy.units as u
import astropy.constants as c
from astropy.cosmology import Planck15 as cosmo


def get_subsample(data, nx=100, verbose=False):
    """Randomly sample 1/nxth entries of a data set."""
    n = len(data)//nx
    if verbose:
        print(f"subsampling {n} random particles...")
    idx = np.random.choice(len(data), size=n, replace=False)  # get random indices
    return data[idx]

def linear_2pcf(z, r, cosmo_model='planck15', runit=u.Mpc/u.littleh): # , k=np.logspace(-5.0, 2.0, 500)):
        """
        Return the 2-pt. matter autocorrelation function at redshift `z` and scales `r` as predicted by linear theory from Colossus.
        """
        cosmo = cosmology.setCosmology(cosmo_model, persistence='r')  # persistence='r' sets this to read-only
        # matter power spectrum
        # Pk = cosmo.matterPowerSpectrum(k)  # defaults to the approximation of Eisenstein & Hu 1998
        # 2-pt. matter-matter correlation function is an integral over the power spectrum:
        r = r.to(runit).value if isinstance(r, u.Quantity) else r
        return cosmo.correlationFunction(r, z=z) 

def perh():
    return (cosmo.H(0.) / 100 * u.Mpc / u.km * u.s) / u.littleh

def get_dx(z, sigma_z):
    return sigma_z * (1 + z) * c.c.to(u.km/u.s) / cosmo.H(z) * perh()

def eval_Gaussian(loc, sigma, mean=0.):
    pre = 1 / (sigma * np.sqrt(2 * np.pi))
    exp = -(loc-mean)**2 / (2 * sigma**2)
    return pre * np.e**exp

def redshift_to_comov(z, cosmo=cosmo):
    r = cosmo.comoving_distance(z) * perh() # convert to Mpc/h
    return r

def theta_to_r_comov(theta, redshift):
    return (theta * u.deg * cosmo.kpc_comoving_per_arcmin(redshift).to(u.Mpc/u.deg) * perh())
def r_comov_to_theta(r, redshift):
    return (r / (cosmo.kpc_comoving_per_arcmin(redshift).to(u.Mpc/u.deg) * perh()))

def CartesiantoEquatorial(pos, observer=[0,0,0]):
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
    Returns the (RA,Dec) coordinates given (x,y,z) galaxy coordinates and comoving distance `chi`
    to the observer.
    """
    # convert photometric sample to (RA,Dec), setting LOS positions to box center
    s = np.copy(sample)
    s[:,2] = 0 * s.unit
    if isinstance(chi, u.Quantity):
        observer = np.array([0, 0, chi.value]) << chi.unit
    else:
        observer = np.array([0, 0, chi])
    ra, dec = CartesiantoEquatorial(s, observer)
    return ra, dec


def remove_values(array, minimum=None, maximum=None, axis=0, verbose=True):
    idx = np.full(array.shape[axis], True)
    if minimum is not None:
        idx = idx & np.all(array >= minimum, axis=axis+1)
    if maximum is not None:
        idx = idx & np.all(array <= maximum, axis=axis+1)
    if verbose and np.sum(~idx):
        print(f"removing {np.sum(~idx)} values")
    array = np.delete(array, ~idx, axis=axis)
    return array


def r_from_rppi(rp, pi):
    r_arr = np.empty((len(rp), len(pi)))
    for i, rp_ in enumerate(rp):
        for j, pi_ in enumerate(pi):
            r_arr[i,j] = np.sqrt(rp_**2 + pi_**2)
            # if r_arr[i,j] > max(r_avg):
            #     print(f"r to evaluate ({r_arr[i,j]:.3f} Mpc/h) is greater than max r_avg ({max(r_avg):.3f})!")
            #     continue
    return r_arr