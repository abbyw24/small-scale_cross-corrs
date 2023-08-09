import numpy as np
from colossus.cosmology import cosmology

def get_subsample(data, nx=100):
    """Randomly sample 1/nxth entries of a data set."""
    n = len(data)//nx
    print(f"subsampling {n} random particles...")
    idx = np.random.choice(len(data), size=n, replace=False)  # get random indices
    return data[idx]

def linear_2pcf(z, r, cosmo_model='planck15', k=np.logspace(-5.0, 2.0, 500)):
        """
        Return the 2-pt. matter autocorrelation function at redshift `z` and scales `r` as predicted by linear theory from Colossus.
        """
        cosmo = cosmology.setCosmology(cosmo_model, persistence='r')  # persistence='r' sets this to read-only
        # matter power spectrum
        Pk = cosmo.matterPowerSpectrum(k)  # defaults to the approximation of Eisenstein & Hu 1998
        # 2-pt. matter-matter correlation function is an integral over the power spectrum:
        return cosmo.correlationFunction(r, z=z) 