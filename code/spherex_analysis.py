import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import astropy.units as u
import pickle
import os
import sys

from illustris_sim import TNGSim
import tools
from xcorr_cases import SPHEREx_Xcorr


def main():
    # main inputs
    snapshotss = [
        np.arange(25, 37),  # center at z=2.3
        np.arange(43, 58), # center at z=1.
        np.arange(59, 76) # center at z~0.4
    ]
    # the 5 redshift error bins in SPHEREx
    sigma_zs = [
        0.003,
        0.01,
        0.03,
        0.1,
        0.2
    ]
    # galaxy number density
    density = 2e-3 * (u.littleh / u.Mpc)**3

    for snapshots in snapshotss:
        for sigma_z in sigma_zs:
            run_analysis(snapshots, sigma_z, density, overwrite=True)


def run_analysis(snapshots, sigma_z, density, overwrite=False):

    print(f"starting snapshots {min(snapshots)}-{max(snapshots)}, sigma_z={sigma_z}")
    # instantiate spherex set for cross-correlation
    X = SPHEREx_Xcorr(snapshots, sigma_z, density_type='fixed', density=density)

    case_tag = f'z-{min(X.redshifts):.2f}-{max(X.redshifts):.2f}_sigma-z-{X.sigma_z}_ns-{X.density_type}_{density.value:.1e}'
    
    save_dir = os.path.join(X.scratch, 'TNG300-3/xcorr_res', case_tag)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_fn = os.path.join(save_dir, f'xcorr_object.pkl')
    if os.path.exists(save_fn) and not overwrite:
        print(f"already exists at {save_fn}")
        continue
    
    # plot key info for this set
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11,4), tight_layout=True)
    # number densities
    ax0.plot(X.redshifts, X.target_ns, 'k.', alpha=0.8, label='From lookup table')
    ax0.plot(X.redshifts, X.ns, 'k-', alpha=0.5, label='Interpolated')
    ax0.grid(alpha=0.5, lw=0.5)
    ax0.set_xlabel(r'Redshift $z$')
    ax0.set_ylabel(r'Target number density (Mpc/$h)^{-3}$')
    ax0.legend()
    ax0.set_title(r'Galaxy number densities, $\sigma_z=$'f'{X.sigma_z}')
    # dN/dz
    ax1.plot(X.chis, X.dNdz, 'k.-', alpha=0.8, lw=0.5)
    ax1.grid(alpha=0.5, lw=0.5)
    ax1.set_xlabel(r'Comoving distance $\chi$ (Mpc/h)')
    ax1.set_ylabel(r'$W_\mathrm{phot}$')
    ax1.set_title(r'Photometric distribution dN/dz, $\sigma_z=$'f'{X.sigma_z}')
    # save
    fig.savefig(os.path.join(save_dir, 'densities_and_dNdz.png'))
    
    # compute the spectroscopic galaxies in each snapshot
    X.construct_spectroscopic_galaxy_samples()
    
    # plot
    norm = mpl.colors.Normalize(vmin=min(X.redshifts), vmax=max(X.redshifts))
    smap = mpl.cm.ScalarMappable(norm=norm, cmap='turbo')
    fig, ax = plt.subplots(figsize=(15,2.6), tight_layout=True)
    for i, chi in enumerate(X.chis):
        gal_pos_spec_ = np.copy(X.gal_pos_specs[i])
        gal_pos_spec_[:,2] += chi
        kwargs = dict(c=smap.to_rgba(X.redshifts[i]), ls='None')
        ax.plot(gal_pos_spec_[:,2], gal_pos_spec_[:,0], marker=',', alpha=0.4, **kwargs)
        ax.plot(chi, 0, marker='o', c=smap.to_rgba(X.redshifts[i]), mec='k', zorder=100)
        ax.axvline((chi - X.boxsize/2).value, alpha=0.8, **kwargs)
        ax.axvline((chi + X.boxsize/2).value, alpha=0.8, **kwargs)
    ax.axhline(0, c='k', alpha=0.5, lw=0.5)
    ax.set_aspect('equal')
    ax.set_xlim((min(X.chis)-0.6*X.boxsize).value, (max(X.chis)+0.6*X.boxsize).value)
    ax.set_xlabel(r'LOS comoving distance $\chi$ (Mpc/h)')
    ax.set_ylabel(r'LOS $\perp$ (Mpc/h)')
    ax.set_title(f'{X.sim} snapshots, 'r'$\sigma_z=$'f'{X.sigma_z}')
    fig.colorbar(smap, ax=ax, label='Redshift $z$', pad=0.01)
    # save
    fig.savefig(os.path.join(save_dir, 'gal_pos_specs.png'))
    
    # compute angular cross-correlation
    X.compute_angular_xcorrs()
    
    # save the class
    with open(save_fn, 'wb') as output:
        pickle.dump(X, output)
    print(f"saved class instance to {save_fn}")


if __name__=='__main__':
    main()