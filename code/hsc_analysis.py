import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import astropy.units as u
import astropy.cosmology.units as cu
import pickle
import os
import sys

sys.path.insert(0, '/work2/08811/aew492/frontera/venv/illustris3.9/lib/python3.9/site-packages')
from illustris_sim import TNGSim
import tools
from xcorr_cases import HSC_Xcorr


def main():
    # main inputs
    snapshotss = [
        np.arange(59, 84), # center at z~0.45
        np.arange(47, 70), # center at z=0.75
        np.arange(40, 62),  # center at z=1.
        np.arange(35, 54)  # center at z=1.25
    ]

    # the photo-z bins
    photzbins = np.arange(0, 4)

    # galaxy number density
    density = 2e-3 * (cu.littleh / u.Mpc)**3

    # reference tracers
    tracers = ['ELG', 'LRG']

    # other arguments for the analysis
    hsc_kwargs = dict(density_type='fixed', density=density, reference_survey='DESI')

    for (snapshots, photzbin) in zip(snapshotss, photzbins):
        for tracer in tracers:
            run_analysis(snapshots, photzbin, reference_tracer=tracer, overwrite=True, **hsc_kwargs)


def run_analysis(snapshots, photzbin, overwrite=False, **hsc_kwargs):

    print(f"starting snapshots {min(snapshots)}-{max(snapshots)}, photzbin={photzbin}")

    # instantiate spherex set for cross-correlation
    X = HSC_Xcorr(snapshots, photzbin, **hsc_kwargs)

    # tags for saving
    density_tag = X.ns_tag
    if X.density_type == 'fixed':
        density_tag += f'-{X.density.value:.1e}'
    case_tag = f'z-{min(X.redshifts):.2f}-{max(X.redshifts):.2f}_photzbin{X.photzbin}' + density_tag + \
                f'_{X.reference_survey}-{X.reference_tracer}'
    save_dir = os.path.join(X.scratch, 'TNG300-3/xcorr_res/HSC', case_tag)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_fn = os.path.join(save_dir, f'xcorr_object.pkl')
    if os.path.exists(save_fn) and not overwrite:
        print(f"already exists at {save_fn}")
        return

    # plot key info for this set
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11,4), tight_layout=True)
    # number densities
    ax0.plot(X.redshifts, X.target_ns, 'k.', alpha=0.8, label='From lookup table')
    ax0.plot(X.redshifts, X.ns, 'k-', alpha=0.5, label='For mock TNG sample')
    ax0.grid(alpha=0.5, lw=0.5)
    ax0.set_xlabel(r'Redshift $z$')
    ax0.set_ylabel(r'Number density (Mpc/$h)^{-3}$')
    ax0.legend()
    ax0.set_title(f'Galaxy number densities, {X.density_type}')
    # dN/dz
    ax1.plot(X.chis, X.dNdz, 'k.-', alpha=0.8, lw=0.5)
    ax1.grid(alpha=0.5, lw=0.5)
    ax1.set_xlabel(r'Comoving distance $\chi$ (Mpc/h)')
    ax1.set_ylabel(r'$W_\mathrm{phot}$')
    ax1.set_title(f'Photometric distribution dN/dz: bin {X.photzbin}')
    # save
    fig.savefig(os.path.join(save_dir, 'densities_and_dNdz.png'))

    # construct the spectroscopic galaxies in each snapshot
    X.construct_spectroscopic_galaxy_samples()

    # plot reference sample
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
    ax.set_title(f'{X.sim} snapshots: {X.reference_survey}-like {X.reference_tracer}')
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