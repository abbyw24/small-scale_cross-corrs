"""
Functions to select certain types of galaxies from TNG subhalos, given an IllustrisSim() instance.
"""

import numpy as np
import astropy.units as u
import astropy.cosmology.units as cu


def get_LRG_galaxies(sim, n=2e-4, sSFR_cutval=-9.09, V=(205 * u.Mpc/cu.littleh)**3):
    """
    Make cuts in subhalo sSFR and stellar mass to select for LRGs and reach a target galaxy number density,
    as outlined in Sullivan, Prijon, & Seljak (2023).
    """

    subhalos_nonzero, stellar_mass, SFR = get_nonzero_SFR_and_stellar_mass(sim)
    
    # specific star formation rate (sSFR) = star formation rate per stellar mass; and make cut
    sSFR = SFR / stellar_mass
    sSFR_cut = (np.log10(sSFR.value) < sSFR_cutval)
    subhalos_LRG = subhalos_nonzero[sSFR_cut]
    
    # target number of galaxies
    target_N = int(V * n * (u.Mpc / cu.littleh)**(-3))
    # sort subhalos by decreasing stellar mass, then take only the first target_N
    galaxies = subhalos_LRG[np.argsort(stellar_mass[sSFR_cut])[::-1]][:target_N]
    return galaxies

def get_ELG_galaxies(sim, n=2e-4, sSFR_cutval=-9.09, V=(205 * u.Mpc/cu.littleh)**3):
    """
    Make cuts in subhalo sSFR and stellar mass to select for ELGs and reach a target galaxy number density,
    as outlined in Sullivan, Prijon, & Seljak (2023).
    """

    subhalos_nonzero, stellar_mass, SFR = get_nonzero_SFR_and_stellar_mass(sim)
    
    # specific star formation rate (sSFR) = star formation rate per stellar mass; and make cut
    sSFR = SFR / stellar_mass
    sSFR_cut = (np.log10(sSFR.value) > sSFR_cutval)
    subhalos_ELG = subhalos_nonzero[sSFR_cut]
    
    # target number of galaxies
    target_N = int(V * n * (u.Mpc / cu.littleh)**(-3))
    # sort subhalos by decreasing star formation rate, then take only the first target_N
    galaxies = subhalos_ELG[np.argsort(SFR[sSFR_cut])[::-1]][:target_N]
    return galaxies


""" HELPER FUNCTIONS """
def get_nonzero_SFR_and_stellar_mass(sim, return_all=True):
    """Returns the subhalos with nonzero star formation rate and stellar mass, given an IllustrisSim() instance."""
    
    # load the subhalo positions, star formation rate and stellar masses
    sim.load_subfind_subhalos()  # load table of subhalos and removes any flagged
    sim.load_stellar_mass()      # pull out stellar mass and gives proper units
    sim.load_SFR()               # pull out SFR and gives proper units

    # take only the subhalos with nonzero stellar mass and SFR
    idx_nonzero = (sim.stellar_mass > 0) & (sim.SFR > 0)
    subhalos_nonzero = sim.subhalo_info[idx_nonzero]

    if return_all:
        return subhalos_nonzero, sim.stellar_mass[idx_nonzero], sim.SFR[idx_nonzero]
    else:
        return subhalos_nonzero