"""
Functions to select certain types of galaxies from TNG subhalos, given an IllustrisSim() instance.
"""

import numpy as np
import astropy.units as u
import astropy.cosmology.units as cu

from survey_params_gal import eBOSS_param, DESI_param


# def get_LRG_galaxies(sim, n=2e-4, sSFR_cutval=-9.09, V=(205 * u.Mpc/cu.littleh)**3):
#     """
#     Make cuts in subhalo sSFR and stellar mass to select for LRGs and reach a target galaxy number density,
#     as outlined in Sullivan, Prijon, & Seljak (2023).
#     """

#     subhalos_nonzero, stellar_mass, SFR = get_nonzero_SFR_and_stellar_mass(sim)
    
#     # specific star formation rate (sSFR) = star formation rate per stellar mass; and make cut
#     sSFR = SFR / stellar_mass
#     sSFR_cut = (np.log10(sSFR.value) < sSFR_cutval)
#     subhalos_LRG = subhalos_nonzero[sSFR_cut]
    
#     # target number of galaxies
#     target_N = int(V * n * (u.Mpc / cu.littleh)**(-3))
#     # sort subhalos by decreasing stellar mass, then take only the first target_N
#     galaxies = subhalos_LRG[np.argsort(stellar_mass[sSFR_cut])[::-1]][:target_N]
#     return galaxies

# def get_ELG_galaxies(sim, n=2e-4, sSFR_cutval=-9.09, V=(205 * u.Mpc/cu.littleh)**3):
#     """
#     Make cuts in subhalo sSFR and stellar mass to select for ELGs and reach a target galaxy number density,
#     as outlined in Sullivan, Prijon, & Seljak (2023).
#     """

#     subhalos_nonzero, stellar_mass, SFR = get_nonzero_SFR_and_stellar_mass(sim)
    
#     # specific star formation rate (sSFR) = star formation rate per stellar mass; and make cut
#     sSFR = SFR / stellar_mass
#     sSFR_cut = (np.log10(sSFR.value) > sSFR_cutval)
#     subhalos_ELG = subhalos_nonzero[sSFR_cut]
    
#     # target number of galaxies
#     target_N = int(V * n * (u.Mpc / cu.littleh)**(-3))
#     # sort subhalos by decreasing star formation rate, then take only the first target_N
#     galaxies = subhalos_ELG[np.argsort(SFR[sSFR_cut])[::-1]][:target_N]
#     return galaxies


### UPDATED FUNCTIONS WITH CORRECT NUMBER DENSITIES
def LRG_idx(sim, survey='DESI', sSFR_cutval=-9.09):
    """
    Make cuts in subhalo sSFR and stellar mass to select for LRGs and reach a target galaxy number density,
    as outlined in Sullivan, Prijon, & Seljak (2023).

    S, P & S 2023 uses two different cutoff values for sSFR, log10(sSFR) =
        1. -9.09 (https://arxiv.org/abs/2210.10068 using MilleniumTNG)
        2. -9.23 (https://arxiv.org/abs/2011.05331 using TNG300-1)
    """
    # target number of galaxies
    V = sim.boxsize**3
    n = survey_params(survey, sim.redshift, 'LRG').n_Mpc3
    target_N = int(V * n)

    # load values from the simulation
    subhalos_nonzero, stellar_mass, SFR = get_nonzero_SFR_and_stellar_mass(sim)
    
    # specific star formation rate (sSFR) = star formation rate per stellar mass; and make cut
    sSFR = sim.SFR() / sim.stellar_mass()
    sSFR_cut = (np.log10(sSFR.value) < sSFR_cutval)

    # sort subhalos by decreasing stellar mass, then take only the first target_N
    return np.argsort(stellar_mass[sSFR_cut])[::-1][:target_N]