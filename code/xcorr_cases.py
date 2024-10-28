import numpy as np
import time
import os
import sys
import astropy.units as u

from illustris_sim import TNGSim
from xcorr import Xcorr
import tools


class SPHEREx_Xcorr(Xcorr):
    """
    Class to manage SPHEREx-like galaxy and dark matter auto- and cross-correlations
    from IllustrisTNG and linear theory.
    Inherits properties from the Xcorr() class.
    """

    def __init__(self, snapshots, sigma_z,
                    density_type='interpolated',
                    density=None,                       # optional fixed density input
                    rpmin=0.1, rpmax=60, nrpbins=10,    # bins for projected c.f.
                    pimax_frac=0.45,                    # fraction of the boxsize for pimax
                    rmin=0.1, rmax=100, nbins=10,       # bins for full 3D c.f.
                    nrepeats=10, periodic=True, randmult=3,   # other inputs for both projected and full c.f.
                    sim='TNG300-3',
                    scratch='/scratch1/08811/aew492'):

        # check inputs (all variables passed to Xcorr() are checked within that class)
        assert density_type.lower() in ['interpolated', 'fixed', 'target'], \
            "density_type must be one of 'interpolated', 'fixed', or 'target'"

        # inputs (other attributes are set when initiating XCorr() class, below)
        self.sigma_z = sigma_z
        self.density_type = density_type

        # loads key info, including redshifts and distance to the center of each snapshot
        super().__init__(snapshots, dNdz=None,  # dNdz is None until we get photometric weights
                        rpmin=rpmin, rpmax=rpmax, nrpbins=nrpbins,
                        pimax_frac=pimax_frac,
                        rmin=rmin, rmax=rmax, nbins=nbins,
                        nrepeats=nrepeats, periodic=periodic, randmult=randmult,
                        sim=sim, scratch=scratch)

        # target number densities
        self._get_number_densities(density)

        # now we can get the photometric distribution, determined by sigma_z
        self._get_photometric_distribution()

        # bias file name
        self.bias_fn = os.path.join(self.scratch, self.sim, 'bias',
                                f'bias_z-{min(self.redshifts):.2f}-{max(self.redshifts):.2f}_sigmaz-{self.sigma_z}' + \
                                self.ns_tag + '.npy')


    def _get_number_densities(self, density):

        if density is not None:
            assert self.density_type == 'fixed', "density_type must be 'fixed' if density is not None"

        # get the "target" number densities from the SPHEREx lookup table in TNGSim()
        self.target_ns = np.array([
            TNGSim(self.sim, snapshot=snapshot).survey_params('SPHEREx', '', self.sigma_z).n_Mpc3 \
            for snapshot in self.snapshots
        ]) * (u.littleh / u.Mpc)**3  # return units on the outside

        # do we want to keep these as they are, or interpolate,
        #   or take the mean for constant number density across all snapshots?
        if self.density_type == 'target':
            self.ns = self.target_ns
            self.ns_tag = '_ns_target'
        elif self.density_type == 'interpolated':
            self.ns = tools.interpolate_number_densities(self.redshifts, self.target_ns)
            self.ns_tag = '_ns_interp'
        else:
            assert self.density_type == 'fixed'
            self.ns_tag = '_ns_fixed'
            if density is not None:
                self.density = density.to((u.littleh / u.Mpc)**3) if isinstance(density, u.Quantity) \
                    else density * (u.littleh / u.Mpc)**3
                self.ns = self.density * np.ones(len(self.snapshots))
            else:
                self.ns = np.mean(self.target_ns) * np.ones(len(self.snapshots))


    def _get_photometric_distribution(self):
        """
        Gaussian distribution with width `dx` set by `sigma_z`.
        """

        self.central_chi = np.nanmean(self.chis)
        self.dx = tools.get_dx(np.nanmean(self.redshifts), self.sigma_z)

        # at the snapshot centers
        self.W_phot = [
            tools.eval_Gaussian(chi.value, self.dx.value, mean=self.central_chi.value) for chi in self.chis
        ]
        # this is dN/dz ! (this method comes from parent XCorr() class)
        self.set_dNdz(self.W_phot)

    
    def construct_spectroscopic_galaxy_samples(self, verbose=False):
        """
        Construct a set of galaxy positions, to constitute the spectroscopic sample
        in each snapshot.
        """

        gal_pos_specs = [] # where to store galaxy positions
        for i, snapshot in enumerate(self.snapshots):
            # instantiate simulation
            sim = TNGSim(self.sim, snapshot=snapshot)
            # get the positions of the subhalos that we're counting as galaxies
            gal_pos_spec = sim.subhalo_pos()[sim.gal_idx('','SPHEREx',
                                                sigma_z=self.sigma_z,
                                                n=self.ns[i],
                                                verbose=verbose)]
            # remove any values which (still not sure why) fall just outside of the boxsize
            #   (this only happens with one galaxy every few snapshots)
            gal_pos_spec = tools.remove_values(gal_pos_spec, minimum=0, maximum=sim.boxsize, verbose=verbose)
            gal_pos_spec -= sim.boxsize / 2  # center at zero
            assert np.all(gal_pos_spec >= -sim.boxsize / 2) and np.all(gal_pos_spec <= sim.boxsize / 2), \
                f"galaxy positions out of bounds! min = {np.nanmin(gal_pos_spec):.3f}, max = {np.nanmax(gal_pos_spec):.3f}"
            gal_pos_specs.append(gal_pos_spec)
        
        self.gal_pos_specs = gal_pos_specs


class HSC_Xcorr(Xcorr):
    """
    Class to manage Hyper Suprime-Cam-like galaxy and dark matter auto- and cross-correlations
    from IllustrisTNG and linear theory.
    Inherits properties from the Xcorr() class.

    Bugs/Comments:
        - Requires that input `snapshots` correspond to the redshift range of input `photzbin`. (Ideally, the
        redshifts of the input snapshots center at peak dN/dz of the HSC photzbin.)
    """

    def __init__(self, snapshots, photzbin,
                    density_type='interpolated',
                    density=None,                       # optional fixed density input
                    rpmin=0.1, rpmax=60, nrpbins=10,    # bins for projected c.f.
                    pimax_frac=0.45,                    # fraction of the boxsize for pimax
                    rmin=0.1, rmax=100, nbins=10,       # bins for full 3D c.f.
                    nrepeats=10, periodic=True, randmult=3,   # other inputs for both projected and full c.f.
                    sim='TNG300-3',
                    scratch='/scratch1/08811/aew492'):
        
        # which redshift bin
        assert photzbin in (0,1,2,3), "photometric redshift bin must be 0, 1, or 2"
        self.photzbin = photzbin

        # reference sample
        self.reference_survey = 'DESI'
        self.reference_tracer = 'ELG'

        # loads key info, including redshifts and distance to the center of each snapshot
        super().__init__(snapshots, dNdz=None,  # dNdz is None until we get photometric weights
                        rpmin=rpmin, rpmax=rpmax, nrpbins=nrpbins,
                        pimax_frac=pimax_frac,
                        rmin=rmin, rmax=rmax, nbins=nbins,
                        nrepeats=nrepeats, periodic=periodic, randmult=randmult,
                        sim=sim, scratch=scratch)

        # target number densities
        self._get_number_densities(density)

        # now we can get the photometric distribution based on input photzbin
        self._get_photometric_distribution()


    def _get_number_densities(self, density):
        """
        Using a DESI-like reference sample.

        Bugs/Comments:
        - hard-coded to use DESI ELGs rather than LRGs (add a toggle depending on photzbin?)

        """

        if density is not None:
            assert self.density_type == 'fixed', "density_type must be 'fixed' if density is not None"

        # get the "target" number densities from the DESI lookup table in TNGSim()
        self.target_ns = np.array([
            TNGSim(self.sim, snapshot=snapshot).survey_params(self.reference_survey, self.reference_tracer, self.sigma_z).n_Mpc3 \
            for snapshot in self.snapshots
        ]) * (u.littleh / u.Mpc)**3  # return units on the outside

        # do we want to keep these as they are, or interpolate,
        #   or take the mean for constant number density across all snapshots?
        if self.density_type == 'target':
            self.ns = self.target_ns
            self.ns_tag = '_ns_target'
        elif self.density_type == 'interpolated':
            self.ns = tools.interpolate_number_densities(self.redshifts, self.target_ns)
            self.ns_tag = '_ns_interp'
        else:
            assert self.density_type == 'fixed'
            self.ns_tag = '_ns_fixed'
            if density is not None:
                self.density = density.to((u.littleh / u.Mpc)**3) if isinstance(density, u.Quantity) \
                    else density * (u.littleh / u.Mpc)**3
                self.ns = self.density * np.ones(len(self.snapshots))
            else:
                self.ns = np.mean(self.target_ns) * np.ones(len(self.snapshots))


    def _get_photometric_distribution(self):
        """
        Photometric distribution dN/dz based on Fig. 4 of Dalal et al. (2023) (https://arxiv.org/abs/2304.00701)
        """

        # load in data interpolated from Fig. 4
        photzdata = np.load(f'../data/HSC/HSC_dNdz_zbin{self.photzbin}.npy', allow_pickle=True).item()

        # interpolate to get the P(z) at each snapshot center
        self.pz = np.interp(self.redshifts, photzdata['z'], photzdata['pz'])

        # make sure that P(z) integrates to 1
        print("photometric distribution:", self.redshifts, self.pz)
        print(sum(self.pz))

        # this is dN/dz ! (this method comes from parent XCorr() class)
        self.set_dNdz(self.W_phot)

    
    def construct_spectroscopic_galaxy_samples(self, verbose=False):
        """
        Construct a set of galaxy positions, to constitute the spectroscopic sample
        in each snapshot.
        """

        gal_pos_specs = [] # where to store galaxy positions
        for i, snapshot in enumerate(self.snapshots):
            # instantiate simulation
            sim = TNGSim(self.sim, snapshot=snapshot)
            # get the positions of the subhalos that we're counting as galaxies
            gal_pos_spec = sim.subhalo_pos()[sim.gal_idx(self.reference_tracer, self.reference_survey,
                                                sigma_z=self.sigma_z,
                                                n=self.ns[i],
                                                verbose=verbose)]
            # remove any values which (still not sure why) fall just outside of the boxsize
            #   (this only happens with one galaxy every few snapshots)
            gal_pos_spec = tools.remove_values(gal_pos_spec, minimum=0, maximum=sim.boxsize, verbose=verbose)
            gal_pos_spec -= sim.boxsize / 2  # center at zero
            assert np.all(gal_pos_spec >= -sim.boxsize / 2) and np.all(gal_pos_spec <= sim.boxsize / 2), \
                f"galaxy positions out of bounds! min = {np.nanmin(gal_pos_spec):.3f}, max = {np.nanmax(gal_pos_spec):.3f}"
            gal_pos_specs.append(gal_pos_spec)
        
        self.gal_pos_specs = gal_pos_specs