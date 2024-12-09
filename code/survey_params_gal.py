"""
Modified copy from Yun-Ting Chen.
Class to get source number density as a function of redshift for SDSS and DESI spectroscopic surveys.
+ added analogous class for SPHEREx
"""

import numpy as np
import os
import astropy.units as u
import astropy.cosmology.units as cu
from astropy.cosmology import Planck15 as cosmo


class eBOSS_param:
    '''
    https://ui.adsabs.harvard.edu/abs/2016AJ....151...44D/abstract
    deltaz = sigma_z/(1+z) ~ 1e-3 (p.9)
    ELG:
    https://ui.adsabs.harvard.edu/abs/2017MNRAS.471.3955R/abstract
    '''
    def __init__(self, z=0, tracer_name='LRG',
                 field_name=None, ns_field_name=None):
        
        self.tracer_names = ['CMASS', 'LRG', 'ELG', 'QSO']        
        self.z = z
        self.tracer_name = tracer_name
        self.print_name = 'eBOSS ' + tracer_name
        self.print_name_survey = '(e)BOSS'
        if tracer_name == 'CMASS':
            self.print_name = self.print_name[1:]
        self._calc_params()

    
    def _calc_params(self):
        self._get_area()
        self._get_density()
    
    def _get_area(self, field_name=None):
        
        self.survey_area_eBOSS = 7500 # [deg^2]
        self.survey_area_early = 3193 # obs in fisrt year[deg^2]
        self.survey_area_ELG_SGC = 620 # [deg^2]
        self.survey_area_ELG_NGC = 600 # [deg^2]
        self.survey_area_CMASS = 10000 # [deg^2]
        
        if self.tracer_name != 'ELG':
            Adeg = self.survey_area_eBOSS
            if field_name == 'early':
                Adeg = self.survey_area_early
            elif self.tracer_name == 'CMASS':
                Adeg = self.survey_area_CMASS
        else:
            Adeg = self.survey_area_ELG_SGC
            if field_name == 'NGC':
                Adeg = self.survey_area_ELG_NGC
        
        self.survey_area_deg = Adeg
        
        Asr = Adeg * (u.deg**2).to(u.sr)
        self.survey_area_sr = Asr
        
        self.fsky = Asr / 4 / np.pi
        self.lmin = np.pi / np.sqrt(Asr)
        
        return     
        
    def _get_density(self, ns_field_name=None):
        
        if self.tracer_name == 'LRG':
            table = self.LRG_density_table()
            ns_field_name = 'ns_zconf1' if ns_field_name is None else ns_field_name
        elif self.tracer_name == 'QSO':
            table = self.QSO_density_table()
            ns_field_name = 'ns' if ns_field_name is None else ns_field_name
        elif self.tracer_name == 'ELG':
            table = self.ELG_density_table()
            ns_field_name = 'ns_SGC' if ns_field_name is None else ns_field_name
        elif self.tracer_name == 'CMASS':
            table = self.CMASS_density_table()
            ns_field_name = 'ns'
            
        zbins_min, zbins_max = table['zbins_min'], table['zbins_max']
        zidx = np.where((self.z >= zbins_min) & (self.z < zbins_max))[0]

        if len(zidx) == 0:
            self.n_z_deg2 = 0
            self.n_zbin_deg2 = 0
            self.n_z_sr = 0
            self.n_Mpc3 = 0
            return
        
        # [dN/d(deg^2)/(dzbin)]
        self.n_zbin_deg2 = table[ns_field_name][zidx[0]]
        self.zbin_min = zbins_min[zidx[0]]
        self.zbin_max = zbins_max[zidx[0]]

        # [dN/dz/d(deg^2)]
        self.n_z_deg2 = self.n_zbin_deg2 / (self.zbin_max - self.zbin_min)
        
        # [dN/dz/dsr]
        self.n_z_sr = self.n_zbin_deg2 * (u.deg**-2).to(u.sr**-1) \
                        / (self.zbin_max - self.zbin_min)
        
        chi1 = cosmo.comoving_distance(self.zbin_max)
        chi2 = cosmo.comoving_distance(self.zbin_min)
        
        dchi = (chi1 - chi2)
        dA = (((chi1 + chi2) / 2) / (u.rad).to(u.deg))**2
        dV = (dchi * dA).value * cosmo.h**3 # [(Mpc/h)**3]
        
        # [dN/d(Mpc/h)^3]
        self.n_Mpc3 = self.n_zbin_deg2 / dV
                
            
    def LRG_density_table(self):
        '''
        Dawson+16 table 1
        ns [deg^-2]
        '''
        zbinedges = np.array([0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8,
                    2.0, 2.1, 2.2, 2.4, 2.6, 2.8, 3.0, 3.5])
        zbins_min = zbinedges[:-1]
        zbins_max = zbinedges[1:]
        zbins = (zbins_min + zbins_max) / 2
        
        ns_zconf0 = np.zeros_like(zbins)
        ns_zconf0[:7]= [0.6, 6.2, 15.2, 15.3, 9.4, 3.2, 0.6]
        
        ns_zconf1 = np.zeros_like(zbins)
        ns_zconf1[:7]= [0.6, 5.9, 14.8, 14.7, 8.7, 2.7, 0.5]
        
        LRGtable = {'zbins': zbins, 'zbins_min': zbins_min, 'zbins_max': zbins_max,
                   'ns_zconf0': ns_zconf0, 'ns_zconf1': ns_zconf1}
        
        return LRGtable
    
    def QSO_density_table(self):
        '''
        Dawson+16 table 1
        ns [deg^-2]
        '''
        zbinedges = np.array([0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8,
                    2.0, 2.1, 2.2, 2.4, 2.6, 2.8, 3.0, 3.5])
        zbins_min = zbinedges[:-1]
        zbins_max = zbinedges[1:]
        zbins = (zbins_min + zbins_max) / 2
        
        ns_new = np.array([1.0, 1.1, 1.4, 1.4, 2.2, 3.6, 8.4, 10.3, 10.3,
                  9.9, 9.2, 4.0, 2.2, 1.8, 1.1, 0.7, 0.3, 0.4])
        
        ns_known= np.array([0.4, 0.4, 0.7, 1.3, 1.5, 1.0, 1.8, 1.8, 2.1, 2.0,
                   1.9, 1.0, 1.6, 4.5, 3.1, 1.4, 0.8, 1.2])
        
        
        QSOtable = {'zbins': zbins, 'zbins_min': zbins_min, 'zbins_max': zbins_max,
                   'ns': ns_new + ns_known, 'ns_new': ns_new, 'ns_known': ns_known}
        
        return QSOtable
    
    def ELG_density_table(self):
        '''
        Raichoor table 4
        ns [deg^-2]
        '''
        zbinedges = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 
                              0.8, 0.9, 1. , 1.1, 1.2, 1.3, 1.4, 1.5])
        zbins_min = zbinedges[:-1]
        zbins_max = zbinedges[1:]
        zbins = (zbins_min + zbins_max) / 2
        
        ns_SGC = np.array([0.2, 1.1, 2.0, 1.9, 1.1, 1.4, 9.2, 56.6,
                 61.6, 31.6, 13.4, 6.4, 2.9, 1.5, 0.7])
        
        ns_NGC = np.array([0.3, 1.1, 2.6, 2.6, 1.7, 2.2, 10.3, 42.0,
                           48.5, 26.3, 12.0, 5.4, 2.5, 0.9, 0.4])
        
        ELGtable = {'zbins': zbins, 'zbins_min': zbins_min, 'zbins_max': zbins_max,
                   'ns_SGC': ns_SGC, 'ns_NGC': ns_SGC}
        
        return ELGtable
    
    def CMASS_density_table(self):
        '''
        Dawson+16 table 1
        ns [deg^-2]
        '''
        zbinedges = np.array([0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        zbins_min = zbinedges[:-1]
        zbins_max = zbinedges[1:]
        zbins = (zbins_min + zbins_max) / 2
        
        ns = np.array([27.3, 45.7, 19.4, 3.5, 0.2, 0.03])

        LRGtable = {'zbins': zbins, 'zbins_min': zbins_min, 'zbins_max': zbins_max,
                   'ns': ns}
        
        return LRGtable    


class DESI_param:
    '''
    https://arxiv.org/pdf/2404.03000 
    '''
    def __init__(self, z=0, tracer_name='ELG'):

        assert tracer_name in ['ELG', 'LRG', 'QSO'], "unknown input tracer_name"

        self.z = z
        self.tracer_name = tracer_name
        self.print_name = 'DESI ' + tracer_name
        self.print_name_survey = 'DESI'
        self._calc_params()

    def _calc_params(self):
        self._get_density()

    def _get_density(self):

        # get density info (z,n)
        z, n = self.density_table()

        # closest data point to the input redshift
        z_idx = np.argmin(np.abs(self.z - z))

        self.n_Mpc3 = n[z_idx]

    def density_table(self):

        z, n = np.load(f'../data/DESI/DESI_{self.tracer_name}_density.npy').T

        # check that the input redshift is within the data range
        assert (self.z >= min(z)) & (self.z <= max(z)), \
            f"input redshift ({self.z}) outside density table range ({min(z):.2f}-{max(z):.2f})"

        n *= 1e-4  # units in plot (Mpc/h)^3

        return z, n


class SPHEREx_param:
    '''
    public products repo at https://github.com/SPHEREx/Public-products

    note unlike the other surveys, there's no distinction between different galaxy types.
    also note that this one is stripped down because the number densities are provided in (h/Mpc)^3 \
    and for my purposes this is the only unit I currently need.
    but here we add the different redshift uncertainties sigma_z.
    '''
    def __init__(self, z=0, tracer_name='ELG', sigma_z=None):
      
        self.z = z
        if sigma_z is None:
            sigma_z = 0.01
            print(f"defaulting to sigma_z={sigma_z}")
        self.sigma_z = sigma_z
        self._set_sigma_z_bin()
        self.tracer_name = tracer_name
        self.print_name = 'SPHEREx ' + tracer_name
        self.print_name_survey = 'SPHEREx'
        self._calc_params()
    
    def _calc_params(self):
        self._get_density()
        

    def _get_density(self):
        
        table = self.galaxy_density_table()
            
        zbins_min, zbins_max = table['zbins_min'], table['zbins_max']
        zidx = np.where((self.z >= zbins_min) & (self.z < zbins_max))[0]

        if len(zidx) == 0:
            self.n_zbin_deg2 = 0
            self.n_z_deg2 = 0
            self.n_z_sr = 0
            self.n_Mpc3 = 0
            return
        
        self.n_Mpc3 = table['ns'][zidx[0]]

        self.bias = table['bs'][zidx[0]]
    
            
    def galaxy_density_table(self):
        '''
        https://github.com/SPHEREx/Public-products/blob/master/galaxy_density_v28_base_cbe.txt

        ns [dN/d(Mpc/h)^3]
        '''
        zbinedges = np.append(np.arange(0.0, 1.0, 0.2), np.arange(1.0, 5., 0.6))
        zbins_min = zbinedges[:-1]
        zbins_max = zbinedges[1:]
        zbins = (zbins_min + zbins_max) / 2        
        dz = np.diff(zbinedges)
        
        ns = np.zeros_like(zbins)
        # number densities in each redshift bin for each redshift uncertainty sample
        ns_all = np.array([
            [0.00997, 0.00411, 0.000501, 7.05e-05, 3.16e-05, 1.64e-05, 3.59e-06, 8.07e-07, 1.84e-06, 1.5e-06, 1.13e-06],
            [0.0123, 0.00856, 0.00282, 0.000937, 0.00043, 5e-05, 8.03e-06, 3.83e-06, 3.28e-06, 1.07e-06, 6.79e-07],
            [0.0134, 0.00857, 0.00362, 0.00294, 0.00204, 0.000212, 6.97e-06, 2.02e-06, 1.43e-06, 1.93e-06, 6.79e-07],
            [0.0229, 0.0129, 0.00535, 0.00495, 0.00415, 0.000796, 7.75e-05, 7.87e-06, 2.46e-06, 1.93e-06, 1.36e-06],
            [0.0149, 0.00752, 0.00327, 0.0025, 0.00183, 0.000734, 0.000253, 5.41e-05, 2.99e-05, 9.41e-06, 2.04e-06]
        ])
        ns[:] = ns_all[self.sigma_z_idx]

        bs = np.zeros_like(zbins)
        # galaxy bias in each redshift bin for each redshift uncertainty sample
        bs_all = np.array([
            [1.3, 1.5, 1.8, 2.3, 2.1, 2.7, 3.6, 2.3, 3.2, 2.7, 3.8],
            [1.2, 1.4, 1.6, 1.9, 2.3, 2.6, 3.4, 4.2, 4.3, 3.7, 4.6],
            [1.0, 1.3, 1.5, 1.7, 1.9, 2.6, 3.0, 3.2, 3.5, 4.1, 5.0],
            [0.98, 1.3, 1.4, 1.5, 1.7, 2.2, 3.6, 3.7, 2.7, 2.9, 5.0],
            [0.83, 1.2, 1.3, 1.4, 1.6, 2.1, 3.2, 4.2, 4.1, 4.5, 5.0]
        ])
        bs[:] = bs_all[self.sigma_z_idx]
        
        table = {'zbins': zbins, 'zbins_min': zbins_min, 'zbins_max': zbins_max,
                   'ns': ns, 'bs': bs}
        
        return table
    

    def _set_sigma_z_bin(self):
        """
        which of 5 redshift uncertainty bins are we in?
        this determines the number densities and galaxy bias in each redshift bin.
        """
        sigma_zs = np.array([0., 0.003, 0.01, 0.03, 0.1, 0.2])
        sigmazbins_min = sigma_zs[:-1]
        sigmazbins_max = sigma_zs[1:]
        sigmazbins = (sigmazbins_min + sigmazbins_max) / 2

        self.sigma_z_idx = np.where((self.sigma_z > sigmazbins_min) & (self.sigma_z <= sigmazbins_max))[0]


class HSC_param:
    '''
    HSC weak lensing tomographic inference: https://arxiv.org/pdf/2211.16516v2 
    HSC Y3 results: http://arxiv.org/pdf/2304.00701 


    '''

    def __init__(self, z=0, zbin=None):

        self.z = z
        self.zbin = zbin

        self._calc_params()

    def _calc_params(self):
        self._get_area()
        self._get_density()

    def _get_area(self):

        self.survey_area_deg = 417 * (u.deg**2) # area of HSC Wide data in all 5 broadbands at full depth (Section 2.1)
        
        self.survey_area_sr = self.survey_area_deg.to(u.sr)
        
        self.fsky = self.survey_area_sr / (4 * np.pi)

    def _get_density(self):

        table = self.density_table()

        z = table['z']
        Pz = table['Pz']
        self.dz = table['dz']
        n = table['n'] << u.arcmin**(-2)
        self.n_sr = n.to(1 / u.sr)

        # total number of galaxies in this redshift bin
        self.N = self.n_sr * self.survey_area_sr

        zidx = np.argmin(np.abs(z[self.zbin] - self.z))
        self.Pz = Pz[zidx]

        # total number of galaxies at this redshift
        self.N_z = self.N * self.Pz * self.dz
        # print(f"total number of galaxies at redshift {self.z:.2f}: {self.N_z:.2f}")

        # comoving volume element: thin shell from z to z+dz
        chi1 = cosmo.comoving_distance(self.z)
        chi2 = cosmo.comoving_distance(self.z + self.dz)
        dchi = chi2 - chi1 # [Mpc]

        # area of a sphere: A = r^2 * Omega -> A(z) = chi(z)^2 * A[sr]
        A = chi1**2 * self.survey_area_sr.value # [Mpc^2]

        # then the volume is
        dV = dchi * A * (cosmo.h / cu.littleh)**3 # [(Mpc/h)^3]

        # so the target number density is n(z) [(h/Mpc)^3] = n(z) [sr^(-1)] / dV
        self.n_Mpc3 = self.N_z.value / dV
        # print(f"target number density = {self.n_Mpc3.value:.3e}")
    

    def density_table(self):
        """
        If input zbin is `None`, determines the redshift bin based on which distribution has the highest P(z)
        at the input redshift. Otherwise asserts that the input redshift falls within input zbin.

        Returns
        -------
        table : dict of galaxy number densities as a function of redshift

        """

        if self.zbin is not None:
            data = np.load(f'../data/HSC/HSC_dNdz_zbin{zbin}.npy', allow_pickle=True).item()
            assert (self.z >= min(data['z'])) & (self.z <= max(data['z'])), \
                f"input redshift ({self.z:.2f}) outside of zbin {self.zbin} range ({min(data['z']):.2f}-{max(data['z']):.2f})"

            z_thisbin = data['z']
            Pz_thisbin = data['pz']
        
        else:
            # load data for all bins
            z = []
            Pz = []
            for zbin in range(0, 4):
                data = np.load(f'../data/HSC/HSC_dNdz_zbin{zbin}.npy', allow_pickle=True).item()
                z.append(data['z'])
                Pz.append(data['pz'])

            Pz_target = []
            for zbin in range(0, 4):
                if (self.z >= min(z[zbin])) & (self.z <= max(z[zbin])):
                    idx = np.argmin(np.abs(z[zbin] - self.z))
                    Pz_target.append(Pz[zbin][idx])
                else:
                    Pz_target.append(0)
            assert sum(Pz_target) > 0., f"no P(z) found at this redshift ({self.z})"

            # the redshift bin is the one with the highest P(z) at the input redshift:
            self.zbin = np.argmax(Pz_target)

            z_thisbin = z[self.zbin]
            Pz_thisbin = Pz[self.zbin]

        # effective galaxy number densities in each bin: from Section 3.1 of https://arxiv.org/pdf/2211.16516v2
        n = [
            3.77, 5.07, 4.00, 2.12
        ] # per square arcmin

        dz = np.diff(z_thisbin)
        assert np.allclose(dz, dz[0]), "dz are not all equal"

        table = { 'z' : z_thisbin, 'Pz' : Pz_thisbin, 'dz' : dz[0], 'n' : n[self.zbin] }

        return table


"""
DEPRECATED DESI CLASS

class DESI_param:
    '''
    https://arxiv.org/pdf/1611.00036.pdf
    table 2.3 & 2.6
    deltaz~5e-4 p.8
    '''
    def __init__(self, z=0, tracer_name='LRG'):
        
        self.data_dir = '/work2/08811/aew492/frontera/small-scale_cross-corrs/data/DESI'
        self.tracer_names = ['BGS', 'LRG', 'ELG', 'QSO', 'LAF']        
        self.z = z
        self.tracer_name = tracer_name
        self.print_name = 'DESI ' + tracer_name
        if tracer_name == 'LAF':
            self.print_name = 'DESI QSO'
        self.print_name_survey = 'DESI'
        self._calc_params()
    
    def _calc_params(self):
        self._get_area()
        self._get_density()
    
    
    def _get_area(self):
        
        Adeg = 14000
        self.survey_area_deg = Adeg
        
        Asr = Adeg * (u.deg**2).to(u.sr)
        self.survey_area_sr = Asr
        
        self.fsky = Asr / 4 / np.pi
        self.lmin = np.pi / np.sqrt(Asr)
        
        return     
        
    def _get_density(self):
        
        if self.tracer_name == 'LRG':
            table = self.LRG_density_table()
        elif self.tracer_name == 'QSO':
            table = self.QSO_density_table()
        elif self.tracer_name == 'ELG':
            table = self.ELG_density_table()
        elif self.tracer_name == 'BGS':
            table = self.BGS_density_table()
        elif self.tracer_name == 'LAF':
            table = self.LAF_density_table()
            
        zbins_min, zbins_max = table['zbins_min'], table['zbins_max']
        zidx = np.where((self.z >= zbins_min) & (self.z < zbins_max))[0]

        if len(zidx) == 0:
            self.n_zbin_deg2 = 0
            self.n_z_deg2 = 0
            self.n_z_sr = 0
            self.n_Mpc3 = 0
            return
        
        # [dN/d(deg^2)/(dzbin)]
        self.n_zbin_deg2 = table['ns'][zidx[0]]
        self.zbin_min = zbins_min[zidx[0]]
        self.zbin_max = zbins_max[zidx[0]]
        
        # [dN/dz/d(deg^2)]
        self.n_z_deg2 = self.n_zbin_deg2 / (self.zbin_max - self.zbin_min)

        # [dN/dz/dsr]
        self.n_z_sr = self.n_zbin_deg2 * (u.deg**-2).to(u.sr**-1) \
                        / (self.zbin_max - self.zbin_min)
        
        chi1 = cosmo.comoving_distance(self.zbin_max)
        chi2 = cosmo.comoving_distance(self.zbin_min)
        
        dchi = (chi1 - chi2)
        dA = (((chi1 + chi2) / 2) / (u.rad).to(u.deg))**2
        dV = (dchi * dA).value * cosmo.h**3 # [(Mpc/h)**3]
        
        # [dN/d(Mpc/h)^3]
        self.n_Mpc3 = self.n_zbin_deg2 / dV
    

    def ELG_density_table(self):
        '''
        https://arxiv.org/pdf/2208.08513.pdf
        Fig 19 top; blue line
        ns/zbin/deg^2 [deg^-2]
        '''
        data = np.loadtxt(os.path.join(self.data_dir, f'DESI_ELG_South-DECaLS.csv'),
                          delimiter=',')
        z_data = data[:, 0]
        N_data = data[:, 1]/0.05 # dN/dz/ddeg^2
        zbinedges = np.linspace(z_data[0], z_data[-1], 500)
        zbinedges = (zbinedges[1:] + zbinedges[:-1])/2
        zbins_min = zbinedges[:-1]
        zbins_max = zbinedges[1:]
        zbins = (zbins_min + zbins_max) / 2 
        dz = np.diff(zbinedges)

        ns = np.interp(zbins, z_data, N_data)
        ns *= dz

        ELGtable = {'zbins': zbins, 'zbins_min': zbins_min, 'zbins_max': zbins_max,
                   'ns': ns}

        return ELGtable

    def LRG_density_table(self):
        '''
        https://arxiv.org/pdf/2208.08515.pdf
        Fig 1; gray hist
        ns/zbin/deg^2 [deg^-2]
        '''
        data = np.loadtxt(os.path.join(self.data_dir, f'DESI_LRG.csv'),
                          delimiter=',')
        z_data = data[:, 0]
        N_data = data[:, 1]/0.05 # dN/dz/ddeg^2
        zbinedges = np.linspace(z_data[0], z_data[-1], 500)
        zbinedges = (zbinedges[1:] + zbinedges[:-1])/2
        zbins_min = zbinedges[:-1]
        zbins_max = zbinedges[1:]
        zbins = (zbins_min + zbins_max) / 2 
        dz = np.diff(zbinedges)

        ns = np.interp(zbins, z_data, N_data)
        ns *= dz

        ELGtable = {'zbins': zbins, 'zbins_min': zbins_min, 'zbins_max': zbins_max,
                   'ns': ns}

        return ELGtable
    
    def QSO_density_table(self):
        '''
        https://arxiv.org/pdf/1611.00036.pdf
        table 2.3
        
        ns/zbin/deg^2 [deg^-2]
        '''
        zbinedges = np.arange(0.6,2,0.1)
        zbins_min = zbinedges[:-1]
        zbins_max = zbinedges[1:]
        zbins = (zbins_min + zbins_max) / 2        
        dz = np.diff(zbinedges)
        
        ns = np.zeros_like(zbins)
        ns[:] = [47,55,61,67,72,76,80,83,85,87,87,87,86]
        ns *= dz
        
        QSOtable = {'zbins': zbins, 'zbins_min': zbins_min, 'zbins_max': zbins_max,
                   'ns': ns}
        
        return QSOtable
    
    def BGS_density_table(self):
        '''
        https://arxiv.org/pdf/1611.00036.pdf
        table 2.5
        
        ns/zbin/deg^2 [deg^-2]
        '''
        zbinedges = np.arange(0,0.6,0.1)
        zbins_min = zbinedges[:-1]
        zbins_max = zbinedges[1:]
        zbins = (zbins_min + zbins_max) / 2        
        dz = np.diff(zbinedges)
        
        ns = np.array([1165, 3074, 1909,732,120], dtype=float)
        ns *= dz
        
        BGStable = {'zbins': zbins, 'zbins_min': zbins_min, 'zbins_max': zbins_max,
                   'ns': ns}
        
        return BGStable

    def LAF_density_table(self):
        '''
        https://arxiv.org/pdf/1611.00036.pdf
        table 2.7
        NOTE: these numbers are for the background QSO of LAF
        ns/zbin/deg^2 [deg^-2]
        '''
        zbins = np.array([1.96,2.12,2.28,2.43,2.59,2.75,2.91,
                          3.07,3.23,3.39,3.55, 3.70, 3.86, 4.02])
        zbinedges = (zbins[1:] + zbins[:-1])/2
        zbinedges = np.concatenate(([zbinedges[0]-0.16], zbinedges, [zbinedges[-1]+0.16]))
        zbinedges[0] = 1.9
        zbins_min = zbinedges[:-1]
        zbins_max = zbinedges[1:]
        dz = np.diff(zbinedges)
        ns = np.array([82, 69, 53, 43, 37, 31, 26, 21, 16, 13, 9 ,7, 5, 3], dtype=float)
        ns *= dz
        
        LAFtable = {'zbins': zbins, 'zbins_min': zbins_min, 'zbins_max': zbins_max,
                   'ns': ns}
        
        return LAFtable

"""