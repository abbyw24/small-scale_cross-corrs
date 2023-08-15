"""
Copy from Yun-Ting Chen.
Class to get source number density as a function of redshift for SDSS and DESI spectroscopic surveys.
"""

import numpy as np
import astropy.units as u
from astropy.cosmology import Planck15 as cosmo

class eBOSS_param:
    '''
    https://ui.adsabs.harvard.edu/abs/2016AJ....151...44D/abstract
    deltaz = sigma_z/(1+z) ~ 1e-3 (p.9)
    ELG:
    https://ui.adsabs.harvard.edu/abs/2017MNRAS.471.3955R/abstract
    '''
    def __init__(self, z=0, tracer_name='LRG',
                 field_name=None, ns_field_name=None, get_bias=True):
        
        self.tracer_names = ['CMASS', 'LRG', 'ELG', 'QSO']        
        self.z = z
        self.tracer_name = tracer_name
        self.print_name = 'eBOSS ' + tracer_name
        self.print_name_survey = '(e)BOSS'
        if tracer_name == 'CMASS':
            self.print_name = self.print_name[1:]
        self._calc_params(get_bias=get_bias)

    
    def _calc_params(self, get_bias=True):
        self._get_area()
        self._get_density()
        if get_bias:
            self.bias = self.tracer_bias()
    
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
    https://arxiv.org/pdf/1611.00036.pdf
    table 2.3 & 2.6
    deltaz~5e-4 p.8
    '''
    def __init__(self, z=0, tracer_name='LRG', get_bias=True):
        
    
        self.tracer_names = ['BGS', 'LRG', 'ELG', 'QSO', 'LAF']        
        self.z = z
        self.tracer_name = tracer_name
        self.print_name = 'DESI ' + tracer_name
        if tracer_name == 'LAF':
            self.print_name = 'DESI QSO'
        self.print_name_survey = 'DESI'
        self._calc_params(get_bias=get_bias)
    
    def _calc_params(self, get_bias=True):
        self._get_area()
        self._get_density()
        if get_bias:
            self.bias = self.tracer_bias()
    
    
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
                
            
    def LRG_density_table(self):
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
        ns[:6] = [832, 986, 662, 272, 51, 17]
        ns *= dz
        
        LRGtable = {'zbins': zbins, 'zbins_min': zbins_min, 'zbins_max': zbins_max,
                   'ns': ns}
        
        return LRGtable
    
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
     
    def ELG_density_table(self):
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
        ns[:-2] = [309,2269,1923,2094,1441,1353,1337,523,466,329,126]
        ns *= dz
        
        ELGtable = {'zbins': zbins, 'zbins_min': zbins_min, 'zbins_max': zbins_max,
                   'ns': ns}
        
        return ELGtable
    
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