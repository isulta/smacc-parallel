### PARAMS ###
SIMNAME = 'FP'
DELTATFACTOR = 0.5

Tcmb0 = 0
Neff = 3.04
m_nu = None
Ob0 = None
zarr = [10.04, 9.81, 9.56, 9.36, 9.15, 8.76, 8.57, 8.39, 8.05, 7.89,7.74, 7.45, 7.31, 7.04, 6.91, 6.67, 6.56, 6.34, 6.13, 6.03, 5.84,5.66, 5.48, 5.32, 5.24, 5.09, 4.95, 4.74, 4.61, 4.49, 4.37, 4.26,4.10, 4.00, 3.86, 3.76, 3.63,  3.55, 3.43, 3.31, 3.21, 3.10,3.04,2.94, 2.85, 2.74, 2.65, 2.58, 2.48, 2.41, 2.32, 2.25, 2.17, 2.09,2.02, 1.95, 1.88, 1.80, 1.74, 1.68, 1.61, 1.54, 1.49, 1.43, 1.38,1.32, 1.26, 1.21, 1.15, 1.11, 1.06, 1.01, 0.96, 0.91, 0.86, 0.82,0.78, 0.74, 0.69, 0.66, 0.62, 0.58, 0.54, 0.50, 0.47, 0.43, 0.40,0.36, 0.33, 0.30, 0.27, 0.24, 0.21, 0.18, 0.15, 0.13, 0.10, 0.07,0.05,0.02, 0.00]

cc_input_template = '/cosmo/scratch/rangel/Farpoint/core_properties/*.coreproperties'
cc_output_dir = '/cosmo/scratch/isultan/Farpoint/output'
AFID = 1.1
ZETAFID = 0.1

writeOutputFlag = True
useLocalHost = True
save_cc_prev = True

resume_smacc = False
resume_dir = '/cosmo/scratch/isultan/Farpoint/output/output_19374'
resume_step = 50
# SMACC will start from step `resume_step` if `resume_smacc` is True. SMACC will start from the first step otherwise.
# `resume_dir` must contain resume-files N.*.ccprev.hdf5 for step N which immediately precedes step `resume_step`.
# The resume-files must have been written with the same number of ranks/machine topology as the current run.
### END PARAMS ###

import numpy as np
import os
import glob
from natsort import natsorted
import itk
from astropy.cosmology import FlatLambdaCDM

PARTICLES100MASS = itk.SIMPARAMS[SIMNAME]['PARTICLEMASS']*100.
BOXSIZE = itk.SIMPARAMS[SIMNAME]['Vi']
OMEGA_M = itk.SIMPARAMS[SIMNAME]['OMEGA_M']
OMEGA_L = itk.SIMPARAMS[SIMNAME]['OMEGA_L']
LITTLEH = itk.SIMPARAMS[SIMNAME]['h']
OMEGA_0 = OMEGA_M

cc_data_dir = os.path.dirname(cc_input_template)
cc_input_list = natsorted(glob.glob(cc_input_template))
steps = [int(f.replace(cc_input_template.split('*')[0], '').replace(cc_input_template.split('*')[1], '')) for f in cc_input_list]

cosmoFLCDM = FlatLambdaCDM(H0=LITTLEH*100, Om0=OMEGA_M,  Tcmb0=Tcmb0, Neff=Neff, m_nu=m_nu, Ob0=Ob0)

step2z = {step:z for step, z in zip(steps, zarr)}
step2lookback = {step : cosmoFLCDM.lookback_time(z).value*LITTLEH for step, z in zip(steps, zarr)} #in h^-1 Gyr

def E(z):
    """E(z) = H(z)/H0"""
    return (OMEGA_M*((1+z)**3) + OMEGA_L)**0.5

def Omega(z):
    return OMEGA_0 * (1+z)**3 / E(z)**2

def x(z):
    return Omega(z) - 1

def delta_vir(z):
    return 18*np.pi**2 + 82*x(z) - 39*x(z)**2

def tau(z, A):
    """returns in units h^-1 Gyr"""
    return 1.628/A * ( delta_vir(z)/delta_vir(0) )**(-0.5) * E(z)**(-1)

def m_evolved(m0, M0, step, step_prev, A, zeta, dtFactorFlag=False):
    z = step2z[step]
    delta_t = step2lookback[step_prev] - step2lookback[step]
    if dtFactorFlag:
        delta_t *= DELTATFACTOR

    if zeta == 0:
        return m0 * np.exp( -delta_t/tau(z,A) )
    else:
        return m0 * ( 1 + zeta * (m0/M0)**zeta * delta_t/tau(z,A) )**(-1/zeta)