import yaml
import numpy as np
import os
import glob
from natsort import natsorted
import itk
from astropy.cosmology import FlatLambdaCDM

# Load input parameters from `input_params.yaml`
with open('input_params.yaml', 'r') as f:
    INPUTPARAMS = yaml.safe_load(f)

SIMNAME = INPUTPARAMS['SIMNAME']
DELTATFACTOR = INPUTPARAMS['DELTATFACTOR']
Tcmb0 = INPUTPARAMS['Tcmb0']
Neff = INPUTPARAMS['Neff']
m_nu = INPUTPARAMS['m_nu']
Ob0 = INPUTPARAMS['Ob0']
zarr = INPUTPARAMS['zarr']
cc_input_template = INPUTPARAMS['cc_input_template']
cc_output_dir = INPUTPARAMS['cc_output_dir']
AFID = INPUTPARAMS['AFID']
ZETAFID = INPUTPARAMS['ZETAFID']
writeOutputFlag = INPUTPARAMS['writeOutputFlag']
useLocalHost = INPUTPARAMS['useLocalHost']
save_cc_prev = INPUTPARAMS['save_cc_prev']
resume_smacc = INPUTPARAMS['resume_smacc']
resume_dir = INPUTPARAMS['resume_dir']
resume_step = INPUTPARAMS['resume_step']

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