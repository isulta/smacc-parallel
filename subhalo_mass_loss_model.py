import yaml
import numpy as np
import os
import glob
from natsort import natsorted
import itk
from scipy.integrate import quad

# Load input parameters from `input_params.yaml`
with open( os.path.join(os.path.dirname(__file__), 'input_params.yaml'), 'r' ) as f:
    INPUTPARAMS = yaml.safe_load(f)

SIMNAME = INPUTPARAMS['SIMNAME']
DELTATFACTOR = INPUTPARAMS['DELTATFACTOR']
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

cc_data_dir = os.path.dirname(cc_input_template)
cc_input_list = natsorted(glob.glob(cc_input_template))
steps = [int(f.replace(cc_input_template.split('*')[0], '').replace(cc_input_template.split('*')[1], '')) for f in cc_input_list]

def lookback_time(z):
    """
    Returns in units h^-1 Gyr.
    See https://ned.ipac.caltech.edu/level5/Hogg/Hogg10.html.
    """
    return itk.THUBBLE * quad( lambda z_: 1/(E(z_)*(1+z_)), 0, z )[0]

def E(z):
    """E(z) = H(z)/H0"""
    return ( OMEGA_M * (1+z)**3 + OMEGA_L )**0.5

def Omega_m(z):
    return OMEGA_M * (1+z)**3 / E(z)**2

def x(z):
    return Omega_m(z) - 1

def delta_vir(z):
    xz = x(z)
    return 18*np.pi**2 + 82*xz - 39*xz**2

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

step2z = {step:z for step, z in zip(steps, zarr)}
step2lookback = {step : lookback_time(z) for step, z in zip(steps, zarr)} #in h^-1 Gyr