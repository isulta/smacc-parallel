from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
ranks = comm.Get_size()

import numpy as np
import subhalo_mass_loss_model as SHMLM
from itk import intersect1d_parallel, many_to_one_parallel, h5_write_dict_parallel, h5_write_dict, h5_read_dict
import os
import time
import pygio

A, zeta = SHMLM.AFID, SHMLM.ZETAFID
steps = SHMLM.steps

vars_cc_all = [
    'fof_halo_tag',
    'infall_fof_halo_mass',
    'core_tag',
    'tree_node_index',
    'infall_tree_node_mass',
    'central',
    'host_core'
]
vars_cc_min = [
    'core_tag',
    'tree_node_index',
    'infall_tree_node_mass',
    'central',
    'host_core'
]
dtypes_cc_all = {
    'fof_halo_tag': np.int64,
    'infall_fof_halo_mass': np.float32,
    'core_tag': np.int64,
    'tree_node_index': np.int64,
    'infall_tree_node_mass': np.float32,
    'central': np.int32,
    'host_core': np.int64
}

def printr(s, root=0):
    if rank == root:
        print(f'rank {root}: {s}', flush=True)
    comm.Barrier()

def vars_cc(step):
    if step==499 or step==247:
        return vars_cc_all
    else:
        return vars_cc_min

def m_evolved_col(A, zeta, next=False):
    if next:
        return 'next_m_evolved_{}_{}'.format(A, zeta)
    else:
        return 'm_evolved_{}_{}'.format(A, zeta)

def create_core_catalog_mevolved(writeOutputFlag, useLocalHost, save_cc_prev, resume_smacc):
    '''
    Appends mevolved to core catalog and saves output in HDF5 format.
    '''
    if writeOutputFlag:
        printr(f'Reading data from {SHMLM.cc_data_dir} and writing output to {SHMLM.cc_output_dir}.')
    
    cc = {}
    cc_prev = {}
    M = None
    Mlocal = None

    for step, fn_cc_input in zip(steps, SHMLM.cc_input_list):
        # Resume SMACC at step `SHMLM.resume_step` if `resume_smacc`==True
        if resume_smacc:
            if step < SHMLM.resume_step:
                printr(f'Skipping step {step} (SMACC will resume at step {SHMLM.resume_step}).')
                continue
            elif step == SHMLM.resume_step:
                # Read resume-files from previous step
                fn_prev = os.path.join(SHMLM.resume_dir, f'{steps[steps.index(step)-1]}.{rank}.ccprev.hdf5')
                printr(f'Reading ccprev {fn_prev}...'); start=time.time()
                cc_prev = h5_read_dict(fn_prev, 'ccprev')
                printr(f'Finished reading ccprev hdf5 in {time.time()-start} seconds.')

        # Read in cc for step
        printr(f'Beginning step {step} (step {steps.index(step)+1} of {len(steps)}). Reading GIO core catalog...'); start_step = time.time()
        cc = pygio.read_genericio(fn_cc_input, vars_cc(step))
        printr(f'Finished reading GIO core catalog in {time.time()-start_step} seconds.')

        satellites_mask = cc['central'] == 0
        centrals_mask = cc['central'] == 1
        numSatellites = comm.allreduce( np.sum(satellites_mask) ) # total number of satellites across ALL ranks

        # Verify there are no satellites at first step
        if step == steps[0]:
            assert numSatellites == 0, 'Satellites found at first step.'

        # Add column for m_evolved and initialize to 0
        cc[m_evolved_col(A, zeta)] = np.zeros_like(cc['infall_tree_node_mass'])

        # If there are satellites (not applicable for first step)
        if numSatellites != 0:
            printr('cc_prev m_evolved matching and M for all ranks...'); start=time.time()
            for root in range(ranks):
                # Set m_evolved of all satellites that have core_tag match on prev step to next_m_evolved of prev step.
                idx1, data = intersect1d_parallel(comm, rank, root, (cc['core_tag'][satellites_mask] if rank==root else None), cc_prev['core_tag'], dtypes_cc_all['core_tag'], cc_prev[m_evolved_col(A, zeta, next=True)], dtypes_cc_all['infall_tree_node_mass'])
                if rank == root:
                    cc[m_evolved_col(A, zeta)][ np.flatnonzero(satellites_mask)[idx1] ] = data
                printr(f'Found {len(idx1) if rank==root else None} satellite core_tag matches in cc_prev.', root)
                comm.Barrier()
                
                # Find host halo mass M for satellites.
                Data = many_to_one_parallel(comm, rank, root, (cc['tree_node_index'][satellites_mask] if rank==root else None), cc['tree_node_index'][centrals_mask], dtypes_cc_all['tree_node_index'], cc['infall_tree_node_mass'][centrals_mask], dtypes_cc_all['infall_tree_node_mass'])
                if rank == root:
                    M = Data.copy()
                comm.Barrier()
            printr(f'Finished cc_prev m_evolved matching and M for all ranks in {time.time()-start} seconds.')
            
            # Find parent halo mass Mlocal for satellites.
            if (step != steps[-1]) and useLocalHost:
                printr('Finding Mlocal for all ranks...'); start=time.time()
                for root in range(ranks):
                    Data = many_to_one_parallel(comm, rank, root, (cc['host_core'][satellites_mask] if rank==root else None), cc['core_tag'], dtypes_cc_all['core_tag'], cc[m_evolved_col(A, zeta)], dtypes_cc_all['infall_tree_node_mass'])
                    if rank == root:
                        Mlocal = Data.copy()
                    comm.Barrier()
                printr(f'Finished finding Mlocal for all ranks in {time.time()-start} seconds.')
            
            # Initialize mass of new satellites
            printr('SHMLM (new satellites)...'); start=time.time()
            initMask = cc[m_evolved_col(A, zeta)][satellites_mask] == 0
            minfall = cc['infall_tree_node_mass'][satellites_mask][initMask]
            cc[m_evolved_col(A, zeta)][ np.flatnonzero(satellites_mask)[initMask] ] = SHMLM.m_evolved(m0=minfall, M0=M[initMask], step=step, step_prev=steps[steps.index(step)-1], A=A, zeta=zeta, dtFactorFlag=True)
            printr(f'Finished SHMLM (new satellites) in {time.time()-start} seconds.')
        if writeOutputFlag:
            # Write cc to disk
            fn = os.path.join(SHMLM.cc_output_dir, f'{step}.corepropertiesextend.hdf5')
            printr(f'Writing cc to {fn}...'); start=time.time()
            h5_write_dict_parallel(comm, rank, cc, vars_cc(step), dtypes_cc_all, fn)
            printr(f'Finished writing cc to disk in {time.time()-start} seconds.')

       # Compute m_evolved of satellites according to SHMLModel for NEXT time step and save as cc_prev['next_m_evolved'] in memory.
        if step != steps[-1]:
            cc_prev = { 'core_tag':cc['core_tag'].copy() }
            cc_prev[m_evolved_col(A, zeta, next=True)] = np.zeros_like(cc['infall_tree_node_mass'])

            if numSatellites != 0: # If there are satellites (not applicable for first step)
                printr('SHMLM (next step)...'); start=time.time()
                m = cc[m_evolved_col(A, zeta)][satellites_mask]
                if useLocalHost:
                    M_A_zeta = (Mlocal==0)*M + (Mlocal!=0)*Mlocal
                    cc_prev[m_evolved_col(A, zeta, next=True)][satellites_mask] = SHMLM.m_evolved(m0=m, M0=M_A_zeta, step=steps[steps.index(step)+1], step_prev=step, A=A, zeta=zeta)
                else:
                    cc_prev[m_evolved_col(A, zeta, next=True)][satellites_mask] = SHMLM.m_evolved(m0=m, M0=M, step=steps[steps.index(step)+1], step_prev=step, A=A, zeta=zeta)
                printr(f'Finished SHMLM (next step) in {time.time()-start} seconds.')
            if save_cc_prev:
                printr('Writing ccprev hdf5...'); start=time.time()
                h5_write_dict( os.path.join(SHMLM.cc_output_dir, f'{step}.{rank}.ccprev.hdf5'), cc_prev, 'ccprev' )
                printr(f'Finished writing ccprev hdf5 in {time.time()-start} seconds.')
        
        printr(f'Finished step {step} in {time.time()-start_step} seconds.\n')

if __name__ == '__main__':
    create_core_catalog_mevolved(SHMLM.writeOutputFlag, SHMLM.useLocalHost, SHMLM.save_cc_prev, SHMLM.resume_smacc)
