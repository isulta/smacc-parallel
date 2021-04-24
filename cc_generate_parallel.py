from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
ranks = comm.Get_size()

import numpy as np
import subhalo_mass_loss_model as SHMLM
from itk import intersect1d_parallel_sorted, many_to_one_allranks_numba, h5_write_dict_parallel, h5_write_dict, h5_read_dict, intersect1d_numba
import os
import time
from datetime import datetime, timedelta
import pygio

A, zeta = SHMLM.AFID, SHMLM.ZETAFID
steps = SHMLM.steps

vars_cc_all = SHMLM.vars_cc_all
vars_cc_min = SHMLM.vars_cc_min
dtypes_cc_all = SHMLM.dtypes_cc_all

def printr(s, root=0):
    if rank == root:
        print(f'[{datetime.now()}] rank {root}: {s}', flush=True)
    comm.Barrier()

def vars_cc(step):
    if step in SHMLM.steps_to_read_extra_columns:
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
    dtypes_cc_all[m_evolved_col(A, zeta)] = dtypes_cc_all['infall_tree_node_mass']

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
            # Set m_evolved of all satellites that have core_tag match on prev step to next_m_evolved of prev step.
            printr('cc_prev m_evolved matching for all ranks...'); start=time.time()
            # cc_prev m_evolved matching on local rank first
            idxsrt_sat = np.argsort(cc['core_tag'][satellites_mask])
            idxsrt_prev = np.argsort(cc_prev['core_tag'])

            idx1_local, idx2_local = intersect1d_numba(cc['core_tag'][satellites_mask][idxsrt_sat], cc_prev['core_tag'][idxsrt_prev])
            cc[m_evolved_col(A, zeta)][ np.flatnonzero(satellites_mask)[idxsrt_sat][idx1_local] ] = cc_prev[m_evolved_col(A, zeta, next=True)][idxsrt_prev][idx2_local]

            unmatched_satellites_idx = np.ones(np.sum(satellites_mask), dtype=np.bool)
            unmatched_satellites_idx[idx1_local] = False
            unmatched_satellites_idx = np.flatnonzero(satellites_mask)[idxsrt_sat][unmatched_satellites_idx]

            unmatched_prev_idx = np.ones(len(cc_prev['core_tag']), dtype=np.bool)
            unmatched_prev_idx[idx2_local] = False
            unmatched_prev_idx = idxsrt_prev[unmatched_prev_idx]

            for root in range(ranks):
                idx1, data = intersect1d_parallel_sorted(comm, rank, root,
                                                            ( cc['core_tag'][unmatched_satellites_idx] if rank==root else None ), 
                                                            ( cc_prev['core_tag'][unmatched_prev_idx] if rank!=root else np.array([], dtype=dtypes_cc_all['core_tag']) ), 
                                                            dtypes_cc_all['core_tag'], 
                                                            ( cc_prev[m_evolved_col(A, zeta, next=True)][unmatched_prev_idx] if rank!=root else np.array([], dtype=dtypes_cc_all['infall_tree_node_mass']) ), 
                                                            dtypes_cc_all['infall_tree_node_mass'])
                if rank == root:
                    cc[m_evolved_col(A, zeta)][ unmatched_satellites_idx[idx1] ] = data
                printr(f'Found {len(idx1) if rank==root else None} satellite core_tag matches in cc_prev.', root)
                comm.Barrier()
            printr(f'Finished cc_prev m_evolved matching for all ranks in {time.time()-start} seconds.')

            # Find host halo mass M for satellites.
            printr('Finding M for all ranks...'); start=time.time()
            M = many_to_one_allranks_numba(comm, rank, root, cc['tree_node_index'][satellites_mask], cc['tree_node_index'][centrals_mask], dtypes_cc_all['tree_node_index'], cc['infall_tree_node_mass'][centrals_mask], dtypes_cc_all['infall_tree_node_mass'])
            printr(f'Finished finding M for all ranks in {time.time()-start} seconds.')
            
            # Find parent halo mass Mlocal for satellites.
            if (step != steps[-1]) and useLocalHost:
                printr('Finding Mlocal for all ranks...'); start=time.time()
                Mlocal = many_to_one_allranks_numba(comm, rank, root, cc['host_core'][satellites_mask], cc['core_tag'], dtypes_cc_all['core_tag'], cc[m_evolved_col(A, zeta)], dtypes_cc_all['infall_tree_node_mass'])
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
            h5_write_dict_parallel(comm, rank, cc, sorted(list(cc.keys())), dtypes_cc_all, fn)
            printr(f'Finished writing cc to disk in {time.time()-start} seconds.')

       # Compute m_evolved of satellites according to SHMLModel for NEXT time step and save as cc_prev['next_m_evolved'] in memory.
        if step != steps[-1]:
            cc_prev = { 'core_tag':cc['core_tag'][satellites_mask].copy() }
            cc_prev[m_evolved_col(A, zeta, next=True)] = np.zeros(np.sum(satellites_mask), dtype=dtypes_cc_all['infall_tree_node_mass'])

            if numSatellites != 0: # If there are satellites (not applicable for first step)
                printr('SHMLM (next step)...'); start=time.time()
                m = cc[m_evolved_col(A, zeta)][satellites_mask]
                if useLocalHost:
                    M_A_zeta = (Mlocal==0)*M + (Mlocal!=0)*Mlocal
                    cc_prev[m_evolved_col(A, zeta, next=True)] = SHMLM.m_evolved(m0=m, M0=M_A_zeta, step=steps[steps.index(step)+1], step_prev=step, A=A, zeta=zeta)
                else:
                    cc_prev[m_evolved_col(A, zeta, next=True)] = SHMLM.m_evolved(m0=m, M0=M, step=steps[steps.index(step)+1], step_prev=step, A=A, zeta=zeta)
                printr(f'Finished SHMLM (next step) in {time.time()-start} seconds.')
            if save_cc_prev:
                printr('Writing ccprev hdf5...'); start=time.time()
                h5_write_dict( os.path.join(SHMLM.cc_output_dir, f'{step}.{rank}.ccprev.hdf5'), cc_prev, 'ccprev' )
                printr(f'Finished writing ccprev hdf5 in {time.time()-start} seconds.')
        
        printr(f'Finished step {step} in {time.time()-start_step} seconds.\n')

if __name__ == '__main__':
    printr(f'Beginning SMACC.'); start_smacc = time.time()
    create_core_catalog_mevolved(SHMLM.writeOutputFlag, SHMLM.useLocalHost, SHMLM.save_cc_prev, SHMLM.resume_smacc)
    printr(f'Finished SMACC in {timedelta(seconds=time.time()-start_smacc)}')
