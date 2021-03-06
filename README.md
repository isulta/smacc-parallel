# smacc-parallel
smacc-parallel is a parallel MPI-based implementation of SMACC (Subhalo Mass-loss Analysis using Core Catalogs).
The code takes GenericIO Core Catalog (`.coreproperties`) files as input and evolves the mass of satellite cores at each simulation snapshot according to a subhalo mass loss model.
The Core Catalog with an added evolved core mass column is saved to disk in HDF5 format for each snapshot.

See [the paper](https://arxiv.org/abs/2012.09262) for detailed information on SMACC and the mass loss model.

## Usage
### Setup
To get started, clone the repository with 

    git clone https://github.com/isulta/smacc-parallel 
Ensure that the required [Dependencies](#Dependencies) are installed. 

Next, set the [Input parameters](#Input-parameters) in `input_params.yaml`.

### Generate Core Catalog with modeled satellite core mass
    mpiexec -n $NRANKS python cc_generate_parallel.py
Note that the above command to start the parallel MPI processes may be different depending on the system.
Sample scripts to run smacc-parallel on various HPC systems are provided in `job_submission_scripts/`.

For each snapshot, smacc-parallel will save the output HDF5 Core Catalog as a `.corepropertiesextend.hdf5` file with
- a limited set of columns from the input GenericIO Core Catalog (see [Data Columns](#Data-Columns) for fine-tuning which columns are written)
- `m_evolved_{A}_{zeta}`: modeled satellite core mass column (for central cores, this column has value 0)

## Input parameters
All parameters for smacc-parallel are defined in `input_params.yaml`:

| **Simulation** |  |
|-|-|
| `SIMNAME` | Set to `LJ` for Last Journey, `SV` for Last Journey-SV, `HM` for Last Journey-HM, `AQ` for AlphaQ, or `FP` for Farpoint. (See [Other Simulations](#Other-Simulations) if using a simulation not listed here.) |
| **Mass Loss Model** |  |
| `AFID` | A mass model parameter |
| `ZETAFID` | ζ mass model parameter |
| `useLocalHost` | If `True`, smacc-parallel will use the immediate parent (sub)halo mass for M in the mass model. If `False`, smacc-parallel will use the host halo mass for M. |
| `DELTATFACTOR` | Fraction of the time between the snapshot at which infall is detected for a core and the preceding snapshot, over which to compute the initial mass evolution of the new satellite. |
| **File Input/Output** |  |
| `cc_input_template` | File path of GenericIO Core Catalog (`.coreproperties`) header files. Use an asterisk in place of the snapshot number. smacc-parallel will automatically create a list of all snapshots. |
| `cc_output_dir` | Directory to save output Core Catalog files with `m_evolved_{A}_{zeta}` column (`.corepropertiesextend.hdf5`) |
| `writeOutputFlag` | If `True`, smacc-parallel will save output Core Catalog files at each snapshot to `cc_output_dir`. |
| `save_cc_prev` | If `True`, smacc-parallel will also save "resume-files" (`.ccprev.hdf5`) at each snapshot to `cc_output_dir`. This allows smacc-parallel to resume from a later snapshot instead of having to restart if a run gets interrupted (See *Resume SMACC*). |
| **Data Columns** |  |
| `extra_columns` | List of extra columns to read/write, in addition to the *default columns* required by the mass model (see [Data Columns](#Data-Columns)). If parameter is set to `[]`, only the default columns will read/written. |
| `extra_columns_dtypes` | List of the data types of the columns given in `extra_columns` (each element must be `i 32`, `i 64`, or `f 32`). Parameter must be set to `[]` if `extra_columns` is `[]`. |
| `steps_to_read_extra_columns` | List of snapshots at which to read/write the columns given in `extra_columns`, in addition to the default columns. For snapshots not in `steps_to_read_extra_columns`, only the default columns will be read/written. If this parameter is set to `all`, the columns given in `extra_columns` will be read/written for **every snapshot**. Parameter must be set to `[]` if `extra_columns` is `[]`. |
| **Resume SMACC** |  |
| `resume_smacc` | If `True`, smacc-parallel will start from snapshot `resume_step`. If `False`, smacc-parallel will start from the first snapshot. |
| `resume_dir` | Location of resume-files `N.*.ccprev.hdf5` for snapshot N which immediately precedes snapshot `resume_step`. The resume-files must have been written with the same number of ranks/machine topology as the current run. (Define if `resume_smacc` is set to `True`.) |
| `resume_step` | Snapshot at which to resume smacc-parallel. (Define if `resume_smacc` is set to `True`.) |

## Dependencies
Python 3 is required (smacc-parallel has been tested for Python 3.8.5).
Running in a [conda](https://conda.io/projects/conda/en/latest/index.html) environment is recommended.

The following packages are also required:
- [itk](https://github.com/isulta/itk)
- [pygio](https://xgitlab.cels.anl.gov/hacc/genericio/-/tree/master/new_python)
- HDF5 and H5py (both must be built with [parallel support enabled](https://docs.h5py.org/en/stable/mpi.html#building-against-parallel-hdf5))
- Astropy
- Numpy
- Matplotlib
- mpi4py
- pyyaml
- natsort
- SciPy
- numba

## Data Columns
By default, smacc-parallel read/writes a limited set of columns of the input GenericIO Core Catalog to minimize runtime.
The following *default columns* are required for the mass model and are always read:

    core_tag, tree_node_index, infall_tree_node_mass, central, host_core

The default columns are written for every snapshot. 

Additional columns can be added to the output HDF5 Core Catalog by adding the column names to `extra_columns`, and adding the data types of the new columns to `extra_columns_dtypes`. 
The columns in `extra_columns` will be written at only the snapshots that are given in `steps_to_read_extra_columns` (see [Input parameters](#Input-parameters)).

## Other Simulations
To use smacc-parallel with a simulation not listed above, assign a key and add the key and simulation/cosmological parameters to `simulationParams.yaml` in `itk`.
smacc-parallel will then import the parameters if `SIMNAME` is set to the new key.
Note that `itk` only supports simulations with a cosmology close to the current ΛCDM best-fit cosmology; `itk` assumes Ω<sub>r,0</sub> = Ω<sub>k,0</sub> = 0.

By default, smacc-parallel uses the step number-to-redshift formula given below, which is valid for the HACC simulations listed above (Last Journey, LJ-SV, LJ-HM, AlphaQ, and Farpoint).
If using a simulation with a different step-to-redshift formula, the function `redshift` in `itk.py` must be updated.

![step number-to-redshift formula](steptozformula.png)