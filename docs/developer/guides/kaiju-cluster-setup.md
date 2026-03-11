# Kaiju Cluster Setup

This guide covers installing and running Underworld3 on the **Kaiju** cluster — a Rocky Linux 8.10 HPC system using Spack for module management and Slurm for job scheduling.

Python packages are managed by **pixi** (the same tool used for local development). MPI-dependent packages — `mpi4py`, PETSc+AMR tools, `petsc4py`, and `h5py` — are built from source against Spack's OpenMPI to ensure compatibility with Slurm's parallel interconnect.

---

## Hardware Overview

| Resource | Specification |
|----------|--------------|
| Head node | 1× Intel Xeon Silver 4210R, 40 CPUs @ 2.4 GHz |
| Compute nodes | 8× Intel Xeon Gold 6230R, 104 CPUs @ 2.1 GHz each |
| Shared storage | `/opt/cluster` via NFS (cluster-wide) |
| Scheduler | Slurm with Munge authentication |

---

## Why pixi + spack?

Pixi manages the Python environment consistently with the developer's local machine (same `pixi.toml`, same package versions). Spack provides the cluster's OpenMPI, which is what Slurm uses for inter-node communication.

The key constraint is that **anything linked against MPI must use the same MPI as Slurm**. This means `mpi4py`, `h5py`, PETSc, and `petsc4py` are built from source against Spack's OpenMPI — not from conda-forge (which bundles MPICH).

```
pixi kaiju env  → Python 3.12, sympy, scipy, pint, pydantic, ...  (conda-forge, no MPI)
spack           → openmpi@4.1.6                                    (cluster MPI)
source build    → mpi4py, PETSc+AMR+petsc4py, h5py                (linked to spack MPI)
```

---

## Prerequisites

Spack must have OpenMPI available:

```bash
spack find openmpi
# openmpi@4.1.6
```

Pixi must be installed in your user space (no root needed):

```bash
# Check if already installed
pixi --version

# Install if missing
curl -fsSL https://pixi.sh/install.sh | bash
```

---

## Installation

Use the install script at `kaiju-admin-notes/uw3_install_kaiju_amr.sh`.

### Step 1: Edit configuration

Open the script and set the variables at the top:

```bash
SPACK_MPI_VERSION="openmpi@4.1.6"          # Spack MPI module to load
INSTALL_PATH="${HOME}/uw3-installation"     # Root directory for everything
UW3_BRANCH="development"                    # UW3 git branch
```

### Step 2: Run the full install

```bash
source uw3_install_kaiju_amr.sh install
```

This runs the following steps in order:

| Step | Function | Time |
|------|----------|------|
| Install pixi | `setup_pixi` | ~1 min |
| Clone Underworld3 | `clone_uw3` | ~1 min |
| Install pixi kaiju env | `install_pixi_env` | ~3 min |
| Build mpi4py from source | `install_mpi4py` | ~2 min |
| Build PETSc + AMR tools | `install_petsc` | ~1 hour |
| Build MPI-enabled h5py | `install_h5py` | ~2 min |
| Install Underworld3 | `install_uw3` | ~2 min |
| Verify | `verify_install` | ~1 min |

You can also run individual steps after sourcing:

```bash
source uw3_install_kaiju_amr.sh
install_petsc       # run just one step
```

### What PETSc builds

PETSc is compiled from source (`petsc-custom/build-petsc-kaiju.sh`) with:

- **AMR tools**: mmg, parmmg, pragmatic, eigen, bison
- **Solvers**: mumps, scalapack, slepc
- **Partitioners**: metis, parmetis, ptscotch
- **MPI**: Spack's OpenMPI (`--with-mpi-dir`)
- **HDF5**: downloaded and built with MPI support
- **BLAS/LAPACK**: fblaslapack (Rocky Linux 8 has no guaranteed system BLAS)
- **cmake**: downloaded (not in Spack)
- **petsc4py**: built during configure (`--with-petsc4py=1`)

---

## Activating the Environment

In every new session (interactive or job), source the install script:

```bash
source ~/install_scripts/uw3_install_kaiju_amr.sh
```

This:
1. Loads `spack openmpi@4.1.6`
2. Activates the pixi `kaiju` environment via `pixi shell-hook`
3. Sets `PETSC_DIR`, `PETSC_ARCH`, and `PYTHONPATH` for petsc4py
4. Sets `PMIX_MCA_psec=native` and `OMPI_MCA_btl_tcp_if_include=eno1`

{note}
`pixi shell-hook` is used instead of `pixi shell` because it activates the environment in the current shell without spawning a new one. This is required for Slurm batch jobs.
{/note}

---

## Running with Slurm

Use `kaiju-admin-notes/uw3_slurm_job.sh` as your job script template.

### Submitting a job

```bash
sbatch uw3_slurm_job.sh
```

Monitor progress:

```bash
squeue -u $USER
tail -f uw3_<jobid>.out
```

### The `srun` invocation

`--mpi=pmix` is **required** on Kaiju (Spack has `pmix@5.0.3`):

```bash
srun --mpi=pmix python3 my_model.py
```

### Scaling examples

```bash
# 1 node, 30 ranks
sbatch --nodes=1 --ntasks-per-node=30 uw3_slurm_job.sh

# 4 nodes, 120 ranks
sbatch --nodes=4 --ntasks-per-node=30 uw3_slurm_job.sh
```

---

## Troubleshooting

### `import underworld3` fails on compute nodes

Sourcing the install script in the job script (not the login shell) ensures all paths propagate to compute nodes. The `uw3_slurm_job.sh` template does this correctly.

### h5py HDF5 version mismatch

h5py must be built against the same HDF5 that PETSc built. If you see HDF5 errors, rebuild:

```bash
source uw3_install_kaiju_amr.sh
install_h5py
```

### PETSc needs rebuilding after Spack module update

PETSc links against Spack's OpenMPI at build time. If `openmpi@4.1.6` is reinstalled or updated, rebuild PETSc:

```bash
source uw3_install_kaiju_amr.sh
rm -rf ~/uw3-installation/underworld3/petsc-custom/petsc
install_petsc
install_h5py
```

### Checking what's installed

```bash
source uw3_install_kaiju_amr.sh
verify_install
```

---

## Rebuilding Underworld3 after source changes

After pulling new UW3 code:

```bash
source uw3_install_kaiju_amr.sh
cd ~/uw3-installation/underworld3
git pull
pip install -e .
```

---

## Related

- [Development Setup](development-setup.md) — local development with pixi
- [Branching Strategy](branching-strategy.md) — git workflow
- [Parallel Computing](../../advanced/parallel-computing.md) — writing parallel-safe UW3 code
