# HPC Cluster Setup

This guide covers installing and running Underworld3 on HPC clusters. Install scripts are maintained in the [uw3-hpc-baremetal-install-run](https://github.com/jcgraciosa/uw3-hpc-baremetal-install-run) repository.

---

## Architecture

All supported clusters use the same architecture:

```
pixi hpc env  → Python 3.12, sympy, scipy, pint, pydantic, ...  (conda-forge, no MPI)
cluster MPI   → OpenMPI (spack or module)                        (cluster MPI)
source build  → mpi4py, PETSc+AMR+petsc4py, h5py                (linked to cluster MPI)
```

**Why source builds?** Anything linked against MPI must use the same MPI as the cluster scheduler. conda-forge bundles its own MPI (MPICH), which is incompatible with Slurm/PBS. Building from source ensures the correct linkage.

**Why pixi?** Pixi manages the Python environment consistently with local development — same `pixi.toml`, same package versions. The `hpc` environment is pure Python (no MPI packages from conda-forge).

**PETSc build:** `petsc-custom/build-petsc.sh` auto-detects the cluster from hostname, or can be overridden with `UW_CLUSTER=kaiju|gadi`. Cluster-specific differences (HDF5 source, BLAS, cmake, compiler flags) are handled internally.

---

## Kaiju

### Hardware

| Resource | Specification |
|----------|--------------|
| Head node | 1× Intel Xeon Silver 4210R, 40 CPUs @ 2.4 GHz |
| Compute nodes | 8× Intel Xeon Gold 6230R, 104 CPUs @ 2.1 GHz each |
| Shared storage | `/opt/cluster` via NFS |
| Scheduler | Slurm with Munge authentication |
| MPI | Spack `openmpi@4.1.6` |

### Prerequisites

Spack must have OpenMPI available:

```bash
spack find openmpi
# openmpi@4.1.6
```

Pixi must be installed in your user space:

```bash
pixi --version   # check
curl -fsSL https://pixi.sh/install.sh | bash   # install if missing
```

### Installation

Copy `kaiju_install_user.sh` (per-user) or `kaiju_install_shared.sh` (admin) from [uw3-hpc-baremetal-install-run](https://github.com/jcgraciosa/uw3-hpc-baremetal-install-run) to a convenient location, edit the variables at the top, then:

```bash
source kaiju_install_user.sh install
```

| Step | Function | Time |
|------|----------|------|
| Install pixi | `setup_pixi` | ~1 min |
| Clone Underworld3 | `clone_uw3` | ~1 min |
| Install pixi hpc env | `install_pixi_env` | ~3 min |
| Build mpi4py | `install_mpi4py` | ~2 min |
| Build PETSc + AMR tools | `install_petsc` | ~1 hour |
| Build h5py | `install_h5py` | ~2 min |
| Install Underworld3 | `install_uw3` | ~2 min |
| Verify | `verify_install` | ~1 min |

Individual steps can be run after sourcing:

```bash
source kaiju_install_user.sh
install_petsc   # run just one step
```

#### What PETSc builds on Kaiju

- **AMR tools**: mmg, parmmg, pragmatic, eigen, bison
- **Solvers**: mumps, scalapack, slepc
- **Partitioners**: metis, parmetis, ptscotch
- **MPI**: Spack's OpenMPI (`--with-mpi-dir`)
- **HDF5**: downloaded (not in Spack)
- **BLAS/LAPACK**: fblaslapack (no guaranteed system BLAS on Rocky Linux 8)
- **cmake**: downloaded (not in Spack)
- **petsc4py**: built during configure (`--with-petsc4py=1`)

### Activating the Environment

Source the install script at the start of every session or job:

```bash
source kaiju_install_user.sh
```

This loads `spack openmpi@4.1.6`, activates the pixi `hpc` environment via `pixi shell-hook`, and sets `PETSC_DIR`, `PETSC_ARCH`, and `PYTHONPATH`.

> `pixi shell-hook` is used instead of `pixi shell` because it activates the environment in the current shell without spawning a new one — required for Slurm batch jobs.

### Running with Slurm

Use `kaiju_slurm_job.sh` from [uw3-hpc-baremetal-install-run](https://github.com/jcgraciosa/uw3-hpc-baremetal-install-run). Edit the variables at the top, then:

```bash
sbatch kaiju_slurm_job.sh
```

`--mpi=pmix` is **required** on Kaiju (Spack has `pmix@5.0.3`):

```bash
srun --mpi=pmix python3 my_model.py
```

Monitor progress:

```bash
squeue -u $USER
tail -f uw3_<jobid>.out
```

### Shared Installation (Admin)

Deploys to `/opt/cluster/software/underworld3/` so all users access it via Environment Modules:

```bash
source kaiju_install_shared.sh install
module load underworld3/development-12Mar26
```

The shared script adds `fix_permissions()` and `install_modulefile()` on top of the per-user steps. The TCL modulefile hardcodes the Spack OpenMPI and pixi env paths — if Spack is rebuilt (hash changes), update `mpi_root` in `modulefiles/underworld3/development.tcl`.

### Troubleshooting (Kaiju)

#### `import underworld3` fails on compute nodes

Source the install script inside the job script (not the login shell) so all paths propagate to compute nodes. The `kaiju_slurm_job.sh` template does this correctly.

#### PETSc needs rebuilding after Spack module update

PETSc links against Spack's OpenMPI at build time. If `openmpi@4.1.6` is reinstalled:

```bash
source kaiju_install_user.sh
rm -rf ~/uw3-installation/underworld3/petsc-custom/petsc
install_petsc
install_h5py
```

#### h5py replaces source-built mpi4py

`pip install h5py` without `--no-deps` silently replaces the source-built mpi4py with a wheel linked to a different MPI. The install script uses `--no-deps` to prevent this. If mpi4py was accidentally replaced:

```bash
pip install --no-binary :all: --no-cache-dir --force-reinstall "mpi4py>=4,<5"
```

#### PARMMG configure failure

pixi's conda linker requires transitive shared library dependencies to be explicitly linked. `libmmg.so` built with SCOTCH support causes PARMMG's link test to fail. This is fixed in `build-petsc.sh` by building MMG without SCOTCH (`-DUSE_SCOTCH=OFF`).

---

## Gadi

### Hardware

| Resource | Specification |
|----------|--------------|
| System | NCI Gadi (CentOS, Lustre filesystem) |
| Compute | Multiple node types (normal, hugemem, gpuvolta) |
| Shared storage | `/g/data` (project quota), `/scratch` (temporary) |
| Scheduler | PBS Pro |
| MPI | Module `openmpi/4.1.7` |

### Prerequisites

The following Gadi modules must be available:

```bash
module load openmpi/4.1.7 hdf5/1.12.2p gmsh/4.13.1 cmake/3.31.6
```

Pixi must be installed:

```bash
pixi --version   # check
curl -fsSL https://pixi.sh/install.sh | bash   # install if missing
```

> **Inode quota:** Gadi's `/g/data` has strict inode limits. PETSc (which creates many files during build) may need to be built on `/scratch` and symlinked from `/g/data`. The install script handles this if you set `PETSC_DIR` to a `/scratch` path.

### Installation

Copy `gadi_install_user.sh` (per-user) or `gadi_install_shared.sh` (admin) from [uw3-hpc-baremetal-install-run](https://github.com/jcgraciosa/uw3-hpc-baremetal-install-run) to a convenient location, edit the variables at the top, then:

```bash
source gadi_install_shared.sh install
```

| Step | Function | Time |
|------|----------|------|
| Install pixi | `setup_pixi` | ~1 min |
| Clone Underworld3 | `clone_uw3` | ~1 min |
| Install pixi hpc env | `install_pixi_env` | ~3 min |
| Build mpi4py | `install_mpi4py` | ~2 min |
| Build PETSc + AMR tools | `install_petsc` | ~1 hour |
| Build h5py | `install_h5py` | ~2 min |
| Install Underworld3 | `install_uw3` | ~2 min |
| Verify | `verify_install` | ~1 min |

#### What PETSc builds on Gadi

- **AMR tools**: mmg, parmmg, pragmatic, eigen
- **Solvers**: mumps, scalapack, slepc, superlu, superlu_dist, hypre
- **Partitioners**: metis, parmetis, ptscotch
- **MPI**: Gadi's OpenMPI module (`--with-cc/cxx/fc`)
- **HDF5**: Gadi's `hdf5/1.12.2p` module (`--with-hdf5-dir`)
- **BLAS/LAPACK**: fblaslapack (auto-detection fails due to compiler env manipulation)
- **petsc4py**: built during configure (`--with-petsc4py=1`)

### Activating the Environment

Source the install script at the start of every session or job:

```bash
source gadi_install_shared.sh
```

This loads Gadi modules, activates the pixi `hpc` environment via `pixi shell-hook`, and sets `PETSC_DIR`, `PETSC_ARCH`, and `PYTHONPATH`. Gadi's HDF5 lib dir is prepended to `LD_LIBRARY_PATH` to ensure the parallel HDF5 1.12.2p is loaded at runtime (not conda's serial HDF5 1.14).

### Running with PBS

Use `gadi_pbs_job.sh` from [uw3-hpc-baremetal-install-run](https://github.com/jcgraciosa/uw3-hpc-baremetal-install-run). Edit the variables at the top, then:

```bash
qsub gadi_pbs_job.sh
```

Monitor progress:

```bash
qstat -u $USER
tail -f <jobid>.o*
```

### Shared Installation (Admin)

Deploys to `/g/data/m18/software/uw3-pixi/` so all m18 project members can use it:

```bash
source gadi_install_shared.sh install
```

The install script is then copied to the install directory so users can source it directly:

```bash
source /g/data/m18/software/uw3-pixi/gadi_install_shared.sh
```

### Troubleshooting (Gadi)

#### h5py undefined symbol: H5E_BADATOM_g

The pixi `hpc` env ships a serial HDF5 1.14 (transitive conda-forge dependency). If h5py links against it instead of Gadi's parallel HDF5 1.12.2p, this symbol (removed in 1.14) is missing at runtime. The install script fixes this by temporarily hiding conda's HDF5 during the h5py build so meson can only find Gadi's. If you see this error, re-run:

```bash
source gadi_install_shared.sh
install_h5py
```

#### Compiler interference during PETSc build

The pixi `hpc` env ships a full conda toolchain (`x86_64-conda-linux-gnu-*`) that interferes with Gadi's OpenMPI wrappers. `build-petsc.sh` handles this via `setup_gadi_build_env()`, which unsets conda compiler variables and forces the MPI wrappers to use system compilers (`/usr/bin/gcc`).

#### Fortran MPI library not found

Gadi ships compiler-tagged Fortran MPI libraries (`libmpi_usempif08_GNU.so`) rather than the standard untagged names. `build-petsc.sh` creates symlinks in `petsc-custom/mpi-gadi-gnu-libs/` to bridge this.

#### `import underworld3` fails in PBS job

Ensure the install script is sourced inside the job script (not just in the login shell). The `gadi_pbs_job.sh` template does this correctly.

---

## Rebuilding Underworld3 after source changes

```bash
source kaiju_install_user.sh   # or gadi_install_shared.sh
cd <UW3_PATH>
git pull
pip install -e .
```

---

## Related

- [Development Setup](development-setup.md) — local development with pixi
- [Branching Strategy](branching-strategy.md) — git workflow
- [Parallel Computing](../../advanced/parallel-computing.md) — writing parallel-safe UW3 code
