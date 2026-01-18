---
title: "Next Steps"
# keywords: []
authors:
- name: Underworld Team

exports:
- format: pdf
- template: arxiv_nips
---


### Underworld Documentation and Examples

In addition to the notebooks in this brief set of examples, there are a number of sources of information on using `Underworld3` that you can access:

  - [The Underworld Website / Blog](https://www.underworldcode.org)

  - [The API documentation](https://underworldcode.github.io/underworld3/main_api/underworld3/index.html)
    (all the modules and functions and their full sets of arguments) is automatically generated from the source code and uses the same rich markdown content as the notebook help text.

  - [The API documentation (development branch)](https://underworldcode.github.io/underworld3/development_api/underworld3/index.html)

  - The [`underworld3` GitHub repository](https://github.com/underworldcode/underworld3) is the most active development community for the code.


### Benchmarks

The [Underworld3 Benchmarks Repository](https://github.com/underworld-community/UW3-benchmarks) is a useful place to find community benchmarks coded in `underworld3` along with accuracy and convergence analysis. This is an open repository where you can make a pull request with new benchmark submissions once you get the hang of things.

### The Underworld Community

The [Underworld Community](https://github.com/underworld-community) organisation on Github is a collection of contributed repositories from the underworld user community for all versions of the code and now includes scripts for underworld3.

### Parallel Execution

`Underworld3` is inherently parallel and designed for high performance computing. The symbolic layer is a relatively thin veneer that sits on top of the `PETSc` machinery. A script developed in Jupyter should transfer to an HPC environment with no changes (except that inherently serial operations such as visualization are best left to post-processing).

We recommend using `jupytext` which provides seamless two-way conversion between `.ipynb` and annotated Python scripts. The Python form doesn't store cell outputs, which is advantageous for version control. Almost all of our notebook examples use this format.

```bash
mpirun -np 1024 python3 Modelling_Script.py -uw_resolution 96
```

Or using the `./uw` wrapper for local parallel runs:

```bash
./uw mpirun -np 4 python3 Modelling_Script.py
```

The main difference between notebook development and HPC is interactivity—particularly sending parameters at launch time. We use PETSc's command line parsing so notebooks can ingest runtime parameters when run as scripts.

#### Parallel scaling / performance

Running geodynamic models on a single CPU/processor (i.e. serial) is time-consuming and limits us to low resolution. Underworld is build from the ground-up as a parallel computing solution which means we can easily run large models on high performance computing clusters (HPC); that is, sub-divide the problem into many smaller chunks and use multiple processors to solve each one, taking care to combine and synchronise the answers from each processor to obtain the correct solution to the original problem.

Parallel computation can reduce time we need to wait for the our results to be computed but it does happen at the expense of some overhead The overhead does depend on the nature of the computer we are using but typically we need to think about:

 - **Code complexity**: any time we manage computations across different processors, we have additional coding to reassemble the calculations correctly and we need to think about many special cases. For example, integrating a quantity of the surface of a mesh: many processes contribute, some do not, the results have to be computed independently then combined.

 - **Additional memory is often required**: to manage copies of information that lives on / near boundaries, to store the topology of the decomposed domain and to help navigate the passing of information between processes.

 - **The time taken to synchronise results** and the work required to keep track of who is doing what, when they are done, and in making sure everyone waits for everyone else. There is a time-cost in actually sending information as part of a synchronisation and a computational cost in ensuring that work is distributed efficiently.

To determine the efficiency of parallel computation, we introduce the *strong scaling test* which measures the time taken to solve a problem in parallel compared to the same problem solved in serial. In strong scaling tests, the size of the problem is kept constant, while the number of processors is increased. The reduction in run-time due to the addition of more processors is commonly expressed in terms of the speed-up:

$$
\textrm{speed up} = \frac{t(N_{ref})}{t(N)}
$$

where $t(N_{ref})$ is the run-time for a reference number of processors, $N_{ref}$, and $t(N)$ is the run-time when $N$ processors are used. In the ideal case, $N$ additional processors should contribute all of its resources in solving the problem and reduce the compute time by a factor of $N$ relative to the reference run time. For example, using $2 N_{ref}$ processors will ideally halve the run-time resulting to a speed-up = 2.

```{figure} media/UW3-StrongScalingSolvers.png
:name: fig-strong-scaling

Strong parallel-scaling tests run on Australia's peak computing system, [GADI, at the National Computational Infrastructure](https://nci.org.au/our-systems/hpc-systems?ref=underworldcode.org). This is a typical High Performance Computing facility with large numbers of dedicated, identical CPUs and fast communication links.
```


### Advanced capabilities

Digging a bit deeper into `underworld3`, there are many capabilities that require a clear understanding of the concepts that underlie the implementation. The following examples are not *plug-and-play*, but they do only require python coding using the `underworld3` API and no detailed knowledge of `petsc4py` or `PETSc`. [Get in touch with us](https://github.com/underworldcode/underworld3/issues) if you want to try this sort of thing but can't figure it out for yourself.

#### Deforming meshes

In [Example 8](tutorials/8-Particle_Swarms.ipynb), we made small variations to the mesh to conform to basal topography. We did not remesh, so we had to be careful to apply a smooth, analytic displacement to every node. For more general free-surface models, we need to calculate a smooth function using the computed boundary motions (e.g, solving a poisson equation with known boundary displacements as boundary conditions). We need to step through time and it is common to stabilize the surface motions through a surface integral term that accounts for the interface displacement during the timestep. The example below shows an `underworld3` forward model with internal loads timestepped close to isostatic equilibrium.

```{figure} media/RelaxingMesh.png
:width: 50%

Stokes flow driven by buoyancy in an annulus defined by two embedded surfaces within an enveloping disk mesh. The surfaces deform in response to the flow. The embedding medium has a very low viscosity but still acts to damp rotational modes. The outer boundary of the disk can be set to a far-field gravitational potential for whole-Earth relaxation models.
```

In a more general case, we need to account to horizontal motions. This is more complicated because the horizontal velocities can be large even when vertical equilibrium is achieved. So we need to solve for the advected component of vertical motion in addition to the local component. Hooray for symbolic algebra !

#### Weak / penalty boundary conditions

[Example 8](tutorials/8-Particle_Swarms.ipynb) introduced the idea of penalty-based boundary conditions where the constraint is weakly enforced by providing a large penalty for violation of the condition. This is very flexible as the penalizing conditions can be adjusted during the run, including changing which part of the boundary is subject to constraints based on the solution or a coupled problem. The channel flow model shown below has a boundary condition that depends on a externally sourced model for ponded material at the base that is derived from a simple topography filling algorithm.

```{raw} html
<center>
<iframe src="../pyvista/ChannelFlow.html" width="600" height="300">
</iframe>
</center>
```

*Live Image: Stokes flow in a channel with multiple obstructions. Flow is driven from the inlet (a velocity boundary condition). The geometry was constructed with `gmsh`. This is an example for education which demonstrates the emergence of a large-scale pressure gradient as a result of the presence of the obstructions, and also the dispersion of tracers through the complicated flow geometry*

The penalty approach does allow the solution to deviate from the exact value of the boundary condition, in a similar way to the iterative solvers delivering a solution to within a specified tolerance. There are some cases, for example, enforcing plate motions at the surface, where there are uncertainties in the applied boundary conditions and that these uncertainties may vary in space and time.

#### Mesh Adaptation

It is also possible to use the PETSc mesh adaption capabilities, to refine the resolution of a model where it is most needed and coarsen elsewhere. Dynamically-adapting meshes are possible but the interface is very low level at present, particularly in parallel.

```{raw} html
<center>
<iframe src="../pyvista/AdaptedSphere.html" width="600" height="500">
</iframe>
</center>
```

*Live Image: Static mesh adaptation to the slope of a field. The driving buoyancy term is a plume-like upwelling and the slope of this field is shown in colour (red high, blue low). Don't forget to zoom in !*

```python

    # t is the driving "temperature". We form an isotropic refinement metric from its slope

    refinement_fn = 1.0 + sympy.sqrt(
          t.diff(x) ** 2
        + t.diff(y) ** 2
        + t.diff(z) ** 2
    )

    icoord, meshA = adaptivity.mesh_adapt_meshVar(mesh0, refinement_fn, Metric, redistribute=True)
```
