---
title: "Development Environment Setup"
---

# Setting up an Underworld3 development environment

Underworld3 builds Cython extensions against a specific PETSc, so the build is
not as forgiving as a pure-Python package. The constraints below are the ones
that cost real time when they are broken.

## Everyday commands

```bash
./uw build                    # rebuild after ANY source change
./uw test                     # run the test suite
pixi run -e default python    # a Python in the environment
```

## Rebuild after every source change

Underworld3 is *installed* into the pixi environment, not imported from `src/`.
Editing a file under `src/underworld3/` changes nothing until you rebuild:

```bash
./uw build
```

Check what you are actually running with `uw.__file__` — it must point into
`.pixi/envs/<env>/lib/python3.12/site-packages/underworld3/`, never `src/`.

`./uw build` passes `--no-cache-dir`, because the version is always `0.0.0` and
pip will otherwise reuse a stale wheel. If you still suspect stale code, clear
the intermediate build tree as well:

```bash
rm -rf build/lib.* build/bdist.*
./uw build
```

A `.pyx` change that appears not to take effect is almost always this: see also
[the JIT cache notes](../subsystems/jit-cache.md).

## Never use an editable install

`pip install -e .` is prohibited. There are no exceptions, and the reason is
that the damage outlives the install:

- the `.pth` files it writes **contaminate every pixi environment** sharing the
  source directory, and worktrees share environments by symlink;
- they **persist after uninstall**, silently redirecting imports back to `src/`
  even once a proper `./uw build` has run;
- `.so` files left in the source tree get loaded by an environment expecting a
  different PETSc arch, which surfaces as a `dlopen` symbol error rather than
  as anything that names the real cause.

Always `./uw build`, which runs a non-editable `pip install .`. Where `./uw` is
unavailable:

```bash
pixi run -e <env> pip install . --no-build-isolation --no-cache-dir
```

### Recovering from editable-install contamination

```bash
find .pixi/envs -name "__editable__*underworld*" -delete   # stale .pth, all envs
find src/underworld3 -name "*.so" -delete                  # .so belong in site-packages
rm -rf build/
./uw build
```

## PETSc is not relocatable

```
/Users/lmoresi/+Underworld/underworld-pixi-2/petsc/
```

**Do not move this directory.** PETSc hardcodes paths at configure time, so
moving it breaks petsc4py and every pixi task that depends on it, and the only
repair is a full rebuild of roughly an hour. Worktrees share this one PETSc by
symlink for exactly that reason; everything else in a worktree environment is
its own.

## Worktrees have their own environments

Each worktree carries its own pixi environment — its own site-packages and its
own compiled extensions — with only PETSc shared. `./uw build` installs the
source of whichever worktree you run it from, into that worktree's environment.
So: enter the worktree first, then build, then run. Building from the main
checkout does nothing for a worktree.

See [Branching and Release Strategy](branching-strategy.md) for the worktree
lifecycle and the branch policy.
