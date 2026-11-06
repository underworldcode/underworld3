"""Geometry-neutral Stokes observer registration and KSP monitor ownership.

Callbacks are collective and borrow PETSc objects: never mutate or destroy the
operator, RHS, solver settings or iterates. Retained copies belong to the caller.
"""
from __future__ import annotations
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, ExitStack
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import underworld3 as uw
from underworld3.utilities._stokes_state import StokesSolveState

class StokesObserver:
    """Optional read-only hooks shared by cylindrical and spherical Stokes solves.

    Both backends emit solve_started, iteration, solve_finished and finalized.
    Background/physical operators, assembly phases and PC timings are optional
    backend-specific observations; they do not prescribe a common discretization.
    """
    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        return False

    def background(self, matrix, velocity_size, pressure_size):
        pass


    def physical(self, matrix, velocity_size, pressure_size, pressure_mode, rotations, rotation_weights, walls, components):
        pass


    def phase(self, name: str, snapshot: Mapping[str, object]) -> None:
        pass


    def pc_started(self, application: int) -> None:
        pass


    def pc_finished(self, application: int) -> None:
        pass


    def solve_started(self, ksp: PETSc.KSP, state: StokesSolveState) -> None:
        pass


    def iteration(self, ksp: PETSc.KSP, iteration: int, norm: float, state: StokesSolveState) -> None:
        pass


    def solve_finished(self, ksp: PETSc.KSP, solution: PETSc.Vec, state: StokesSolveState) -> None:
        pass


    def finalized(self, result):
        pass


def solve_with_observer(ksp: PETSc.KSP, solve_rhs: PETSc.Vec, solution: PETSc.Vec, state: StokesSolveState, observer: StokesObserver) -> None:
    """Run the unchanged KSP, removing only this observer's Python monitor.

    PETSc ASCII/native monitors and pre-existing Python monitors remain installed.
    The petsc4py monitor context is a list of (callback,args,kwargs) registrations;
    removing our exact registration preserves the other entries and C monitors.
    """
    observer.solve_started(ksp, state)
    def monitor(solver, iteration, norm):
        observer.iteration(solver, iteration, norm, state)
    ksp.setMonitor(monitor)
    registration = ksp.getMonitor()[-1]
    try:
        answer = ksp.solve(solve_rhs, solution)
        observer.solve_finished(ksp, solution, state)
        return answer
    finally:
        entries = ksp.getMonitor()
        entries[:] = [entry for entry in entries if entry is not registration]


@contextmanager
def observe_stokes(stokes, observer, *, _attribute='_stokes_observer'):
    """Register one scoped observer, preserving historical spherical registration."""
    if any(getattr(stokes, name, None) is not None for name in ('_stokes_observer', '_spherical_gap_observer')):
        raise RuntimeError('A Stokes observer is already registered')
    existed = hasattr(stokes, _attribute)
    previous = getattr(stokes, _attribute, None)
    setattr(stokes, _attribute, observer)
    try:
        with observer:
            yield observer
    finally:
        if existed:
            setattr(stokes, _attribute, previous)
        else:
            delattr(stokes, _attribute)
