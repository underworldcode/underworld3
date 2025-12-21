"""
Underworld3 Diagnostics Module

Provides diagnostic tools to detect common configuration issues,
especially PETSc library mismatches that cause cryptic runtime errors.

Usage:
    import underworld3 as uw
    uw.doctor()  # Run all diagnostics

    # Or from command line:
    ./uw doctor
"""

import sys
import os
import platform
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class Status(Enum):
    """Diagnostic status levels."""
    OK = "ok"
    WARNING = "warning"
    ERROR = "error"
    INFO = "info"


@dataclass
class DiagnosticResult:
    """Result of a single diagnostic check."""
    name: str
    status: Status
    message: str
    details: Optional[str] = None
    fix_hint: Optional[str] = None


def _get_extension_library_paths(extension_path: str) -> List[str]:
    """
    Get the libraries an extension is linked against.

    Uses otool on macOS, ldd on Linux.
    """
    import subprocess

    libraries = []
    system = platform.system()

    try:
        if system == "Darwin":
            result = subprocess.run(
                ["otool", "-L", extension_path],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                for line in result.stdout.split('\n')[1:]:  # Skip first line (filename)
                    line = line.strip()
                    if line and '(' in line:
                        lib_path = line.split('(')[0].strip()
                        libraries.append(lib_path)
        elif system == "Linux":
            result = subprocess.run(
                ["ldd", extension_path],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if '=>' in line:
                        parts = line.split('=>')
                        if len(parts) > 1:
                            lib_path = parts[1].strip().split()[0]
                            if lib_path and lib_path != 'not':
                                libraries.append(lib_path)
    except Exception:
        pass

    return libraries


def _resolve_rpath(lib_ref: str, extension_path: str) -> Optional[str]:
    """
    Resolve @rpath references to actual library paths.

    On macOS, @rpath is resolved using DYLD_LIBRARY_PATH and embedded rpaths.
    """
    if not lib_ref.startswith('@rpath/'):
        return lib_ref

    lib_name = lib_ref.replace('@rpath/', '')

    # Check DYLD_LIBRARY_PATH (macOS) or LD_LIBRARY_PATH (Linux)
    system = platform.system()
    path_var = "DYLD_LIBRARY_PATH" if system == "Darwin" else "LD_LIBRARY_PATH"

    for path in os.environ.get(path_var, '').split(':'):
        if path:
            candidate = os.path.join(path, lib_name)
            if os.path.exists(candidate):
                return os.path.realpath(candidate)

    # Check common conda/pixi paths
    if 'CONDA_PREFIX' in os.environ:
        candidate = os.path.join(os.environ['CONDA_PREFIX'], 'lib', lib_name)
        if os.path.exists(candidate):
            return os.path.realpath(candidate)

    return None


def check_petsc_library_match() -> DiagnosticResult:
    """
    Check if the PETSc library used at compile time matches runtime.

    This is the most common source of cryptic errors like:
    "DMInterpolationSetUp_UW failed with error 98"
    """
    try:
        import petsc4py
        from petsc4py import PETSc

        # Get petsc4py's view of where PETSc is
        petsc_config = petsc4py.get_config()
        runtime_petsc_dir = petsc_config.get('PETSC_DIR', '')

        # Find our extension module
        import underworld3
        uw_path = Path(underworld3.__file__).parent
        extension_candidates = [
            uw_path / 'function' / '_dminterp_wrapper.cpython-312-darwin.so',
            uw_path / 'function' / '_function.cpython-312-darwin.so',
            uw_path / 'cython' / 'petsc_discretisation.cpython-312-darwin.so',
        ]

        # Also check Linux naming
        extension_candidates.extend([
            uw_path / 'function' / '_dminterp_wrapper.cpython-312-x86_64-linux-gnu.so',
            uw_path / 'function' / '_function.cpython-312-x86_64-linux-gnu.so',
        ])

        extension_path = None
        for candidate in extension_candidates:
            if candidate.exists():
                extension_path = str(candidate)
                break

        if not extension_path:
            return DiagnosticResult(
                name="PETSc Library Match",
                status=Status.WARNING,
                message="Could not find extension modules to check",
                details="Extension files not found in expected locations"
            )

        # Get libraries the extension links to
        linked_libs = _get_extension_library_paths(extension_path)

        # Find PETSc library reference
        petsc_lib = None
        for lib in linked_libs:
            if 'petsc' in lib.lower():
                petsc_lib = lib
                break

        if not petsc_lib:
            return DiagnosticResult(
                name="PETSc Library Match",
                status=Status.WARNING,
                message="Could not find PETSc library reference in extension",
                details=f"Checked: {extension_path}"
            )

        # Resolve @rpath if needed
        resolved_lib = _resolve_rpath(petsc_lib, extension_path)

        # Check for mismatch
        issues = []

        # Check 1: Is it using an absolute path to a different location?
        if not petsc_lib.startswith('@rpath'):
            # Absolute path - check if it matches runtime
            if runtime_petsc_dir and runtime_petsc_dir not in petsc_lib:
                if 'petsc-custom' in petsc_lib or 'petsc-4-uw' in petsc_lib:
                    issues.append(
                        f"Extension linked to CUSTOM PETSc: {petsc_lib}\n"
                        f"But runtime uses conda-forge PETSc: {runtime_petsc_dir}"
                    )

        # Check 2: If using @rpath, verify it resolves correctly
        if petsc_lib.startswith('@rpath') and resolved_lib:
            # Good - using relocatable path
            pass
        elif petsc_lib.startswith('@rpath') and not resolved_lib:
            issues.append(
                f"Extension uses @rpath but library not found: {petsc_lib}\n"
                f"Expected in: {runtime_petsc_dir}/lib/"
            )

        if issues:
            return DiagnosticResult(
                name="PETSc Library Match",
                status=Status.ERROR,
                message="PETSc library MISMATCH detected!",
                details="\n".join(issues),
                fix_hint=(
                    "Rebuild underworld3 with a clean cache:\n"
                    "  1. pixi run pip cache purge\n"
                    "  2. ./uw clean\n"
                    "  3. ./uw build"
                )
            )

        return DiagnosticResult(
            name="PETSc Library Match",
            status=Status.OK,
            message="PETSc libraries match",
            details=f"Extension: {petsc_lib}\nRuntime: {runtime_petsc_dir}"
        )

    except Exception as e:
        return DiagnosticResult(
            name="PETSc Library Match",
            status=Status.WARNING,
            message=f"Could not check PETSc libraries: {e}",
        )


def check_petsc_version() -> DiagnosticResult:
    """Check PETSc version compatibility."""
    try:
        from petsc4py import PETSc
        version = PETSc.Sys.getVersion()
        version_str = f"{version[0]}.{version[1]}.{version[2]}"

        # Check minimum version
        if version[0] < 3 or (version[0] == 3 and version[1] < 21):
            return DiagnosticResult(
                name="PETSc Version",
                status=Status.WARNING,
                message=f"PETSc {version_str} may be too old",
                details="Underworld3 requires PETSc >= 3.21",
                fix_hint="Update PETSc: pixi install or conda install petsc>=3.21"
            )

        return DiagnosticResult(
            name="PETSc Version",
            status=Status.OK,
            message=f"PETSc {version_str}",
        )
    except Exception as e:
        return DiagnosticResult(
            name="PETSc Version",
            status=Status.ERROR,
            message=f"Could not check PETSc version: {e}",
        )


def check_petsc_version_match() -> DiagnosticResult:
    """
    Check if the PETSc version at compile time matches runtime version.

    This detects the common issue where underworld3 was built against
    PETSc 3.X but is running with PETSc 3.Y, causing cryptic errors like:
    "Must call SNESSetFunction() or SNESSetDM() before SNESComputeFunction()"

    The check works by:
    1. Finding the libpetsc.X.Y.dylib that the extension links to
    2. Comparing X.Y to the runtime petsc4py version
    """
    try:
        import re
        from petsc4py import PETSc
        import underworld3

        # Get runtime PETSc version
        runtime_version = PETSc.Sys.getVersion()
        runtime_major_minor = (runtime_version[0], runtime_version[1])
        runtime_str = f"{runtime_version[0]}.{runtime_version[1]}"

        # Find an extension module that links to PETSc
        uw_path = Path(underworld3.__file__).parent

        # Prefer _function or petsc_discretisation as they definitely link to PETSc
        extension_path = None
        priority_patterns = [
            'function/_function*.so',
            'cython/petsc_discretisation*.so',
            'cython/generic_solvers*.so',
        ]

        for pattern in priority_patterns:
            for candidate in uw_path.glob(pattern):
                extension_path = str(candidate)
                break
            if extension_path:
                break

        if not extension_path:
            return DiagnosticResult(
                name="PETSc Version Match",
                status=Status.INFO,
                message="Could not find extension to check",
            )

        # Get linked libraries
        linked_libs = _get_extension_library_paths(extension_path)

        # Find the PETSc library and extract version from filename
        # e.g., @rpath/libpetsc.3.24.dylib, libpetsc.so.3.24.2
        compile_version = None
        petsc_lib_name = None

        for lib in linked_libs:
            lib_lower = lib.lower()
            if 'petsc' in lib_lower and 'petsc4py' not in lib_lower:
                petsc_lib_name = lib
                # Try to extract version from library name
                # Patterns: libpetsc.3.24.dylib, libpetsc.so.3.24.2
                match = re.search(r'libpetsc\.(\d+)\.(\d+)', lib)
                if match:
                    compile_version = (int(match.group(1)), int(match.group(2)))
                break

        if not petsc_lib_name:
            return DiagnosticResult(
                name="PETSc Version Match",
                status=Status.INFO,
                message="Could not find PETSc library reference",
            )

        if not compile_version:
            return DiagnosticResult(
                name="PETSc Version Match",
                status=Status.INFO,
                message="Could not determine compile-time PETSc version",
                details=f"Library: {petsc_lib_name}"
            )

        compile_str = f"{compile_version[0]}.{compile_version[1]}"

        # Compare versions
        if compile_version != runtime_major_minor:
            return DiagnosticResult(
                name="PETSc Version Match",
                status=Status.ERROR,
                message=f"VERSION MISMATCH: compiled={compile_str}, runtime={runtime_str}",
                details=(
                    f"Extension linked against: {petsc_lib_name}\n"
                    f"Runtime petsc4py version: {runtime_str}\n\n"
                    f"This causes cryptic errors like:\n"
                    f'  "Must call SNESSetFunction() or SNESSetDM()..."'
                ),
                fix_hint=(
                    "Rebuild underworld3 against the current PETSc:\n"
                    "  pip uninstall underworld3\n"
                    "  rm -rf build/\n"
                    "  pip install . --no-build-isolation --no-deps\n\n"
                    "Or use ./uw:\n"
                    "  ./uw clean && ./uw build"
                )
            )

        return DiagnosticResult(
            name="PETSc Version Match",
            status=Status.OK,
            message=f"Compile/runtime versions match ({compile_str})",
            details=f"Library: {petsc_lib_name}"
        )

    except Exception as e:
        return DiagnosticResult(
            name="PETSc Version Match",
            status=Status.WARNING,
            message=f"Could not check version match: {e}",
        )


def check_extension_modules() -> DiagnosticResult:
    """Check that all required extension modules load correctly."""
    extensions = [
        ('underworld3.function._function', 'Core function evaluation'),
        ('underworld3.function._dminterp_wrapper', 'DM interpolation'),
        ('underworld3.cython.petsc_discretisation', 'PETSc discretisation'),
        ('underworld3.cython.petsc_maths', 'PETSc math operations'),
        ('underworld3.cython.generic_solvers', 'SNES solvers'),
        ('underworld3.ckdtree', 'KD-tree operations'),
    ]

    failed = []
    loaded = []

    for module_name, description in extensions:
        try:
            __import__(module_name)
            loaded.append(module_name.split('.')[-1])
        except ImportError as e:
            failed.append(f"{description}: {e}")

    if failed:
        return DiagnosticResult(
            name="Extension Modules",
            status=Status.ERROR,
            message=f"{len(failed)} extension(s) failed to load",
            details="\n".join(failed),
            fix_hint="Rebuild underworld3: ./uw clean && ./uw build"
        )

    return DiagnosticResult(
        name="Extension Modules",
        status=Status.OK,
        message=f"All {len(loaded)} extensions loaded",
        details=", ".join(loaded)
    )


def check_mpi_configuration() -> DiagnosticResult:
    """Check MPI is configured correctly."""
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

        return DiagnosticResult(
            name="MPI Configuration",
            status=Status.OK,
            message=f"MPI working (rank {rank} of {size})",
            details=f"Implementation: {MPI.Get_library_version().split()[0]}"
        )
    except Exception as e:
        return DiagnosticResult(
            name="MPI Configuration",
            status=Status.ERROR,
            message=f"MPI error: {e}",
        )


def check_basic_evaluation() -> DiagnosticResult:
    """
    Test basic function evaluation - this catches PETSc ABI mismatches.

    This is the "smoke test" that would have caught the error 98 issue.
    """
    try:
        import numpy as np
        import underworld3 as uw

        # Create minimal test
        uw.reset_default_model()
        mesh = uw.meshing.StructuredQuadBox(
            elementRes=(2, 2),
            minCoords=(0, 0),
            maxCoords=(1, 1)
        )

        P = uw.discretisation.MeshVariable("P_test", mesh, 1, degree=1)
        P.data[:, 0] = 1.0

        # This is the critical test - evaluate at a point
        test_coords = np.array([[0.5, 0.5]], dtype=np.float64)
        result = uw.function.evaluate(P.sym, test_coords)

        if abs(result[0, 0, 0] - 1.0) < 0.01:
            return DiagnosticResult(
                name="Function Evaluation",
                status=Status.OK,
                message="Basic evaluation working",
                details="DMInterpolation test passed"
            )
        else:
            return DiagnosticResult(
                name="Function Evaluation",
                status=Status.WARNING,
                message=f"Unexpected result: {result[0, 0, 0]} (expected 1.0)",
            )

    except RuntimeError as e:
        error_msg = str(e)
        if "DMInterpolationSetUp_UW failed" in error_msg:
            return DiagnosticResult(
                name="Function Evaluation",
                status=Status.ERROR,
                message="DMInterpolation FAILED - likely PETSc library mismatch!",
                details=error_msg,
                fix_hint=(
                    "This error typically means the extension was compiled against\n"
                    "a different PETSc library than the one loaded at runtime.\n\n"
                    "Fix:\n"
                    "  pixi run pip cache purge\n"
                    "  ./uw clean\n"
                    "  ./uw build"
                )
            )
        return DiagnosticResult(
            name="Function Evaluation",
            status=Status.ERROR,
            message=f"Evaluation failed: {e}",
        )
    except Exception as e:
        return DiagnosticResult(
            name="Function Evaluation",
            status=Status.ERROR,
            message=f"Evaluation error: {e}",
        )


def check_environment_variables() -> DiagnosticResult:
    """Check relevant environment variables."""
    vars_to_check = [
        ('PETSC_DIR', 'PETSc installation directory'),
        ('PETSC_ARCH', 'PETSc architecture'),
        ('CONDA_PREFIX', 'Conda/Pixi environment'),
    ]

    info = []
    warnings = []

    for var, description in vars_to_check:
        value = os.environ.get(var, '')
        if value:
            info.append(f"{var}={value}")
            # Check for potential issues
            if var == 'PETSC_DIR' and 'petsc-custom' in value:
                conda = os.environ.get('CONDA_PREFIX', '')
                if conda and 'petsc-custom' not in conda:
                    warnings.append(
                        f"PETSC_DIR points to custom PETSc but CONDA_PREFIX suggests conda-forge"
                    )

    if warnings:
        return DiagnosticResult(
            name="Environment Variables",
            status=Status.WARNING,
            message="Potential environment conflict",
            details="\n".join(info + ["", "Warnings:"] + warnings)
        )

    return DiagnosticResult(
        name="Environment Variables",
        status=Status.INFO,
        message=f"{len(info)} relevant variables set",
        details="\n".join(info) if info else "No PETSc environment variables set (using defaults)"
    )


def run_diagnostics(verbose: bool = True, quick: bool = False) -> List[DiagnosticResult]:
    """
    Run all diagnostic checks.

    Parameters
    ----------
    verbose : bool
        Print results as they're generated
    quick : bool
        Skip slow tests (like basic evaluation)

    Returns
    -------
    List[DiagnosticResult]
        Results from all checks
    """
    checks = [
        ("Checking PETSc version...", check_petsc_version),
        ("Checking PETSc version match...", check_petsc_version_match),
        ("Checking extension modules...", check_extension_modules),
        ("Checking MPI configuration...", check_mpi_configuration),
        ("Checking environment variables...", check_environment_variables),
        ("Checking PETSc library match...", check_petsc_library_match),
    ]

    if not quick:
        checks.append(("Testing function evaluation...", check_basic_evaluation))

    results = []

    # Status symbols
    symbols = {
        Status.OK: "\033[92m✓\033[0m",      # Green checkmark
        Status.WARNING: "\033[93m⚠\033[0m",  # Yellow warning
        Status.ERROR: "\033[91m✗\033[0m",    # Red X
        Status.INFO: "\033[94mℹ\033[0m",     # Blue info
    }

    # Check if terminal supports colors
    if not sys.stdout.isatty():
        symbols = {
            Status.OK: "[OK]",
            Status.WARNING: "[WARN]",
            Status.ERROR: "[ERROR]",
            Status.INFO: "[INFO]",
        }

    if verbose:
        print("\n" + "="*60)
        print("  Underworld3 Diagnostics")
        print("="*60 + "\n")

    for description, check_func in checks:
        if verbose:
            print(f"  {description}", end=" ", flush=True)

        result = check_func()
        results.append(result)

        if verbose:
            print(f"{symbols[result.status]} {result.message}")
            if result.details and (result.status in (Status.ERROR, Status.WARNING)):
                for line in result.details.split('\n'):
                    print(f"      {line}")
            if result.fix_hint and result.status == Status.ERROR:
                print(f"\n    Fix:")
                for line in result.fix_hint.split('\n'):
                    print(f"      {line}")
                print()

    if verbose:
        # Summary
        errors = sum(1 for r in results if r.status == Status.ERROR)
        warnings = sum(1 for r in results if r.status == Status.WARNING)

        print("\n" + "-"*60)
        if errors > 0:
            print(f"  {symbols[Status.ERROR]} {errors} error(s) found - action required!")
        elif warnings > 0:
            print(f"  {symbols[Status.WARNING]} {warnings} warning(s) - review recommended")
        else:
            print(f"  {symbols[Status.OK]} All checks passed!")
        print("-"*60 + "\n")

    return results


def doctor(verbose: bool = True, quick: bool = False) -> bool:
    """
    Run diagnostics and return True if all checks pass.

    This is the main entry point for users.

    Parameters
    ----------
    verbose : bool
        Print detailed output (default True)
    quick : bool
        Skip slow tests like function evaluation (default False)

    Returns
    -------
    bool
        True if no errors found, False otherwise

    Examples
    --------
    >>> import underworld3 as uw
    >>> uw.doctor()  # Run full diagnostics
    >>> uw.doctor(quick=True)  # Quick check without evaluation test
    """
    results = run_diagnostics(verbose=verbose, quick=quick)
    return not any(r.status == Status.ERROR for r in results)


# Convenience function for quick health check
def health_check() -> bool:
    """Quick silent health check. Returns True if OK."""
    results = run_diagnostics(verbose=False, quick=True)
    return not any(r.status == Status.ERROR for r in results)
