# Bounding and diagnosing an MPI collective hang

A rank blocked inside a collective is the worst kind of test failure: it produces
no output, returns no status, and takes the whole job with it. In CI it costs the
entire job budget and then gets cancelled from outside, so you learn nothing at
all. That happened for 76 minutes on a single run ([#675]).

`scripts/mpi_supervisor.py` bounds that, and says what went wrong.

## Why not the mechanisms we already had

Everything we tried first asks the stuck process to cooperate, which is precisely
what it has stopped being able to do.

| mechanism | why it fails |
|---|---|
| `threading.Timer` and a report | needs the GIL. Measured at np=4 against a 4 s block in `comm.allreduce`: a re-arming timer at 0.5 s fired **zero** times on the blocked ranks. |
| `faulthandler.dump_traceback_later` | a C thread walking other threads' live frames. It can wedge or crash the process it is diagnosing ([#661]). |
| `MPI_Abort` from a watchdog thread | needs `THREAD_MULTIPLE` and a schedulable thread; not safe from inside a collective. |
| `mpirun --timeout` | Open MPI spells it `--timeout`, MPICH spells it `-timeout`, and the semantics differ. Kills the job without telling you anything. |

The supervisor asks nothing of the job. It is a parent process holding a clock
and a signal, so it behaves identically under Open MPI and MPICH, on macOS and
Linux, because it is POSIX process management and knows nothing about MPI.

## Usage

```bash
scripts/mpi_supervisor.py --silence 300 -- mpirun -n 4 python -m pytest --with-mpi tests/parallel/
```

The exit status is the job's own, or `124` if the supervisor had to kill it —
the same number `timeout(1)` uses, so it reads the way people expect.

`scripts/test.sh` already routes its parallel batches through it. Override the
budget with `PARALLEL_SILENCE` if a batch legitimately goes quiet for longer.

## It watches for silence, not for elapsed time

A batch that is still printing is alive. Judging on total runtime would mean
inventing a wall-clock budget for a test matrix nobody has measured, and that
number is wrong the first time somebody adds a slow test. Silence is both the
stronger signal and the one that needs no calibration — the failure in [#675]
was 76 minutes of *nothing*, not 76 minutes of slow progress.

## What it reports

On silence it signals every rank to dump its Python stack, samples twice, and
compares:

```
=== MPI SUPERVISOR: no output for 300 s ===
    no rank moved in 20 s: this is stuck, not slow
    rank(s) 0 are somewhere the other 3 are not — that is where to look first
      rank 0 is in _rank_zero_misses_the_collective()
      the other 3 are in main()
    rank 0: 0% CPU, idle — blocked rather than polling
    rank 1: 100% CPU, spinning — a busy-wait, which is what an MPI progress engine does
```

Three separate pieces of evidence, and each answers a different question:

- **Two samples, not one.** Identical stacks 20 s apart mean stuck; one stack
  only ever means slow.
- **The odd rank out.** The ranks that agree are waiting for the one that does
  not. At np=2 there is no majority to appeal to, and the report says so rather
  than nominating a rank on a one-all split.
- **CPU per rank.** A rank inside a busy-waiting MPI progress engine burns a
  whole core; one blocked on a lock sleeps. Note the direction of that signal:
  in the example above the *spinning* ranks are the innocent ones, waiting in
  the barrier, and the idle rank is the guilty one.

Stacks are written to `--artifacts` so CI can keep them; a diagnosis that dies
with the job is no better than the silence it replaced.

## How the dump is armed

The supervisor writes a `usercustomize` module onto the job's `PYTHONPATH`, and
that registers a `SIGUSR1` handler through `faulthandler.register`. The job's own
code knows nothing about it.

Registering a *signal handler* is the important difference from
`dump_traceback_later`. The handler runs on the thread that receives the signal,
only when the supervisor asks, and never walks live frames from a concurrent
thread — so arming it costs a healthy run nothing and it cannot reproduce
[#661].

## Killing the tree, and checking

Killing the launcher is not killing the job. Descendants are enumerated and
killed individually before the process group is signalled, because the group
call is the one that fails: `mpirun` puts its children in a group of their own,
and `killpg` on macOS has been seen to refuse with `EPERM`.

The supervisor then re-checks and reports anything that survived. This is not
ceremony — an unsupervised kill leaves orphaned `mpirun`, `hydra_pmi_proxy` and
pytest children reparented to init and spinning at 100% CPU, which is how a
"finished" run keeps eating cores for hours ([#639]).

## The control

`tests/test_0063_mpi_hang_supervisor.py` plants a hang whose answer is known:
rank 0 sits in a function no other rank can be inside while the others wait in a
barrier. The supervisor must end the job, report it as stuck, and name rank 0.

It also asserts the other direction — a healthy job passes through untouched
with its own exit status. Without that, every other assertion could be satisfied
by a supervisor that simply kills everything it is given.

A harness that catches hangs is worth nothing until it has caught one, so the
planted hang runs whenever the supervisor changes. It lives outside the
`test_*.py` pattern so pytest can never collect it by accident: running it under
a harness that does not kill it is the exact failure this all exists to prevent.

[#639]: https://github.com/underworldcode/underworld3/issues/639
[#661]: https://github.com/underworldcode/underworld3/issues/661
[#675]: https://github.com/underworldcode/underworld3/issues/675
