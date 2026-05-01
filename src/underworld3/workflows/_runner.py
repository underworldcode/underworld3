"""WorkflowRunner — sequence workflow steps using produces/requires metadata.

The runner uses ``@workflow_step`` ``produces`` / ``requires`` lists to
resolve a target product by either (a) returning a cached object built
earlier in this session, (b) loading a persisted product from disk via
``WorkflowProducts``, or (c) running the step that produces it.  When
running a step, each of its ``requires`` names is resolved recursively
through the same chain.

Live (non-persistable) objects such as solvers live in the in-memory
cache only.  Persistable objects (Mesh, MeshVariable snapshots, Surface,
ndarray) are also saved to disk via ``WorkflowProducts`` and reloaded
across sessions.  The runner does not distinguish — ``WorkflowProducts``
decides per type, and save attempts that fail are silently skipped.

Step argument resolution: each parameter of a step function is matched
by *name* to either ``config`` (always passed) or to a name in the
step's ``requires`` list.  Steps that produce a single product return
the bare object; steps that produce multiple products return a dict
keyed by product name (preferred) or a tuple aligned with ``produces``.
"""

import inspect
from typing import Optional


def _collect_steps(module):
    """Walk ``module`` for ``@workflow_step`` functions.

    Returns a tuple ``(producers, steps)`` where *producers* maps
    product name → producing step and *steps* is the list of all
    decorated callables.
    """
    producers = {}
    steps = []
    for _name, obj in inspect.getmembers(module, callable):
        if not getattr(obj, "_is_workflow_step", False):
            continue
        steps.append(obj)
        for prod in getattr(obj, "workflow_produces", None) or []:
            if prod in producers:
                raise ValueError(
                    f"Two workflow steps produce {prod!r}: "
                    f"{producers[prod].__name__} and {obj.__name__}"
                )
            producers[prod] = obj
    return producers, steps


class WorkflowRunner:
    """Resolve and run workflow steps in dependency order.

    Parameters
    ----------
    module : module
        A workflow module (e.g. ``import convection_config as convection``).
    config : WorkflowConfig
        Configuration object passed to every step whose signature
        includes a ``config`` parameter.
    products : WorkflowProducts, optional
        Persistence layer.  If omitted, products live in memory only.

    Examples
    --------
    >>> runner = WorkflowRunner(convection, config, products)
    >>> runner.build("evolution_log")     # resolves all dependencies
    >>> runner.dag()                       # show steps + status
    >>> runner.rebuild("temperature_initial")  # invalidate + rebuild
    """

    def __init__(self, module, config, products=None):
        self.module = module
        self.config = config
        self.products = products
        self.cache = {}
        self._producers, self._steps = _collect_steps(module)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def get(self, name: str):
        """Return product *name*, building (and caching) if needed."""
        if name in self.cache:
            return self.cache[name]

        if self.products is not None and self.products.exists(name):
            try:
                obj = self.products.load(name)
                self.cache[name] = obj
                return obj
            except Exception:
                # Loading failed (e.g. needs a mesh argument we don't have).
                # Fall through and rebuild from the producing step.
                pass

        return self._run_step_for(name)

    # Make ``build`` a synonym of ``get`` — handy in notebooks.
    build = get

    def build_all(self):
        """Build every leaf product (one nothing else requires).

        Returns the list of leaf product names that were built.
        """
        all_produces = set()
        all_requires = set()
        for step in self._steps:
            all_produces.update(getattr(step, "workflow_produces", None) or [])
            all_requires.update(getattr(step, "workflow_requires", None) or [])
        leaves = sorted(all_produces - all_requires)
        for name in leaves:
            self.get(name)
        return leaves

    def _run_step_for(self, name: str):
        if name not in self._producers:
            available = sorted(self._producers.keys())
            raise KeyError(
                f"No workflow step produces {name!r}. "
                f"Available products: {available}"
            )
        step = self._producers[name]
        kwargs = self._resolve_kwargs(step)
        result = step(**kwargs)
        self._record_outputs(step, result)
        return self.cache[name]

    def _resolve_kwargs(self, step):
        sig = inspect.signature(step)
        requires = set(getattr(step, "workflow_requires", None) or [])
        kwargs = {}
        for pname, param in sig.parameters.items():
            if pname == "config":
                kwargs["config"] = self.config
            elif pname in requires:
                kwargs[pname] = self.get(pname)
            elif pname in self.cache:
                # Convenience: fall back to cache if the function names
                # a parameter that isn't in `requires` but is available.
                kwargs[pname] = self.cache[pname]
            elif param.default is not inspect.Parameter.empty:
                # Use the function's own default.
                pass
            else:
                # Leave unbound; Python will raise a clear TypeError.
                pass
        return kwargs

    def _record_outputs(self, step, result):
        produces = getattr(step, "workflow_produces", None) or []
        if not produces:
            return

        if len(produces) == 1:
            (name,) = produces
            self.cache[name] = result
            self._save_quietly(name, result)
            return

        if isinstance(result, dict):
            items = result
        elif isinstance(result, (list, tuple)):
            if len(result) != len(produces):
                raise ValueError(
                    f"Step {step.__name__} declares {len(produces)} "
                    f"produces but returned {len(result)} values"
                )
            items = dict(zip(produces, result))
        else:
            raise TypeError(
                f"Step {step.__name__} declares produces={produces} "
                f"but returned a {type(result).__name__}; "
                "expected dict or tuple aligned with produces"
            )

        for key, value in items.items():
            self.cache[key] = value
            self._save_quietly(key, value)

    def _save_quietly(self, name, obj):
        if self.products is None:
            return
        try:
            self.products.save(name, obj)
        except Exception:
            # Not persistable (e.g. live solver) — that's fine.
            pass

    # ------------------------------------------------------------------
    # Invalidation
    # ------------------------------------------------------------------

    def invalidate(self, name: str):
        """Drop *name* from cache and remove its on-disk product if any.

        Does **not** invalidate downstream products.  Caller is
        responsible for rebuilding anything that depended on this.
        """
        self.cache.pop(name, None)
        if self.products is not None and self.products.exists(name):
            self.products.remove(name)

    def rebuild(self, name: str):
        """Invalidate and rebuild a single product."""
        self.invalidate(name)
        return self.get(name)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def status(self, name: str) -> str:
        """Return one of ``'cached'``, ``'on_disk'``, ``'missing'``."""
        if name in self.cache:
            return "cached"
        if self.products is not None and self.products.exists(name):
            return "on_disk"
        return "missing"

    def dag(self):
        """Display all steps with their produces / requires / status.

        In Jupyter this renders as an HTML table; falls back to plain
        text in a terminal.
        """
        rows = []
        for step in sorted(self._steps, key=lambda f: f.__name__):
            produces = getattr(step, "workflow_produces", None) or []
            requires = getattr(step, "workflow_requires", None) or []
            req_str = ", ".join(requires) if requires else "—"
            if not produces:
                rows.append((step.__name__, "—", req_str, "(no product)"))
            else:
                for prod in produces:
                    rows.append(
                        (step.__name__, prod, req_str, self.status(prod))
                    )

        try:
            from IPython.display import HTML, display

            th = (
                '<th style="text-align:left; padding:4px 12px 4px 0; '
                'border-bottom:2px solid #ccc;">'
            )
            html = (
                f"<h4>{self.module.__name__} — runner status</h4>"
                f'<table style="border-collapse:collapse;"><tr>'
                f"{th}Step</th>{th}Produces</th>{th}Requires</th>"
                f"{th}Status</th></tr>"
            )
            for step_name, prod, reqs, status in rows:
                colour = {
                    "cached": "#060",
                    "on_disk": "#063",
                    "missing": "#888",
                    "(no product)": "#888",
                }.get(status, "#888")
                td = '<td style="padding:2px 12px 2px 0;'
                html += (
                    "<tr>"
                    f'{td} font-family:monospace;">{step_name}</td>'
                    f'{td} font-family:monospace; color:#555;">{prod}</td>'
                    f'{td} font-family:monospace; color:#555;">{reqs}</td>'
                    f'{td} color:{colour}; font-weight:bold;">{status}</td>'
                    "</tr>"
                )
            html += "</table>"
            display(HTML(html))

        except ImportError:
            name_w = max(len(r[0]) for r in rows)
            prod_w = max(len(r[1]) for r in rows)
            req_w = max(len(r[2]) for r in rows)
            print(f"{self.module.__name__} — runner status")
            print(
                f"  {'Step':<{name_w}}  {'Produces':<{prod_w}}  "
                f"{'Requires':<{req_w}}  Status"
            )
            print(
                f"  {'-' * name_w}  {'-' * prod_w}  "
                f"{'-' * req_w}  ------"
            )
            for step_name, prod, reqs, status in rows:
                print(
                    f"  {step_name:<{name_w}}  {prod:<{prod_w}}  "
                    f"{reqs:<{req_w}}  {status}"
                )
