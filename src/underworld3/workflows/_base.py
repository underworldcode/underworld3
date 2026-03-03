"""WorkflowConfig — base class for domain-specific workflow configurations."""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field
import yaml

# Fields shown in the header, not repeated in the table body
_HEADER_FIELDS = {"workflow_name", "description"}


def _build_rows(config):
    """Extract (name, value, description) rows, excluding header fields."""
    field_info = config.model_fields
    values = config.model_dump()
    rows = []
    for name, value in values.items():
        if name in _HEADER_FIELDS:
            continue
        info = field_info.get(name)
        desc = info.description if info else ""
        rows.append((name, value, desc or ""))
    return rows


def _build_html(config):
    """Render config as an HTML table string."""
    rows = _build_rows(config)

    title = config.workflow_name or config.__class__.__name__
    header = f"<h4>{title}</h4>" if title else ""
    if config.description:
        header += f"<p><em>{config.description}</em></p>"

    html = header + (
        '<table style="border-collapse:collapse;">'
        "<tr>"
        '<th style="text-align:left; padding:4px 12px 4px 0; '
        'border-bottom:2px solid #ccc;">Parameter</th>'
        '<th style="text-align:left; padding:4px 12px 4px 0; '
        'border-bottom:2px solid #ccc;">Value</th>'
        '<th style="text-align:left; padding:4px 12px 4px 0; '
        'border-bottom:2px solid #ccc;">Description</th>'
        "</tr>"
    )
    for name, value, desc in rows:
        v_str = repr(value) if isinstance(value, str) else str(value)
        html += (
            "<tr>"
            f'<td style="padding:2px 12px 2px 0; font-family:monospace;">{name}</td>'
            f'<td style="padding:2px 12px 2px 0; font-family:monospace;">{v_str}</td>'
            f'<td style="padding:2px 12px 2px 0; color:#666;">{desc}</td>'
            "</tr>"
        )
    html += "</table>"
    return html


class WorkflowConfig(BaseModel):
    """Base for domain-specific workflow configurations.

    Subclass this in your workflow package to define validated,
    serializable parameter sets.  The base class provides:

    * Standard metadata fields (name, description, output directory).
    * Optional reference-quantity strings that ``setup_model`` parses
      into ``uw.quantity`` objects and registers on the global Model.
    * YAML round-trip serialisation.

    Example
    -------
    >>> from underworld3.workflows import WorkflowConfig
    >>> class MyConfig(WorkflowConfig):
    ...     depth_km: float = 100.0
    ...     viscosity: float = 1e21
    ...
    >>> cfg = MyConfig(workflow_name="demo", ref_length="100 km")
    >>> cfg.save_yaml("params.yaml")
    >>> cfg2 = MyConfig.from_yaml("params.yaml")
    """

    model_config = ConfigDict(extra="allow")

    workflow_name: str = Field(default="", description="Short identifier for this workflow")
    description: str = Field(default="", description="Human-readable description")
    output_dir: str = Field(default="output", description="Directory for simulation output")

    # Reference quantities — stored as strings, parsed on demand.
    # Populate whichever are relevant; leave the rest as None.
    ref_length: Optional[str] = Field(default=None, description='e.g. "1000 km"')
    ref_viscosity: Optional[str] = Field(default=None, description='e.g. "1e21 Pa*s"')
    ref_diffusivity: Optional[str] = Field(default=None, description='e.g. "1e-6 m**2/s"')
    ref_temperature: Optional[str] = Field(default=None, description='e.g. "1500 kelvin"')
    ref_density: Optional[str] = Field(default=None, description='e.g. "3300 kg/m**3"')
    ref_velocity: Optional[str] = Field(default=None, description='e.g. "5 cm/year"')

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def view(self):
        """Display configuration as a formatted table.

        In Jupyter this renders as an HTML table via
        ``IPython.display``.  In a terminal it falls back to a
        plain-text table.
        """
        try:
            from IPython.display import HTML, display
            display(HTML(_build_html(self)))
        except ImportError:
            rows = _build_rows(self)
            name_w = max(len(r[0]) for r in rows)
            val_w = max(len(str(r[1])) for r in rows)
            fmt = f"  {{:<{name_w}}}  {{:<{val_w}}}  {{}}"
            title = self.workflow_name or self.__class__.__name__
            print(title)
            if self.description:
                print(f"  {self.description}")
            print(fmt.format("Parameter", "Value", "Description"))
            print(fmt.format("-" * name_w, "-" * val_w, "-" * 20))
            for name, value, desc in rows:
                print(fmt.format(name, str(value), desc))

    def _repr_html_(self):
        """Jupyter auto-display: render as HTML table when a cell returns a config."""
        return _build_html(self)

    # ------------------------------------------------------------------
    # Model integration
    # ------------------------------------------------------------------

    def setup_model(self, name: Optional[str] = None):
        """Create (or reset) a ``uw.Model`` with reference quantities from this config.

        Parameters
        ----------
        name : str, optional
            Model name.  Defaults to ``self.workflow_name``.

        Returns
        -------
        uw.Model
        """
        import underworld3 as uw
        from ._utils import parse_quantity

        uw.reset_default_model()
        model = uw.get_default_model()
        if name or self.workflow_name:
            model.name = name or self.workflow_name

        # Collect reference quantities that are set
        ref_kwargs = {}
        _mapping = {
            "ref_length": "domain_depth",
            "ref_viscosity": "mantle_viscosity",
            "ref_diffusivity": "thermal_diffusivity",
            "ref_temperature": "mantle_temperature",
            "ref_density": "mantle_density",
            "ref_velocity": "plate_velocity",
        }
        for attr, kwarg_name in _mapping.items():
            value = getattr(self, attr)
            if value is not None:
                ref_kwargs[kwarg_name] = parse_quantity(value)

        if ref_kwargs:
            model.set_reference_quantities(**ref_kwargs)

        # Store config snapshot in metadata
        model.metadata["workflow_config"] = self.model_dump()

        return model

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save_yaml(self, path) -> None:
        """Write configuration to a YAML file.

        Parameters
        ----------
        path : str or Path
            Destination file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(
                self.model_dump(),
                f,
                default_flow_style=False,
                sort_keys=False,
            )

    @classmethod
    def from_yaml(cls, path) -> "WorkflowConfig":
        """Load configuration from a YAML file.

        Parameters
        ----------
        path : str or Path
            Source file path.

        Returns
        -------
        WorkflowConfig
            A new instance of this class (or subclass).
        """
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
