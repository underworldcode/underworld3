---
name: cetz-figures
description: Build schematic / labelled-geometry figures for underworld3 papers using Typst + cetz. Use when the figure is primarily about topology, annotation, and math-typeset labels (meshes, solver diagrams, flow charts). Prefer Python → SVG → `#image()` for data-heavy figures (fields, colormaps, arrow plots) instead.
---

# cetz-figures

Scaffold for Typst/cetz figures in `publications/**/figures/` alongside the
existing `arrays-sync-flow.typ`. This skill exists because upstream Claude
sessions draft cetz blind — this one actually compiles.

## When to use cetz

- Mesh schematics with a few labelled triangles / vertices / control points.
- Solver / data-flow diagrams (see `arrays-sync-flow.typ` in this repo).
- Anything where labels should render in the paper's math/text fonts.
- Anything that benefits from recompiling with the paper.

## When to use something else

- **Data-heavy plots** (scalar fields, colormaps, quiver plots, anything with
  dense per-pixel or per-cell data) — generate SVG from Python/matplotlib,
  include via `#image("foo.svg")`. cetz will fight you here.
- **Geometry computation** (Delaunay, intersections, interpolation) — do it in
  Python offline, emit JSON with the shape `{"vertices": [...],
  "triangles": [...], ...}`, let Typst just draw.

## When NOT to use TikZ

Evaluated and rejected:
- Slower compile than Typst.
- Drags in a LaTeX toolchain that isn't otherwise required by the project.
- No Typst math-font advantage over cetz for labels in our paper context.

Keep TikZ in back pocket only if a co-author insists on TikZ source.

## Before you draw (thesis-first discipline)

When a user hands you a figure request — especially one replacing an
existing ASCII sketch, whiteboard photo, or reference figure — **do not
start transcribing**.  The source artefact is a hypothesis about what
to communicate, not a specification.  Before opening cetz:

1. **State the thesis in one sentence.**  What is the figure arguing?
   If you can't write it plainly, you don't yet understand the figure.
   Ask the user to articulate it.

2. **Honestly audit the source.**  If the original ASCII / sketch is
   being replaced, it's often replaced *because it doesn't land well*.
   Say what's broken about it before proposing the replacement — the
   user often agrees and the new figure can do more than the old.

3. **Enumerate design decisions as explicit questions, not assumptions.**
   For a curved-boundary normals figure that's typically:
   - Geometry (circle / ellipse / arc span / zoom level)
   - Sampling density (number of facets, quadrature points per facet)
   - Overlay vs. side-by-side
   - Whether to show error quantitatively (arcs, annotations) or leave
     it as a visible angle
   - Where the figure lives in the repo (which doc / which branch)
   Present a proposed interpretation with the decisions flagged; let
   the user resolve them before you compile.

4. **Only then open cetz.**  Iterate visually — the discipline above is
   about not committing to a design prematurely, not about planning
   exhaustively.  Once you start, compile often.

The user's phrase "I'm not quite sure what this is intended to
illustrate" is the canonical trigger for this discipline.  If you hear
it (or catch yourself about to transcribe without checking), stop and
do the four steps above.

## Key gotchas (hit during iteration — not hypothetical)

1. **Don't give a helper a parameter named after a `cetz.draw` export**
   — `anchor`, `fill`, `stroke`. The cause is the `import cetz.draw: *` the
   helper needs (gotcha 6): it runs *inside* the function body and shadows the
   parameter, so the parameter name resolves to cetz's function rather than to
   the value you passed. `anchor` panics with `"Unknown anchor 'anchor' for
   element 'none'"`; `fill` gives `"expected color, gradient, tiling, or none,
   found function"` pointing into `canvas.typ`, nowhere near your code. Rename
   to `align-to`, `bg`, `edge`. See `cetz-cheatsheet.md`.

2. **Clipping is a Typst concern, not a cetz one.** cetz has no `\clip`.
   Wrap the canvas in `#box(clip: true, width: ..., height: ..., ...)` and
   draw slightly oversized inside the canvas — the box clips the overflow.

3. **Painter's algorithm — order matters.** Draw the background first, the
   highlight second. No z-index exists. Verified in `mesh-demo.typ`.

4. **Semi-transparency via `rgb(r, g, b, a)`** (alpha 0–255) or
   `color.transparentize(col, 50%)`. Both work for fill and stroke.

5. **Math in labels just works.** `content(pos, $v_1$)` renders in the
   document math font. No escape hatch needed. This is a real cetz win over
   SVG.

6. **`import cetz.draw: *` inside the canvas closure.** Without it, `line`,
   `circle`, `content` aren't in scope. Helper functions that draw need
   their own `import cetz.draw: *` line inside.

## Project layout pattern

Each blog post or paper section gets its own subdirectory under
`figures/`, so a post's figures travel together:

```
publications/blog-posts/figures/
└── <post-slug>/
    ├── <figure-name>.typ           # cetz drawing
    ├── <figure-name>.png           # committed output
    ├── <figure-name>-data.json     # (optional) precomputed geometry
    └── generate-<figure-name>.py   # (optional) Python that writes the JSON
```

Concrete example: `publications/blog-posts/figures/finding-particles/`
holds `mesh-demo.*` and `domain-demo.*` for the post
`finding-particles.md`.

The JSON intermediate is the forward bridge to underworld3 — see
`underworld-bridge.md`.

## Reference files

- `cetz-cheatsheet.md` — what worked from memory vs. needed lookup.
- `underworld-bridge.md` — JSON schema for future `uw.meshing` export.
- `examples/` — self-contained copies (each with `.typ`, `.png`, `.json`,
  and generator `.py`) of the figures this skill is scaffolded from.
  These are snapshots; the live versions may have drifted if a post or
  doc was iterated on further.
  - `mesh-demo.*` — element-level point-in-cell test.
    Live: `publications/blog-posts/figures/finding-particles/`.
  - `domain-demo.*` — parallel domain centroid ambiguity.
    Live: `publications/blog-posts/figures/finding-particles/`.
  - `facet-vs-true-normals.*` — facet normal vs. smooth-surface normal
    on a curved boundary.  Live:
    `docs/advanced/figures/curved-bc/`.

## Canonical reference in the repo

`publications/blog-posts/figures/arrays-sync-flow.typ` — prior cetz figure
in the repo, established version (0.3.4) and house style (hex colours,
helper-function pattern). Follow its conventions.
