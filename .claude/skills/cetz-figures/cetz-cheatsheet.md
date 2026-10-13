# cetz cheatsheet — honest edition

Separates **what I used correctly from memory** (so you probably can too)
from **what actually needed diagnosis** (so you should check the manual
at https://cetz-package.github.io/docs/ rather than trust the pattern).

Version: cetz 0.3.4, Typst 0.13.1. Don't assume these patterns hold for
later cetz versions — the API has been unstable between minor versions.

**External reference worth checking first:**
<https://diagrams.janosh.dev> — curated gallery of ~130 scientific
diagrams with source (~120 in cetz, ~110 in TikZ). Heaviest coverage is
theoretical physics and ML; lighter on FE/meshes specifically, but
strong patterns for coordinate frames, Euler angles, 3D orientation,
and lattice structures.  Worth browsing before inventing a new figure
from scratch — the answer to "how do people draw X in cetz" is often
sitting there.

## Worked from memory (verified correct)

```typst
#import "@preview/cetz:0.3.4"

#cetz.canvas(length: 4cm, {
  import cetz.draw: *

  // Straight line / polyline / closed polygon
  line((0, 0), (1, 0), (1, 1), close: true,
       fill: rgb(70, 120, 180, 60),
       stroke: 1.2pt + rgb("#1f3a6b"))

  // Filled dot
  circle((0.5, 0.5), radius: 0.02, fill: black, stroke: none)

  // Label at a position — math content renders in document math font
  content((0.5, 0.5), $v_1$, anchor: "west")
})
```

- `#box(clip: true, width: 8cm, height: 8cm, cetz.canvas(...))` — the
  *Typst* box does the clipping. cetz itself doesn't clip.
- `rgb("#b8b8b8")` for hex colours, `rgb(r, g, b, a)` for RGBA (alpha 0–255).
- `json("file.json")` from outside the canvas, then `.at(i)` and field
  access inside. Values flow through into cetz coordinates cleanly.
- `for tri in mesh.triangles { line(...) }` — plain Typst for-loops work
  inside the canvas closure.

## Needed diagnosis (check the manual if you do this)

### Parameter names shadowed by `import cetz.draw: *`

A helper that draws needs `import cetz.draw: *` in its own body (see below).
That import runs *inside* the function, **after** the parameters are bound, and
shadows any parameter sharing a name with a `cetz.draw` export — `anchor`,
`fill`, `stroke`. The parameter name then resolves to cetz's function instead of
your value.

```typst
// BROKEN — panic: "Unknown anchor 'anchor' for element 'none'"
#let dot(p, label, anchor: "west") = {
  import cetz.draw: *
  circle(p, radius: 0.02, fill: black)
  content(p, label, anchor: anchor)
}

// BROKEN — "expected color, gradient, tiling, or none, found function",
// reported inside cetz's own canvas.typ, nowhere near your code
#let panel(tl, br, fill, edge) = {
  import cetz.draw: *
  rect(tl, br, fill: fill, stroke: (paint: edge, thickness: 1pt))
}
```

```typst
// FIX — rename the parameter
#let dot(p, label, align-to: "west") = { ... content(p, label, anchor: align-to) }
#let panel(tl, br, bg, edge)        = { ... rect(tl, br, fill: bg, ...) }
```

Verified by isolation: the same helper **without** the inner import compiles
fine, which is what identifies the import as the culprit rather than the
parameter name itself. So the rule is not a blacklist of names — it is "do not
name a parameter after anything `cetz.draw` exports", and if the helper does not
actually draw, drop the inner import instead.

Diagnosis cost: two compile failures across two sessions. Neither error message
points at the parameter.

### Shadowing a Typst builtin, especially `h`

Same class, wider blast radius. `h` is Typst's horizontal-spacing function, so a
figure that defines a surface height as `h(x)` breaks every `#h(0.4em)` used for
spacing in a label — with an error that names neither:

```typst
let h(x) = 0.9 * calc.exp(-x * x)      // BROKEN
content(p, [$a$, #h(0.6em) $b$])       // "cannot add length and float"
```

Rename the domain function (`surf-y`, `depth`, `topo`). `v`, `text`, `line`,
`rect`, `circle`, `content` and `place` are all live names too.

### `let f(x) = ...` ends at the line break

A multi-line arithmetic expression needs parentheses, or the statement ends at
the first newline and the continuation is parsed as a fresh expression —
reported as `unknown variable: x`, which sends you looking in the wrong place:

```typst
// BROKEN
let surf-y(x) = 0.95 * calc.exp(-calc.pow((x + 1.75) / 1.3, 2))
              - 0.80 * calc.exp(-calc.pow((x - 1.70) / 1.15, 2))

// FIX — wrap the whole expression
let surf-y(x) = (0.95 * calc.exp(-calc.pow((x + 1.75) / 1.3, 2))
                 - 0.80 * calc.exp(-calc.pow((x - 1.70) / 1.15, 2)))
```

### Multi-line `content()` anchored `"west"` rides up over what is above it

`"west"` centres the block vertically on the anchor point, so a two- or
three-line caption placed below a heading grows *upwards* into it. Nothing
errors; the text simply overlaps and you only notice in the PNG.

```typst
content((x, y - 0.3), anchor: "west", note)        // overlaps when note wraps
content((x, y - 0.3), anchor: "north-west", note)  // grows downward — use this
```

Rule of thumb: any label whose text might wrap gets a `north-*` anchor.

### Anchor values for `content()`

Confirmed to work: `"north"`, `"south"`, `"east"`, `"west"`,
`"north-east"`, `"south-east"`, `"north-west"`, `"south-west"`, `"center"`.

Not tried in this session: named anchors of other elements (e.g.
`"my-rect.east"`), element-specific anchors beyond compass. Verify
against the cetz anchors docs if you need those.

### Content wider than the Typst box needs `align()` to re-centre

`#box(clip: true, width: W, height: H, cetz.canvas(...))` does not centre
the cetz canvas if the cetz bounding box is larger than `W × H` (e.g. a
mesh drawn oversize for clipping).  Typst default-aligns the canvas to
the top-left, which shifts the interesting region off-frame.  Wrap:

```typst
#box(clip: true, width: W, height: H,
  align(center + horizon, cetz.canvas(length: L, { ... })))
```

Encountered when the background mesh extent grew past `[-1, 1]` —
first compile looked fine because content fit; the next compile didn't.

### Dashed / dotted strokes in cetz

Pass a dict for the stroke:

```typst
stroke: (paint: rgb("#1f3a6b"), thickness: 0.5pt, dash: "dotted")
stroke: (paint: rgb("#059669"), thickness: 0.7pt, dash: "dashed")
```

Valid `dash` values include `"dotted"`, `"dashed"`, `"densely-dotted"`,
`"loosely-dotted"`, etc. (Typst stroke spec, not a cetz extension.)

### Line-to-viewport clipping is your job

cetz does not clip lines at an imaginary viewport. If you want to
extend the line through `(p, q)` out to the edge of a `[-V, V]²` box,
compute the `t` values at each axis crossing, take the inner `t-enter`
and `t-exit`, and evaluate the line there. See `extend-to-viewport` in
`examples/mesh-demo.typ`.

### Draw helpers that take `cetz.draw` primitives

Helper functions defined **outside** `cetz.canvas({...})` need their
own `import cetz.draw: *` inside the function body:

```typst
#let dot(p, label, ...) = {
  import cetz.draw: *      // REQUIRED — cetz.draw symbols not in scope otherwise
  circle(p, ...)
  content(...)
}
```

This feels like it shouldn't be necessary but is (cetz 0.3.4).

## Probably wrong / would need checking

These are *guesses* from adjacent Typst/cetz knowledge, not verified here:

- `plot` module usage (axes, line plots, scatter plots). Upstream sessions
  were shaky on this; assume the API has drifted and check the cetz plot
  docs before using.
- `decorations` module (brace, waveform, etc.). Not touched in this session.
- Named-anchor references across draw calls via `name:` — syntax is
  `element-name.anchor-name` but context behaviour needs testing.
- Gradient fills — not used, not verified.
- Transformations (`translate`, `rotate`, `scale`) — likely work but not
  exercised here.

## Compile workflow

```bash
# First run downloads cetz and its deps (small, ~90 KB total)
typst compile mesh-demo.typ mesh-demo.png

# PDF is default if no output specified
typst compile mesh-demo.typ        # → mesh-demo.pdf

# Watch mode for iteration
typst watch mesh-demo.typ mesh-demo.png
```

## Data-driven figures with JSON

The preferred pattern for mesh/particle figures is to generate data in
Python (scipy.spatial.Delaunay, numpy), export as JSON, and render in
Typst/cetz. This separates the geometry computation from the drawing.

```typst
#let mesh = json("mesh-data.json")
#let pt(i) = {
  let v = mesh.vertices.at(i)
  (v.at(0), v.at(1))
}
// Loop over triangles
for tri in mesh.triangles {
  line(pt(tri.at(0)), pt(tri.at(1)), pt(tri.at(2)),
       close: true, fill: fill-col, stroke: stroke-mesh)
}
```

### Connect nodes by triangulating, never by distance

A tempting shortcut for a mesh figure is to join node pairs closer than some
radius. It is not a triangulation: it silently omits nodes near the edge of the
domain and in sparse regions, and the figure looks *almost* right, which is
worse than looking wrong. Run a real Delaunay and draw its triangles.

Two follow-on details, both of which show up immediately on a geophysical
domain:

- **Delaunay fills the convex hull.** A domain bounded above by topography or a
  free surface is not convex, so triangles appear above the surface. Cull them
  by centroid: `if centroid_y <= surface_y(centroid_x)`.
- **A perfect lattice gives degenerate triangles.** Stagger alternate rows by
  half a spacing (or jitter) so Delaunay has an unambiguous choice and produces
  triangles rather than near-right-angle slivers.

### Python generator pattern

```python
from scipy.spatial import Delaunay
# Jittered equilateral lattice → Delaunay → JSON
tri = Delaunay(points)
data = {
    "vertices": [[float(x), float(y)] for x, y in points],
    "triangles": [[int(i), int(j), int(k)] for i, j, k in tri.simplices],
    # ... domain membership, centroids, test points, etc.
}
json.dump(data, open("mesh-data.json", "w"), indent=2)
```

### Enumeration with index

```typst
// .enumerate() gives (index, value) pairs — used for domain colouring
for (idx, tri) in mesh.triangles.enumerate() {
  let d = get-domain(idx)
  let fill-col = if d == "A" { fill-a } else { fill-b }
  line(pt(tri.at(0)), pt(tri.at(1)), pt(tri.at(2)),
       close: true, fill: fill-col, stroke: stroke-mesh)
}
```

### High-DPI PNG output

```bash
typst compile figure.typ figure.png --ppi 300
```

## Lessons from domain-demo figure

- **Multiple domain colours**: Use low-alpha fills (25–45) so mesh edges
  show through. Peripheral domains use even lower alpha to stay in background.
- **No heavy boundary lines needed**: Domain colours communicate the
  partition without explicit boundary drawing. This looks more natural.
- **Smaller dots and tight labels** for a cleaner look: `radius: 0.028–0.032`,
  label offsets of `(0.06, 0.06)` or `(0.08, 0.06)`.
- **Finer mesh** (SPACING ≈ 0.28, ~400 vertices, ~800 triangles) reads as
  a realistic FE mesh. Coarser meshes (SPACING ≈ 1.25) work for
  element-level diagrams.
- **View clipping**: Use `VIEW` parameter in JSON and `in-view()` check
  to clip triangles at the edge of the rendered region.

## Visual idioms that worked

### Dashed-on-top-of-solid reveals coincidence

When two arrows (or two paths) represent quantities that *sometimes*
coincide and *sometimes* diverge, draw the **dashed** one LAST so it
layers on top of the **solid** one.  Where they overlap, the dashes
interleave with the solid underneath and both colours remain visible
— reading unmistakably as "they coincide here".  If the solid were on
top, it would simply cover the dashed and the coincidence would be
invisible.

Applied in `facet-vs-true-normals.typ`: the rust `mesh.Gamma` arrow is
drawn first, the green dashed true normal on top.  At each facet
midpoint the green dashes sit over the rust — the coincidence is
obvious.  At off-centre quadrature points both arrows are visible as
distinct directions.

### Quadrature choice encodes pedagogy

If your figure illustrates *within-element variation* of a quantity,
the quadrature rule you pick determines whether the variation is
visible at all:

- **2-point Gauss–Legendre** (±1/√3 on [-1, 1]): both samples off
  the element midpoint.  Can show *average* per-element behaviour
  but hides any quantity that's zero at the midpoint.
- **3-point Gauss–Legendre** (0, ±√(3/5)): one sample AT the
  midpoint plus two off-centre.  Reveals both the midpoint case and
  the at-the-edge case in the same picture.

Switching from 2-point to 3-point was the key move that made the
curved-boundary figure communicate "the error is zero at facet
midpoints" — not just "there is an error".

### Legend in empty interior space beats per-arrow labels

Per-arrow labels near the canvas edge fight cetz's bbox calculation
and often clip at the PNG boundary.  If the figure has an obvious
empty interior region, put a small 2-row legend there:

```typst
// Row 1
line((lx, ly), (lx + 0.28, ly), stroke: gamma-stroke,
     mark: (end: ">", fill: gamma-colour))
content((lx + 0.36, ly), text(fill: gamma-colour, $hat(n)_Gamma$ + [ (label)]),
        anchor: "west")
// Row 2, below
```

The short sample stroke uses the *same* stroke dict as the arrows in
the figure, so it looks identical.  No leader lines, no per-arrow
crowding.

### Short arrows read better than long

Arrow lengths around **0.25–0.3 world units** (with canvas `length:
3cm`) stay visually subordinate to the geometry they annotate.
Longer arrows (0.4+) tend to dominate and obscure the mesh.  Start
short; lengthen only if the arrow directions are genuinely hard to
read.

### Explicit page dimensions + aligned canvas = reliable padding

For docs figures that must look crisp in a PNG of specific dimensions:

```typst
#set page(width: 12cm, height: 8cm, margin: 6pt)
#align(center + horizon, cetz.canvas(length: 3cm, { ... }))
```

This decouples the *rendered size* from cetz's (sometimes generous,
sometimes flaky) auto-bbox.  Your content scales predictably and text
labels near the edges have room.  Use it any time the figure goes
into a doc as a fixed-size image.

## House style in this repo

Figures live in per-post subdirectories under `publications/blog-posts/figures/`:
- `finding-particles/mesh-demo.typ` — element-level inside/outside test with control points
- `finding-particles/domain-demo.typ` — multi-domain parallel mesh with centroid ambiguity
- `arrays-sync-flow.typ` — flow diagram with nodes, regions, bezier connectors
  (older figure, not yet moved into a per-post subdirectory)

Patterns:
- Custom helper functions declared at the top of the canvas closure
  (`node`, `region`, `elabel`, `dot`).
- Hex colours via `rgb("#...")`.
- Arrows via `mark: (end: ">", fill: black)`.
- Bezier connectors via `bezier(start, end, ctrl1, ctrl2, ...)`.
- Data from JSON files generated by companion Python scripts.
