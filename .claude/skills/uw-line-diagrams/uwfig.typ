// House style for underworld3 line diagrams (Typst + CeTZ).
// The same roles, weights, dash patterns and colours as uwfig.py, so a figure
// can be prototyped in matplotlib and redrawn here (or the reverse) without
// changing its look. Import from a figure with
//     #import "uwfig.typ": *
// and spread a role into a draw call:  line(a, b, ..styles.boundary)
//
// Versions: written against cetz 0.3.4 / Typst 0.13 (the versions the
// cetz-figures skill compiles with); check the marks syntax on a newer cetz.

// ---- colour tokens (secondary channel: every role also has a weight and dash)
#let ink = rgb("#1f2a33")        // walls, bodies, text, dimensions
#let mesh-grey = rgb("#9aa3aa")  // mesh edges
#let velocity = rgb("#1b6f6a")   // motion: velocity vectors, inflow profiles, streamlines
#let force = rgb("#a1522a")      // forces, pressure, tractions, applied stress
#let energy = rgb("#6b3d8f")     // heat, work, energy flux
#let tint-fluid = rgb("#eef3f6")
#let tint-solid = rgb("#e4e0da")

// ---- line roles: three weights (1.2 / 0.8 / 0.5 pt at final column width)
#let styles = (
  boundary:   (stroke: (paint: ink, thickness: 1.2pt)),                          // walls, bodies
  interface:  (stroke: (paint: ink, thickness: 0.8pt)),                          // free surface
  streamline: (stroke: (paint: velocity, thickness: 0.5pt)),                     // flow lines
  hidden:     (stroke: (paint: ink, thickness: 0.8pt, dash: "dashed")),
  cvline:     (stroke: (paint: ink, thickness: 0.8pt, dash: (5pt, 2pt, 1pt, 2pt))), // control volume
  axis:       (stroke: (paint: ink, thickness: 0.5pt, dash: (8pt, 2pt, 1pt, 2pt))), // centre line
  mesh:       (stroke: (paint: mesh-grey, thickness: 0.25pt)),
  leader:     (stroke: (paint: ink, thickness: 0.5pt)),
  // ---- arrow classes: each physical quantity has its own head and weight
  vvec:       (stroke: (paint: velocity, thickness: 0.9pt, cap: "round"),
               mark: (end: "stealth", fill: velocity, scale: 0.7, stroke: (dash: none))),
  fvec:       (stroke: (paint: force, thickness: 1.3pt, cap: "round"),
               mark: (end: "stealth", fill: force, scale: 0.9, stroke: (dash: none))),
  evec:       (stroke: (paint: energy, thickness: 1.1pt, cap: "round"),
               mark: (end: "stealth", fill: energy, scale: 0.8, stroke: (dash: none))),
  dim:        (stroke: (paint: ink, thickness: 0.5pt),
               mark: (symbol: ">", fill: ink, scale: 0.5)),                         // both ends
)

// ---- 45 degree hatching for solids / cross-sections (needs modpattern)
// #import "@preview/modpattern:0.1.0": modpattern
// #let hatch = modpattern((3pt, 3pt),
//   std.line(start: (0%, 0%), end: (100%, 100%), stroke: (paint: ink, thickness: 0.4pt)))

// ---- a parabolic inflow profile outside a vertical edge, arrows following the flow
// side: "left" = inlet (profile to the left of x0, arrows into the domain)
//       "right" = outlet (profile to the right of x0, arrows out of the domain)
#let inflow-profile(x0, y0, y1, umax, side: "left", n: 9, scale: 1.0) = {
  import cetz.draw: *
  let h = y1 - y0
  let u(y) = 4 * umax * (y - y0) * (y1 - y) / (h * h) * scale
  let sgn = if side == "left" { -1 } else { 1 }
  let pts = range(0, 41).map(i => { let y = y0 + h * i / 40; (x0 + sgn * u(y), y) })
  line(..pts, stroke: (paint: velocity, thickness: 0.9pt))
  for i in range(1, n + 1) {
    let y = y0 + h * i / (n + 1)
    if side == "left" { line((x0 - u(y), y), (x0, y), ..styles.vvec) }
    else { line((x0, y), (x0 + u(y), y), ..styles.vvec) }
  }
}

// ---- a direct label with a white backing so it reads over mesh lines
#let label(pos, body, anchor: "center") = {
  import cetz.draw: *
  content(pos, box(fill: white.transparentize(15%), inset: 2pt, radius: 2pt, text(8pt, fill: ink, body)),
          anchor: anchor)
}
