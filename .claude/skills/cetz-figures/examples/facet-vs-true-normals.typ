#import "@preview/cetz:0.3.4"

// Explicit page size + aligned-centred canvas lets us control padding
// independently of whatever bbox cetz reports for its content.
#set page(width: 12cm, height: 8cm, margin: 6pt)
#set text(size: 10pt)

#let data = json("curved-bc-data.json")

// ── Colours ───────────────────────────────────────────────────────────
#let arc-colour   = rgb("#b0b0b0")          // true smooth curve
#let facet-colour = rgb("#1f3a6b")          // navy — mesh facets
#let gamma-colour = rgb("#c2410c")          // rust — PETSc facet normal
#let true-colour  = rgb("#059669")          // emerald — true surface normal
#let centre-colour = black

// ── Geometry access ───────────────────────────────────────────────────
#let centre     = (data.centre.at(0),     data.centre.at(1))
#let radius-end = (data.radius_end.at(0), data.radius_end.at(1))

// ── Figure ────────────────────────────────────────────────────────────
#align(center + horizon, cetz.canvas(length: 3cm, {
  import cetz.draw: *

  // 1. True smooth arc — thin dashed grey polyline.
  let arc-stroke = (paint: arc-colour, thickness: 0.6pt, dash: "dashed")
  for i in range(data.arc_points.len() - 1) {
    let p0 = data.arc_points.at(i)
    let p1 = data.arc_points.at(i + 1)
    line((p0.at(0), p0.at(1)), (p1.at(0), p1.at(1)), stroke: arc-stroke)
  }

  // 2. Radius-of-curvature indicator from centre to the arc midpoint.
  line(centre, radius-end,
       stroke: (paint: arc-colour, thickness: 0.5pt, dash: "dotted"))
  //    Label the radius midway along it, offset perpendicular.
  let r-label-pos = (
    0.5 * (centre.at(0) + radius-end.at(0)) + 0.07,
    0.5 * (centre.at(1) + radius-end.at(1)),
  )
  content(r-label-pos, text(fill: arc-colour, size: 10pt, $R$),
          anchor: "west")

  // 3. Facet polyline — thin black.  The control points (vertex dots,
  //    Gauss-point dots) carry the visual weight; the segments between
  //    them are just the domain boundary.
  for i in range(data.facet_vertices.len() - 1) {
    let p0 = data.facet_vertices.at(i)
    let p1 = data.facet_vertices.at(i + 1)
    line((p0.at(0), p0.at(1)), (p1.at(0), p1.at(1)),
         stroke: 0.6pt + black)
  }
  //    Facet vertex markers
  for v in data.facet_vertices {
    circle((v.at(0), v.at(1)), radius: 0.038,
           fill: facet-colour, stroke: none)
  }

  // 4. Quadrature points with their two normals.
  //    - true normal (emerald, dashed)   ← what free-slip wants
  //    - facet normal (rust, solid)      ← what mesh.Gamma gives
  let arrow-len    = 0.28
  let true-stroke  = (paint: true-colour,  thickness: 0.9pt, dash: "dashed")
  let gamma-stroke = 1.3pt + gamma-colour
  for q in data.quadrature {
    let pos = (q.pos.at(0), q.pos.at(1))
    let tn  = q.true_normal
    let fn  = q.facet_normal

    // Facet normal arrow drawn FIRST, so the dashed true normal
    // layers on top.  Where the two coincide (facet midpoints), the
    // green dashes interleave with the rust underneath and both
    // colours stay visible — it reads as "they overlap here".
    let tip-f = (pos.at(0) + arrow-len * fn.at(0),
                 pos.at(1) + arrow-len * fn.at(1))
    line(pos, tip-f, stroke: gamma-stroke,
         mark: (end: ">", fill: gamma-colour))

    // True normal arrow on top
    let tip-t = (pos.at(0) + arrow-len * tn.at(0),
                 pos.at(1) + arrow-len * tn.at(1))
    line(pos, tip-t, stroke: true-stroke,
         mark: (end: ">", fill: true-colour))

    // Small quadrature dot
    circle(pos, radius: 0.024, fill: black, stroke: none)
  }

  // 5. Centre dot + label
  circle(centre, radius: 0.038, fill: centre-colour, stroke: none)
  content((centre.at(0) + 0.06, centre.at(1) - 0.06),
          $O$, anchor: "north-west")

  // 6. Legend in the empty space below the arc.  Short sample strokes
  //    next to each label, so the figure is self-explaining without
  //    leader lines or per-arrow annotations.
  let lx = 0.25       // legend x start (inside the open area)
  let ly = 0.55       // top row y
  let row = 0.22      // row spacing
  let sample-len = 0.28

  // Row 1: facet normal sample
  line((lx, ly), (lx + sample-len, ly),
       stroke: gamma-stroke,
       mark: (end: ">", fill: gamma-colour))
  content((lx + sample-len + 0.08, ly),
          text(fill: gamma-colour, size: 9.5pt,
               $hat(n)_Gamma$ + [ (facet, `mesh.Gamma`)]),
          anchor: "west")

  // Row 2: true normal sample
  line((lx, ly - row), (lx + sample-len, ly - row),
       stroke: true-stroke,
       mark: (end: ">", fill: true-colour))
  content((lx + sample-len + 0.08, ly - row),
          text(fill: true-colour, size: 9.5pt,
               $hat(n)_"true"$ + [ (smooth surface)]),
          anchor: "west")
}))
