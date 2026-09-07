#import "@preview/cetz:0.3.4"

#set page(width: auto, height: auto, margin: 8pt)
#set text(size: 9pt)

// ── Top-level helpers ─────────────────────────────────────────────────
#let vsub(p, q) = (p.at(0) - q.at(0), p.at(1) - q.at(1))
#let vscale(p, s) = (s * p.at(0), s * p.at(1))
#let vlen(p) = calc.sqrt(p.at(0) * p.at(0) + p.at(1) * p.at(1))
#let vnorm(p) = { let m = vlen(p); (p.at(0) / m, p.at(1) / m) }

#let extend-to-viewport(p, q, view: 1.0) = {
  let dx = q.at(0) - p.at(0)
  let dy = q.at(1) - p.at(1)
  let BIG = 1e6
  let tx-lo = if calc.abs(dx) < 1e-9 { -BIG } else { (-view - p.at(0)) / dx }
  let tx-hi = if calc.abs(dx) < 1e-9 {  BIG } else { ( view - p.at(0)) / dx }
  let ty-lo = if calc.abs(dy) < 1e-9 { -BIG } else { (-view - p.at(1)) / dy }
  let ty-hi = if calc.abs(dy) < 1e-9 {  BIG } else { ( view - p.at(1)) / dy }
  let tx-min = calc.min(tx-lo, tx-hi)
  let tx-max = calc.max(tx-lo, tx-hi)
  let ty-min = calc.min(ty-lo, ty-hi)
  let ty-max = calc.max(ty-lo, ty-hi)
  let t-enter = calc.max(tx-min, ty-min)
  let t-exit  = calc.min(tx-max, ty-max)
  (
    (p.at(0) + t-enter * dx, p.at(1) + t-enter * dy),
    (p.at(0) + t-exit  * dx, p.at(1) + t-exit  * dy),
  )
}

// ── Shared colours ────────────────────────────────────────────────────
#let tri-stroke  = rgb("#1f3a6b")
#let tri-fill    = rgb(70, 120, 180, 55)
#let win-colour  = rgb("#059669")
#let fail-colour = rgb(194, 65, 12, 160)

// ── Panel: one figure parameterised by a JSON-loaded mesh dict ────────
#let panel(mesh, subtitle, c2-dx: 0, c2-dy: 0, c3-dx: 0, c3-dy: 0) = {
  let pt(i) = {
    let v = mesh.vertices.at(i)
    (v.at(0), v.at(1))
  }

  let h = mesh.triangles.at(mesh.highlight)
  let a = pt(h.at(0))
  let b = pt(h.at(1))
  let c = pt(h.at(2))
  let centroid = (
    (a.at(0) + b.at(0) + c.at(0)) / 3,
    (a.at(1) + b.at(1) + c.at(1)) / 3,
  )

  let n = mesh.triangles.at(mesh.neighbour)
  let n-centroid = (
    (pt(n.at(0)).at(0) + pt(n.at(1)).at(0) + pt(n.at(2)).at(0)) / 3,
    (pt(n.at(0)).at(1) + pt(n.at(1)).at(1) + pt(n.at(2)).at(1)) / 3,
  )

  let NUDGE = mesh.nudge
  let nudge-of(v) = (
    v.at(0) + NUDGE * (centroid.at(0) - v.at(0)),
    v.at(1) + NUDGE * (centroid.at(1) - v.at(1)),
  )
  let c1 = nudge-of(a)
  let c2 = nudge-of(b)
  let c3 = nudge-of(c)

  let xp = (mesh.x_p.at(0), mesh.x_p.at(1))

  let outward(p, gap: 0.11) = vscale(vnorm(vsub(p, centroid)), gap)
  let inward(p, gap: 0.08)  = vscale(vnorm(vsub(centroid, p)), gap)

  let dot(p, label: none, direction: (0.08, 0.08), align-to: "west",
          radius: 0.022, colour: black) = {
    import cetz.draw: *
    circle(p, radius: radius, fill: colour, stroke: none)
    if label != none {
      content(
        (p.at(0) + direction.at(0), p.at(1) + direction.at(1)),
        text(fill: black, label),
        anchor: align-to,
      )
    }
  }

  align(center, stack(
    spacing: 4pt,
    box(
      clip: true,
      width: 6cm,
      height: 6cm,
      stroke: 0.5pt + luma(50%),
      align(center + horizon, cetz.canvas(length: 3cm, {
        import cetz.draw: *

        // 1. Background mesh
        for t in mesh.triangles {
          line(
            pt(t.at(0)), pt(t.at(1)), pt(t.at(2)),
            close: true,
            stroke: 0.4pt + rgb("#c0c0c0"),
          )
        }

        // 2. Dotted edge extensions
        let dotted = (paint: tri-stroke, thickness: 0.5pt, dash: "dotted")
        for pair in ((a, b), (b, c), (c, a)) {
          let p = pair.at(0)
          let q = pair.at(1)
          let ends = extend-to-viewport(p, q)
          line(ends.at(0), p, stroke: dotted)
          line(q, ends.at(1), stroke: dotted)
        }

        // 3. Highlight cell
        line(a, b, c, close: true, fill: tri-fill, stroke: 1.3pt + tri-stroke)

        // 4. Connectors -- DOTTED so short segments still read as a line.
        //    Failing line to neighbour centroid (rust), winning line to
        //    nearest nudge (emerald).
        let dot-stroke(col) = (
          paint: col,
          thickness: 1.0pt,
          dash: "densely-dotted",
          cap: "round",
        )
        line(xp, n-centroid, stroke: dot-stroke(fail-colour))
        line(xp, c1,         stroke: dot-stroke(win-colour))

        // 5. Vertex dots
        let side-of(p) = if p.at(0) < centroid.at(0) { "east" } else { "west" }
        dot(a, label: $v_1$, direction: outward(a), align-to: side-of(a))
        dot(b, label: $v_2$, direction: outward(b), align-to: side-of(b))
        dot(c, label: $v_3$, direction: outward(c), align-to: side-of(c))

        // 6. Control points
        let inside-side(p) = if p.at(0) < centroid.at(0) { "west" } else { "east" }
        dot(centroid, label: $c$, direction: (0.055, 0), align-to: "west",
            radius: 0.014)
        dot(c1, label: $c_1$, direction: (0, -0.07), align-to: "north",
            radius: 0.014)
        // c_2 / c_3 label offsets: in sliver cells the inward bisector
        // direction lands the label on top of an edge.  c2-dy nudges the
        // label off it (negative pushes a label in the upper half down,
        // positive pushes a label in the lower half up).
        let c2-dir = (inward(b).at(0) + c2-dx, inward(b).at(1) + c2-dy)
        let c3-dir = (inward(c).at(0) + c3-dx, inward(c).at(1) + c3-dy)
        dot(c2, label: $c_2$, direction: c2-dir, align-to: inside-side(b),
            radius: 0.014)
        dot(c3, label: $c_3$, direction: c3-dir, align-to: inside-side(c),
            radius: 0.014)

        // 7. Neighbour centroid
        let n-side = if n-centroid.at(0) < centroid.at(0) { "east" } else { "west" }
        let n-direction = vscale(vnorm(vsub(n-centroid, centroid)), 0.11)
        dot(n-centroid, label: $c'$, direction: n-direction, align-to: n-side,
            radius: 0.014)

        // 8. Test point
        dot(xp, label: $x_p$, direction: (0.10, 0.06), align-to: "west")
      })),
    ),
    text(size: 8pt, fill: luma(40%), subtitle),
  ))
}

// ── Layout ────────────────────────────────────────────────────────────
#let mesh-normal = json("element-location-normal-data.json")
#let mesh-sliver = json("element-location-sliver-data.json")

#stack(
  dir: ltr,
  spacing: 12pt,
  panel(mesh-normal, [Normal cell]),
  panel(mesh-sliver, [Sliver edge case],
        c2-dx: 0.03, c2-dy: -0.04,
        c3-dx: 0.03, c3-dy: 0.04),
)
