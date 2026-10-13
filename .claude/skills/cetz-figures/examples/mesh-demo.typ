#import "@preview/cetz:0.3.4"

#set page(width: auto, height: auto, margin: 8pt)
#set text(size: 10pt)

// ── Mesh data ─────────────────────────────────────────────────────────
#let mesh = json("mesh-data.json")

#let pt(i) = {
  let v = mesh.vertices.at(i)
  (v.at(0), v.at(1))
}

// The central triangle is a genuine Delaunay cell.  Python forces it
// into the triangulation and canonicalises its vertex order so
// triangle[highlight].at(k) is always TRI[k].
#let h = mesh.triangles.at(mesh.highlight)
#let a = pt(h.at(0))
#let b = pt(h.at(1))
#let c = pt(h.at(2))

#let centroid = (
  (a.at(0) + b.at(0) + c.at(0)) / 3,
  (a.at(1) + b.at(1) + c.at(1)) / 3,
)

// ── Vector helpers ────────────────────────────────────────────────────
#let vsub(p, q) = (p.at(0) - q.at(0), p.at(1) - q.at(1))
#let vadd(p, q) = (p.at(0) + q.at(0), p.at(1) + q.at(1))
#let vscale(p, s) = (s * p.at(0), s * p.at(1))
#let vlen(p) = calc.sqrt(p.at(0) * p.at(0) + p.at(1) * p.at(1))
#let vnorm(p) = { let m = vlen(p); (p.at(0) / m, p.at(1) / m) }
#let vmid(p, q) = ((p.at(0) + q.at(0)) / 2, (p.at(1) + q.at(1)) / 2)
#let dist(p, q) = vlen(vsub(p, q))

// ── Edge-midpoint markers (pair per edge: inside + outside) ───────────
// Dot radius is 0.022 world units; gap 0.030 places the dot just clear.
#let edge-marker(p, q, gap: 0.030, side: "inside") = {
  let m = vmid(p, q)
  let dir = vnorm(vsub(centroid, m))
  let sign = if side == "outside" { -1.0 } else { 1.0 }
  vadd(m, vscale(dir, sign * gap))
}

#let m-ab = edge-marker(a, b)
#let m-bc = edge-marker(b, c)
#let m-ca = edge-marker(c, a)

#let m-ab-out = edge-marker(a, b, side: "outside")
#let m-bc-out = edge-marker(b, c, side: "outside")
#let m-ca-out = edge-marker(c, a, side: "outside")

// For each face we store (endpoint0, endpoint1, inside-marker, outside-marker).
#let faces = (
  (a, b, m-ab, m-ab-out),
  (b, c, m-bc, m-bc-out),
  (c, a, m-ca, m-ca-out),
)

// 2D cross product — used for the side-of-line test below.
#let cross2d(u, v) = u.at(0) * v.at(1) - u.at(1) * v.at(0)

// Per-face rule: connect the test point to the *inside* marker if the
// test point lies on the same side of the edge as the centroid, else
// to the *outside* marker.  A point inside the triangle lands on the
// centroid side of all three edges (→ all black); a point outside
// crosses at least one edge to the other side (→ at least one red).
#let marker-for(test, e0, e1, inside-marker, outside-marker) = {
  let edge = vsub(e1, e0)
  let s-test = cross2d(edge, vsub(test, e0))
  let s-ref  = cross2d(edge, vsub(centroid, e0))
  if s-test * s-ref > 0 { inside-marker } else { outside-marker }
}

// ── Sample (test) points ──────────────────────────────────────────────
// x_q: inside the triangle, pulled toward v2 so it sits closer to AB
// and BC, but shifted left enough that its line to the CA midpoint
// clears the centroid.
#let x-q = (
  0.25 * a.at(0) + 0.60 * b.at(0) + 0.15 * c.at(0),
  0.25 * a.at(1) + 0.60 * b.at(1) + 0.15 * c.at(1),
)
// x_p: outside CA, nudged right so the connector to the AB midpoint
// doesn't graze the CA outside marker en route.
#let x-p = (-0.15, 0.80)

// ── Line-through-edge extended to the viewport boundary ───────────────
// Returns (before, after) — the two points where the infinite line
// through (p, q) crosses the axis-aligned box [-VIEW, VIEW]^2.
#let VIEW = 1.0
#let extend-to-viewport(p, q) = {
  let dx = q.at(0) - p.at(0)
  let dy = q.at(1) - p.at(1)
  let BIG = 1e6
  let tx-lo = if calc.abs(dx) < 1e-9 { -BIG } else { (-VIEW - p.at(0)) / dx }
  let tx-hi = if calc.abs(dx) < 1e-9 {  BIG } else { ( VIEW - p.at(0)) / dx }
  let ty-lo = if calc.abs(dy) < 1e-9 { -BIG } else { (-VIEW - p.at(1)) / dy }
  let ty-hi = if calc.abs(dy) < 1e-9 {  BIG } else { ( VIEW - p.at(1)) / dy }
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

// ── Colours ───────────────────────────────────────────────────────────
#let tri-stroke  = rgb("#1f3a6b")          // navy — triangle
#let tri-fill    = rgb(70, 120, 180, 55)   // tinted navy
#let out-marker  = rgb("#c2410c")          // rust — outside midpoint dots
#let x-q-colour  = rgb("#059669")          // emerald — inside sample
#let x-p-colour  = rgb("#7c3aed")          // violet  — outside sample

// ── Labelled dot helper ───────────────────────────────────────────────
#let dot(p, label: none, direction: (0.08, 0.08), align-to: "west",
         radius: 0.022, colour: black, label-colour: none) = {
  import cetz.draw: *
  circle(p, radius: radius, fill: colour, stroke: none)
  if label != none {
    let lcolour = if label-colour == none { black } else { label-colour }
    content(
      (p.at(0) + direction.at(0), p.at(1) + direction.at(1)),
      text(fill: lcolour, label),
      anchor: align-to,
    )
  }
}

#let outward(p, gap: 0.11) = vscale(vnorm(vsub(p, centroid)), gap)

// ── Figure ────────────────────────────────────────────────────────────
#box(
  clip: true,
  width: 8cm,
  height: 8cm,
  stroke: 0.5pt + luma(50%),
  align(center + horizon, cetz.canvas(length: 4cm, {
    import cetz.draw: *

    // 1. Background mesh
    for tri in mesh.triangles {
      line(
        pt(tri.at(0)), pt(tri.at(1)), pt(tri.at(2)),
        close: true,
        stroke: 0.4pt + rgb("#c0c0c0"),
      )
    }

    // 2. Dotted extensions — triangle edges continued to the viewport.
    let dotted = (paint: tri-stroke, thickness: 0.5pt, dash: "dotted")
    for pair in ((a, b), (b, c), (c, a)) {
      let p = pair.at(0)
      let q = pair.at(1)
      let ends = extend-to-viewport(p, q)
      line(ends.at(0), p, stroke: dotted)
      line(q, ends.at(1), stroke: dotted)
    }

    // 3. Main triangle — fill + bold stroke on top of mesh and dots.
    line(a, b, c, close: true, fill: tri-fill, stroke: 1.3pt + tri-stroke)

    // 4. Per-face connectors.  Each test point draws three dashed lines,
    //    one per face, landing on black or red by the side-of-edge rule.
    //    All-black ⇒ inside; any-red ⇒ outside.
    let dashed-stroke(col) = (paint: col, thickness: 0.7pt, dash: "dashed")
    for face in faces {
      let e0 = face.at(0)
      let e1 = face.at(1)
      let m-in = face.at(2)
      let m-out = face.at(3)
      line(x-q, marker-for(x-q, e0, e1, m-in, m-out),
           stroke: dashed-stroke(x-q-colour))
      line(x-p, marker-for(x-p, e0, e1, m-in, m-out),
           stroke: dashed-stroke(x-p-colour))
    }

    // 5. Vertex dots + labels outward from centroid.
    let side-of(p) = if p.at(0) < centroid.at(0) { "east" } else { "west" }
    dot(a, label: $v_1$, direction: outward(a), align-to: side-of(a))
    dot(b, label: $v_2$, direction: outward(b), align-to: side-of(b))
    dot(c, label: $v_3$, direction: outward(c), align-to: side-of(c))

    // 6. Midpoint markers — inside (black) + outside companions (rust).
    dot(m-ab); dot(m-bc); dot(m-ca)
    dot(m-ab-out, colour: out-marker)
    dot(m-bc-out, colour: out-marker)
    dot(m-ca-out, colour: out-marker)

    // 7. Centroid + two sample points (each in its own colour).
    //    Centroid dot is smaller and its label tight against it so it
    //    reads as a derived reference rather than competing with the
    //    test points.
    dot(centroid, label: $c$, direction: (0.055, 0), align-to: "west",
        radius: 0.014)
    dot(x-q, label: $x_q$, direction: (0.10, 0.05), align-to: "west",
        colour: x-q-colour, label-colour: x-q-colour)
    dot(x-p, label: $x_p$, direction: (0.10, 0.05), align-to: "west",
        colour: x-p-colour, label-colour: x-p-colour)
  })),
)
