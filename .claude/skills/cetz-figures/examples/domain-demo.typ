#import "@preview/cetz:0.3.4"

#set page(width: auto, height: auto, margin: 8pt)
#set text(size: 9pt)

// ── Mesh data ─────────────────────────────────────────────────────────
#let mesh = json("domain-demo-data.json")

#let pt(i) = {
  let v = mesh.vertices.at(i)
  (v.at(0), v.at(1))
}

#let VIEW = mesh.view
#let ca = (mesh.centroid_a.at(0), mesh.centroid_a.at(1))
#let cb = (mesh.centroid_b.at(0), mesh.centroid_b.at(1))
#let cc = (mesh.domain_centroids.C.at(0), mesh.domain_centroids.C.at(1))
#let cd = (mesh.domain_centroids.D.at(0), mesh.domain_centroids.D.at(1))
#let xp = (mesh.test_point.at(0), mesh.test_point.at(1))

// ── Helpers ──────────────────────────────────────────────────────────
#let tri-centroid(tri) = {
  let a = pt(tri.at(0))
  let b = pt(tri.at(1))
  let c = pt(tri.at(2))
  ((a.at(0) + b.at(0) + c.at(0)) / 3,
   (a.at(1) + b.at(1) + c.at(1)) / 3)
}

#let in-view(p) = {
  calc.abs(p.at(0)) <= VIEW and calc.abs(p.at(1)) <= VIEW
}

// ── Domain lookup ────────────────────────────────────────────────────
#let domain-a-set = mesh.domains.A
#let domain-b-set = mesh.domains.B
#let domain-c-set = mesh.domains.C
#let domain-d-set = mesh.domains.D

#let get-domain(idx) = {
  if domain-a-set.contains(idx) { "A" }
  else if domain-b-set.contains(idx) { "B" }
  else if domain-c-set.contains(idx) { "C" }
  else { "D" }
}

// ── Colours ──────────────────────────────────────────────────────────
#let fill-a     = rgb(70, 110, 180, 45)    // blue tint
#let fill-b     = rgb(180, 70, 70, 45)     // red tint
#let fill-c     = rgb(90, 155, 90, 40)     // green tint
#let fill-d     = rgb(180, 160, 70, 40)    // yellow tint

#let stroke-mesh = 0.3pt + rgb("#b0b0b0")

#let colour-a   = rgb("#2563eb")           // blue
#let colour-b   = rgb("#dc2626")           // red
#let colour-c   = rgb("#16a34a")           // green
#let colour-d   = rgb("#ca8a04")           // amber
#let colour-xp  = rgb("#7c3aed")           // violet

// ── Dot helper ───────────────────────────────────────────────────────
#let dot(p, label: none, direction: (0.06, 0.06), align-to: "west",
         radius: 0.028, colour: black, label-colour: none) = {
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

// ── Figure ───────────────────────────────────────────────────────────
#box(
  clip: true,
  width: 10cm,
  height: 10cm,
  stroke: 0.5pt + luma(50%),
  align(center + horizon, cetz.canvas(length: 2.5cm, {
    import cetz.draw: *

    // 1. Draw all triangles with domain shading
    for (idx, tri) in mesh.triangles.enumerate() {
      let c = tri-centroid(tri)
      if in-view(c) {
        let d = get-domain(idx)
        let fill-col = if d == "A" { fill-a }
                       else if d == "B" { fill-b }
                       else if d == "C" { fill-c }
                       else { fill-d }
        line(
          pt(tri.at(0)), pt(tri.at(1)), pt(tri.at(2)),
          close: true,
          fill: fill-col,
          stroke: stroke-mesh,
        )
      }
    }

    // 2. Dashed lines from test point to all centroids
    let dashed-a = (paint: colour-a, thickness: 0.8pt, dash: "dashed")
    let dashed-b = (paint: colour-b, thickness: 0.8pt, dash: "dashed")
    let dashed-c = (paint: colour-c, thickness: 0.6pt, dash: "dashed")
    let dashed-d = (paint: colour-d, thickness: 0.6pt, dash: "dashed")
    line(xp, ca, stroke: dashed-a)
    line(xp, cb, stroke: dashed-b)
    line(xp, cc, stroke: dashed-c)
    line(xp, cd, stroke: dashed-d)

    // 3. All domain centroids
    dot(ca, label: $c _ A$, direction: (-0.12, -0.08), align-to: "east",
        radius: 0.032, colour: colour-a, label-colour: colour-a)
    dot(cb, label: $c _ B$, direction: (0.08, 0.06), align-to: "west",
        radius: 0.032, colour: colour-b, label-colour: colour-b)
    dot(cc, label: $c _ C$, direction: (-0.10, 0.06), align-to: "east",
        radius: 0.032, colour: colour-c, label-colour: colour-c)
    dot(cd, label: $c _ D$, direction: (0.08, -0.08), align-to: "west",
        radius: 0.032, colour: colour-d, label-colour: colour-d)

    // 4. Test particle
    dot(xp, label: $x _ p$, direction: (0.08, 0.06), align-to: "west",
        radius: 0.028, colour: colour-xp, label-colour: colour-xp)

  })),
)
