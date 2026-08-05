// Split-node anatomy, 2-D. Thesis: splitting duplicates ONLY the interior
// of the fault chain — cells on the Minus side rewire to replica vertices,
// the tips stay shared, and the two copies are geometrically coincident
// (the lower panel is exploded for display only).
#import "@preview/cetz:0.3.4"

#set page(width: auto, height: auto, margin: 12pt)
#set text(size: 10pt)

#let data = json("split-anatomy-data.json")

#let mesh-fill-plus = rgb("#dce8fc")
#let mesh-fill-minus = rgb("#fce4ec")
#let mesh-stroke = rgb("#8899aa")
#let fault-col = rgb("#c62828")
#let replica-col = rgb("#c62828")
#let tip-col = rgb("#1a1a1a")

#let panel(origin-y, coords, tris, exploded: false) = {
  import cetz.draw: *
  let P(v) = (coords.at(v).at(0), coords.at(v).at(1) + origin-y)
  // cells, painter's algorithm: fills first, then edges, then the fault
  for (k, t) in tris.enumerate() {
    let fill = if data.side.at(k) < 0 { mesh-fill-minus } else { mesh-fill-plus }
    line(P(t.at(0)), P(t.at(1)), P(t.at(2)), close: true,
         fill: fill, stroke: (paint: mesh-stroke, thickness: 0.3pt))
  }
  // fault facets: both copies carry the label
  if exploded {
    // Plus copy: original chain vertices
    for i in range(data.chain.len() - 1) {
      line(P(data.chain.at(i)), P(data.chain.at(i + 1)),
           stroke: (paint: fault-col, thickness: 1.0pt))
    }
    // Minus copy: replicas where they exist, tips where shared
    let lower(v) = if str(v) in data.replicas { data.replicas.at(str(v)) } else { v }
    for i in range(data.chain.len() - 1) {
      line(P(lower(data.chain.at(i))), P(lower(data.chain.at(i + 1))),
           stroke: (paint: fault-col, thickness: 1.0pt, dash: "densely-dashed"))
    }
  } else {
    for i in range(data.chain.len() - 1) {
      line(P(data.chain.at(i)), P(data.chain.at(i + 1)),
           stroke: (paint: fault-col, thickness: 0.9pt))
    }
  }
  // vertices: interior chain (filled red), replicas (open red), tips (black ring)
  for v in data.interior {
    circle(P(v), radius: 0.045, fill: fault-col, stroke: none)
  }
  if exploded {
    for (orig, rep) in data.replicas {
      circle(P(rep), radius: 0.045, fill: white,
             stroke: (paint: replica-col, thickness: 0.9pt))
    }
  }
  for v in data.tips {
    circle(P(v), radius: 0.055, fill: white,
           stroke: (paint: tip-col, thickness: 1.0pt))
    circle(P(v), radius: 0.02, fill: tip-col, stroke: none)
  }
}

#cetz.canvas(length: 1.55cm, {
  import cetz.draw: *

  // ---- panel (a): the conforming labelled chain --------------------------
  panel(2.1, data.coords, data.tris)
  content((-0.35, 3.1), [(a)])
  content((1.5, 1.78), align(center)[conforming chain — facets shared,
    every FE space continuous])
  content((3.35, 2.6), text(fill: fault-col)[$Gamma$])
  content((0.5, 2.36), text(size: 8pt)[tip])
  content((2.5, 2.36), text(size: 8pt)[tip])

  // ---- panel (b): after the split (exploded for display) -----------------
  panel(0.0, data.exploded, data.moved_tris, exploded: true)
  content((-0.35, 1.0), [(b)])
  content((1.5, -0.62), align(center)[split — interior vertices duplicated,
    Minus cells rewired (exploded: the copies are coincident)])
  content((3.42, 0.62), text(fill: fault-col, size: 9pt)[$Gamma^+$])
  content((3.42, 0.13), text(fill: fault-col, size: 9pt)[$Gamma^-$])
  // annotate one replica pair
  line((1.5, 0.44), (1.75, 0.85), stroke: (paint: rgb("#555555"),
       thickness: 0.3pt))
  content((2.15, 0.95), text(size: 8pt)[$v^+$ (original)])
  line((1.5, 0.24), (1.78, -0.12), stroke: (paint: rgb("#555555"),
       thickness: 0.3pt))
  content((2.2, -0.2), text(size: 8pt)[$v^-$ (replica)])
})
