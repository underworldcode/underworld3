// The stack-on progression. Thesis: the base mesh is STATIC; refinement
// layers stack toward the fault manifold; the cut and the split happen
// only at the top of the stack — and when the fault moves, the whole top
// is re-derived from the same base (nothing is cumulative).
#import "@preview/cetz:0.3.4"

#set page(width: auto, height: auto, margin: 12pt)
#set text(font: ("Noto Sans", "Helvetica", "Arial"), size: 5pt)

#let data = json("stack-progression-data.json")

#let mesh-fill = rgb("#eef3fa")
#let mesh-fill-minus = rgb("#fce4ec")
#let mesh-fill-plus = rgb("#dce8fc")
#let mesh-stroke = rgb("#8899aa")
#let fault-col = rgb("#c62828")
#let arrow-col = rgb("#444444")

#let GAP = 0.72
#let PW = 2.0

#let panel-x(k) = k * (PW + GAP)

#let draw-mesh(x0, p, fills: none) = {
  import cetz.draw: *
  let P(v) = (p.coords.at(v).at(0) + x0, p.coords.at(v).at(1))
  for (k, t) in p.tris.enumerate() {
    let fill = if fills == none { mesh-fill } else { fills.at(k) }
    line(P(t.at(0)), P(t.at(1)), P(t.at(2)), close: true,
         fill: fill, stroke: (paint: mesh-stroke, thickness: 0.25pt))
  }
}

#let draw-manifold(x0, dash: "dashed") = {
  import cetz.draw: *
  let a = data.fault.at(0)
  let b = data.fault.at(1)
  line((a.at(0) + x0, a.at(1)), (b.at(0) + x0, b.at(1)),
       stroke: (paint: fault-col, thickness: 0.9pt, dash: dash))
}

#let stage-arrow(k, label) = {
  import cetz.draw: *
  let x = panel-x(k) + PW + 0.06
  line((x, 0.6), (x + GAP - 0.12, 0.6),
       stroke: (paint: arrow-col, thickness: 0.8pt), mark: (end: ">"))
  content((x + GAP / 2 - 0.06, 0.86), text(size: 4pt, label))
}

#cetz.canvas(length: 1.5cm, {
  import cetz.draw: *

  // ---- (a) static base + the fault MANIFOLD (geometry, not mesh) --------
  draw-mesh(panel-x(0), data.panels.base)
  draw-manifold(panel-x(0))
  content((panel-x(0) + PW / 2, -0.32),
          align(center, text(size: 4.5pt)[(a) static base +\ fault manifold]))

  // ---- (b) layer 1 --------------------------------------------------------
  draw-mesh(panel-x(1), data.panels.l1)
  draw-manifold(panel-x(1))
  content((panel-x(1) + PW / 2, -0.32),
          align(center, text(size: 4.5pt)[(b) refined child,\ layer 1]))

  // ---- (c) layer 2 --------------------------------------------------------
  draw-mesh(panel-x(2), data.panels.l2)
  draw-manifold(panel-x(2))
  content((panel-x(2) + PW / 2, -0.32),
          align(center, text(size: 4.5pt)[(c) refined child,\ layer 2]))

  // ---- (d) the cut: a conforming labelled chain ---------------------------
  draw-mesh(panel-x(3), data.panels.cut)
  {
    let p = data.panels.cut
    let P(v) = (p.coords.at(v).at(0) + panel-x(3), p.coords.at(v).at(1))
    for i in range(data.chain.len() - 1) {
      line(P(data.chain.at(i)), P(data.chain.at(i + 1)),
           stroke: (paint: fault-col, thickness: 1.0pt))
    }
  }
  content((panel-x(3) + PW / 2, -0.32),
          align(center, text(size: 4.5pt)[(d) cut: conforming\ facet chain]))

  // ---- (e) the split: the double-line symbol ------------------------------
  // The mesh stays in its true (unexploded) geometry — the copies ARE
  // coincident — and the doubled surface is drawn as two thin parallel
  // lines with white between, offset along the fault normal.
  {
    let x0 = panel-x(4)
    let fills = data.cut_side.map(s => if s < 0 { mesh-fill-minus }
                                       else { mesh-fill-plus })
    draw-mesh(x0, data.panels.cut, fills: fills)
    let a = data.fault.at(0)
    let b = data.fault.at(1)
    let dx = b.at(0) - a.at(0)
    let dy = b.at(1) - a.at(1)
    let ln = calc.sqrt(dx * dx + dy * dy)
    let nx = -dy / ln
    let ny = dx / ln
    let d = 0.022
    line((a.at(0) + x0, a.at(1)), (b.at(0) + x0, b.at(1)),
         stroke: (paint: white, thickness: 3pt))
    for s in (-1.0, 1.0) {
      line((a.at(0) + s * d * nx + x0, a.at(1) + s * d * ny),
           (b.at(0) + s * d * nx + x0, b.at(1) + s * d * ny),
           stroke: (paint: fault-col, thickness: 0.7pt))
    }
    for pt in (a, b) {
      circle((pt.at(0) + x0, pt.at(1)), radius: 0.032, fill: white,
             stroke: (paint: black, thickness: 0.8pt))
    }
  }
  content((panel-x(4) + PW / 2, -0.32),
          align(center, text(size: 4.5pt)[(e) split: doubled surface,\
            coincident sides]))

  // arrows drawn LAST — panel fills would otherwise cover the labels
  stage-arrow(0, [adapt])
  stage-arrow(1, [adapt])
  stage-arrow(2, [pull + cut])
  stage-arrow(3, [split])

  // ---- the non-cumulative loop --------------------------------------------
  let y = -0.85
  line((panel-x(4) + PW / 2, y + 0.12), (panel-x(4) + PW / 2, y),
       (panel-x(0) + PW / 2, y), (panel-x(0) + PW / 2, y + 0.12),
       stroke: (paint: arrow-col, thickness: 0.7pt, dash: "dashed"),
       mark: (end: ">"))
  content(((panel-x(0) + panel-x(4) + PW) / 2, y - 0.28),
          text(size: 4.5pt)[the fault moves $arrow.r$ re-derive the whole
            top from the SAME base — nothing is cumulative, the coarse
            stack never carries the fault])
})
