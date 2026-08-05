// The grid hierarchy, vertical companion to stack-progression. Thesis:
// the static base is itself the finest level of a geometric (FMG)
// hierarchy; the adapt-on-top children EXTEND that hierarchy upward,
// one level per doubling of resolution — and the cut + split mesh sits
// on top of the stack but OUTSIDE the hierarchy: a cut is not a
// multigrid level.
#import "@preview/cetz:0.3.4"

#set page(width: auto, height: auto, margin: 12pt)
#set text(font: ("Noto Sans", "Helvetica", "Arial"), size: 5pt)

#let stack = json("stack-progression-data.json")
#let coarse = json("grid-hierarchy-data.json")

#let mesh-fill = rgb("#eef3fa")
#let mesh-fill-minus = rgb("#fce4ec")
#let mesh-fill-plus = rgb("#dce8fc")
#let mesh-stroke = rgb("#8899aa")
#let fault-col = rgb("#c62828")
#let arrow-col = rgb("#444444")
#let approx-col = rgb("#b57500")

#let PH = 1.0
#let VGAP = 0.62
#let panel-y(k) = k * (PH + VGAP)

#let draw-mesh(y0, p, fills: none) = {
  import cetz.draw: *
  let P(v) = (p.coords.at(v).at(0), p.coords.at(v).at(1) + y0)
  for (k, t) in p.tris.enumerate() {
    let fill = if fills == none { mesh-fill } else { fills.at(k) }
    line(P(t.at(0)), P(t.at(1)), P(t.at(2)), close: true,
         fill: fill, stroke: (paint: mesh-stroke, thickness: 0.25pt))
  }
}

#let draw-manifold(y0) = {
  import cetz.draw: *
  let a = stack.fault.at(0)
  let b = stack.fault.at(1)
  line((a.at(0), a.at(1) + y0), (b.at(0), b.at(1) + y0),
       stroke: (paint: fault-col, thickness: 0.9pt, dash: "dashed"))
}

#let rise-arrow(k, label, col: rgb("#444444")) = {
  import cetz.draw: *
  let y = panel-y(k) + PH + 0.08
  line((-0.35, y), (-0.35, y + VGAP - 0.16),
       stroke: (paint: col, thickness: 0.8pt), mark: (end: ">"))
  content((-0.52, y + (VGAP - 0.16) / 2), angle: 90deg,
          text(size: 4.5pt, fill: col, label))
}

#let caption(k, body) = {
  import cetz.draw: *
  content((2.18, panel-y(k) + PH / 2), anchor: "west",
          align(left, text(size: 4.5pt, body)))
}

#cetz.canvas(length: 1.5cm, {
  import cetz.draw: *

  // ---- the geometric hierarchy, bottom up --------------------------------
  draw-mesh(panel-y(0), coarse.coarse2)
  caption(0, [coarsest level\ (structured)])
  rise-arrow(0, [refine ×2, nested])

  draw-mesh(panel-y(1), coarse.coarse1)
  caption(1, [coarse level\ (structured)])
  rise-arrow(1, [refine ×2, nested])

  draw-mesh(panel-y(2), stack.panels.base)
  draw-manifold(panel-y(2))
  caption(2, [the STATIC base\ (finest NESTED level;\ fault = geometry only)])
  rise-arrow(2, [adapt, non-nested], col: approx-col)

  draw-mesh(panel-y(3), stack.panels.l1)
  draw-manifold(panel-y(3))
  caption(3, [adapted child, layer 1])
  rise-arrow(3, [adapt, non-nested], col: approx-col)

  draw-mesh(panel-y(4), stack.panels.l2)
  draw-manifold(panel-y(4))
  caption(4, [adapted child, layer 2\ (finest mesh)])

  // ---- the separator: everything above is NOT a level --------------------
  let ysep = panel-y(5) - VGAP / 2 + 0.06
  line((-0.7, ysep), (3.6, ysep),
       stroke: (paint: arrow-col, thickness: 0.7pt, dash: "dashed"))
  content((1.45, ysep + 0.11),
          text(size: 4.5pt, style: "italic")[a cut is not a multigrid level])
  {
    let y = panel-y(4) + PH + 0.08
    line((-0.35, y), (-0.35, panel-y(5) - 0.08),
         stroke: (paint: arrow-col, thickness: 0.8pt), mark: (end: ">"))
    content((-0.72, (y + panel-y(5)) / 2 - 0.04), angle: 90deg,
            text(size: 4.5pt)[cut + split])
  }

  // ---- the working mesh on top -------------------------------------------
  {
    let y0 = panel-y(5) + 0.12
    let fills = stack.cut_side.map(s => if s < 0 { mesh-fill-minus }
                                        else { mesh-fill-plus })
    draw-mesh(y0, stack.panels.cut, fills: fills)
    let a = stack.fault.at(0)
    let b = stack.fault.at(1)
    let dx = b.at(0) - a.at(0)
    let dy = b.at(1) - a.at(1)
    let ln = calc.sqrt(dx * dx + dy * dy)
    let nx = -dy / ln
    let ny = dx / ln
    let d = 0.022
    line((a.at(0), a.at(1) + y0), (b.at(0), b.at(1) + y0),
         stroke: (paint: white, thickness: 3pt))
    for s in (-1.0, 1.0) {
      line((a.at(0) + s * d * nx, a.at(1) + s * d * ny + y0),
           (b.at(0) + s * d * nx, b.at(1) + s * d * ny + y0),
           stroke: (paint: fault-col, thickness: 0.7pt))
    }
    for pt in (a, b) {
      circle((pt.at(0), pt.at(1) + y0), radius: 0.032, fill: white,
             stroke: (paint: black, thickness: 0.8pt))
    }
    content((2.18, y0 + PH / 2), anchor: "west",
            align(left, text(size: 4.5pt)[the working mesh:\ split, top of
              the stack,\ OUTSIDE the hierarchy]))
  }

  // ---- brackets: exact nesting below the base, approximate above ---------
  {
    let x = 3.72
    let y0 = panel-y(0)
    let y1 = panel-y(2) + PH
    line((x, y0), (x + 0.1, y0), (x + 0.1, y1), (x, y1),
         stroke: (paint: arrow-col, thickness: 0.7pt))
    content((x + 0.28, (y0 + y1) / 2), angle: 90deg,
            text(size: 4.5pt)[FMG hierarchy: structured refinement,
              EXACTLY nested — level transfers are lossless])
    let y2 = panel-y(3)
    let y3 = panel-y(4) + PH
    line((x, y2), (x + 0.1, y2), (x + 0.1, y3), (x, y3),
         stroke: (paint: approx-col, thickness: 0.7pt))
    content((x + 0.28, (y2 + y3) / 2), angle: 90deg,
            text(size: 4.5pt, fill: approx-col)[adapted levels:
              non-nested — transfers are APPROXIMATE])
  }
})
