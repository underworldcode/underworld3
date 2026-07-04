#import "@preview/cetz:0.3.4"

#set page(width: auto, height: auto, margin: 8pt)
#set text(size: 9pt)

// ── Data ─────────────────────────────────────────────────────────────
#let mesh = json("boundary-demo-data.json")

#let pt(i) = {
  let v = mesh.vertices.at(i)
  (v.at(0), v.at(1))
}

#let VIEW = mesh.view

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

// ── Colours ──────────────────────────────────────────────────────────
// Domain A perspective
#let a-inside   = rgb(40, 80, 160, 70)     // strong blue
#let a-boundary = rgb(120, 150, 200, 45)   // light blue-grey
#let a-outside  = rgb(240, 240, 240, 30)   // near white

// Domain B perspective
#let b-inside   = rgb(180, 50, 50, 70)     // strong red
#let b-boundary = rgb(200, 140, 140, 45)   // light red-grey
#let b-outside  = rgb(240, 240, 240, 30)   // near white

#let stroke-mesh = 0.25pt + rgb("#c0c0c0")
#let boundary-stroke = 1.0pt + rgb("#444444")

// Control point colours
#let cp-inside-colour = black
#let cp-outside-colour = rgb("#b0b0b0")  // grey
#let cp-outside-stroke = 0.3pt + rgb("#555555")  // thin dark outline

#let colour-a   = rgb("#2563eb")           // blue
#let colour-b   = rgb("#dc2626")           // red
#let colour-xp  = rgb("#7c3aed")           // violet

#let xa = (mesh.test_point_a.at(0), mesh.test_point_a.at(1))
#let xb = (mesh.test_point_b.at(0), mesh.test_point_b.at(1))

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

// ── Panel drawing function ───────────────────────────────────────────
#let draw-panel(zones, inside-col, boundary-col, outside-col,
                focus-domain, panel-label, panel-colour) = {
  import cetz.draw: *

  // 1. Draw triangles with zone colouring
  for (idx, tri) in mesh.triangles.enumerate() {
    let c = tri-centroid(tri)
    if in-view(c) {
      let zone = zones.at(str(idx), default: "outside")
      let fill-col = if zone == "inside" { inside-col }
                     else if zone == "boundary" { boundary-col }
                     else { outside-col }
      line(
        pt(tri.at(0)), pt(tri.at(1)), pt(tri.at(2)),
        close: true,
        fill: fill-col,
        stroke: stroke-mesh,
      )
    }
  }

  // 2. Domain boundary edges — heavy if this domain's boundary, thinner otherwise
  let secondary-stroke = 0.5pt + rgb("#555555")
  for edge in mesh.boundary_edges {
    let p0 = pt(edge.at(0))
    let p1 = pt(edge.at(1))
    if in-view(p0) or in-view(p1) {
      let d0 = edge.at(2)
      let d1 = edge.at(3)
      let involves-focus = d0 == focus-domain or d1 == focus-domain
      line(p0, p1, stroke: if involves-focus { boundary-stroke } else { secondary-stroke })
    }
  }

  // 3. Boundary control point pairs — only on edges involving focus domain
  for cp in mesh.boundary_control_points {
    if cp.domain_plus != focus-domain and cp.domain_minus != focus-domain {
      continue
    }
    let p-plus = (cp.pos_plus.at(0), cp.pos_plus.at(1))
    let p-minus = (cp.pos_minus.at(0), cp.pos_minus.at(1))
    if in-view(p-plus) or in-view(p-minus) {
      let plus-is-inside = cp.domain_plus == focus-domain
      let minus-is-inside = cp.domain_minus == focus-domain

      if plus-is-inside {
        circle(p-plus, radius: 0.014, fill: cp-inside-colour, stroke: none)
      } else {
        circle(p-plus, radius: 0.014, fill: cp-outside-colour, stroke: cp-outside-stroke)
      }
      if minus-is-inside {
        circle(p-minus, radius: 0.014, fill: cp-inside-colour, stroke: none)
      } else {
        circle(p-minus, radius: 0.014, fill: cp-outside-colour, stroke: cp-outside-stroke)
      }
    }
  }

  // 4. Test particles and their nearest control point connections
  let test-points = (
    (xa, colour-a, $x _ a$, (-0.10, -0.06), "east"),
    (xb, colour-b, $x _ b$, (0.08, 0.06), "west"),
    ((mesh.test_point.at(0), mesh.test_point.at(1)),
     colour-xp, $x _ p$, (0.08, 0.06), "west"),
  )

  // For each test point, find the nearest control point and draw a connector
  for tp in test-points {
    let pos = tp.at(0)
    let col = tp.at(1)

    // Search boundary control points on THIS domain's boundary
    // (only edges where one side belongs to focus-domain)
    let best-dist = 1e9
    let best-pt = pos
    let best-is-inside = true
    for cp in mesh.boundary_control_points {
      // Skip edges that don't involve this domain
      if cp.domain_plus != focus-domain and cp.domain_minus != focus-domain {
        continue
      }

      let p-plus = (cp.pos_plus.at(0), cp.pos_plus.at(1))
      let p-minus = (cp.pos_minus.at(0), cp.pos_minus.at(1))

      let d-plus = calc.sqrt(
        calc.pow(pos.at(0) - p-plus.at(0), 2) +
        calc.pow(pos.at(1) - p-plus.at(1), 2))
      let d-minus = calc.sqrt(
        calc.pow(pos.at(0) - p-minus.at(0), 2) +
        calc.pow(pos.at(1) - p-minus.at(1), 2))

      if d-plus < best-dist {
        best-dist = d-plus
        best-pt = p-plus
        best-is-inside = cp.domain_plus == focus-domain
      }
      if d-minus < best-dist {
        best-dist = d-minus
        best-pt = p-minus
        best-is-inside = cp.domain_minus == focus-domain
      }
    }

    // Draw thin dashed connector
    let connector-col = if best-is-inside { cp-inside-colour } else { luma(120) }
    line(pos, best-pt,
         stroke: (paint: connector-col, thickness: 0.5pt,
                  dash: (array: (0.5pt, 1.5pt), phase: 0pt)))

    // Draw the test point dot on top
    dot(pos, label: tp.at(2), direction: tp.at(3), align-to: tp.at(4),
        radius: 0.024, colour: col, label-colour: col)
  }

  // 5. Panel label — bottom right
  content((VIEW - 0.15, -VIEW + 0.15),
    text(fill: panel-colour, size: 11pt, weight: "bold", panel-label),
    anchor: "south-east")
}

// ── Two-panel layout ─────────────────────────────────────────────────
#let panel-width = 8cm
#let panel-height = 8cm
#let gap = 0.4cm

#box(
  width: 2 * panel-width + gap,
  height: panel-height,
  {
    // Left panel: Domain A's perspective
    box(
      clip: true,
      width: panel-width,
      height: panel-height,
      stroke: 0.5pt + luma(50%),
      align(center + horizon, cetz.canvas(length: 2cm, {
        draw-panel(
          mesh.zones_a, a-inside, a-boundary, a-outside,
          "A", "Domain A's view", rgb("#2563eb"),
        )
      })),
    )
    h(gap)
    // Right panel: Domain B's perspective
    box(
      clip: true,
      width: panel-width,
      height: panel-height,
      stroke: 0.5pt + luma(50%),
      align(center + horizon, cetz.canvas(length: 2cm, {
        draw-panel(
          mesh.zones_b, b-inside, b-boundary, b-outside,
          "B", "Domain B's view", rgb("#dc2626"),
        )
      })),
    )
  }
)
