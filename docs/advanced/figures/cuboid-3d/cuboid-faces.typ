#import "@preview/cetz:0.3.4"

// Explicit page dims + centred canvas gives reliable padding for the
// external labels, which can sit noticeably outside the cube silhouette.
#set page(width: 14cm, height: 10cm, margin: 6pt)
#set text(size: 10pt)

#let data = json("cuboid-3d-data.json")

// ── Colours ───────────────────────────────────────────────────────────
// Each opposite-face pair is coloured by its coordinate axis (RGB-axes
// convention):  x-pair (Left/Right) = rust,  y-pair (Front/Back) = green,
// z-pair (Upper/Lower) = blue.  Hidden faces use the same hue at lower
// alpha so the pairing is legible regardless of visibility.
#let face-fill(name, visible) = {
  let (r, g, b) = if name == "Upper" or name == "Lower" {
    (85, 130, 195)        // blue — z-axis pair
  } else if name == "Front" or name == "Back" {
    (90, 160, 110)        // green — y-axis pair
  } else {
    (200, 110, 85)        // rust — x-axis pair
  }
  let a = if visible { 140 } else { 30 }
  rgb(r, g, b, a)
}
#let visible-edge  = 0.9pt + black
#let hidden-edge   = (paint: rgb("#808080"), thickness: 0.7pt, dash: "dashed")
#let arrow-colour  = rgb("#1f3a6b")
#let arrow-stroke  = 0.9pt + arrow-colour

#let v(i) = (data.vertices_2d.at(i).at(0), data.vertices_2d.at(i).at(1))

// ── Face label layout ─────────────────────────────────────────────────
// Two vertical columns at x = ±2, three labels each.  Each label is
// paired with a face whose centroid lies on the same side of the
// figure, which prevents line crossings.  Lines are bare (no arrow-
// heads) — labels, not symmetry axes.
#let label-overrides = (
  Upper: (pos: (-2.0, +1.5), anchor: "east"),
  Left:  (pos: (-2.0,  0.0), anchor: "east"),
  Back:  (pos: (-2.0, -1.5), anchor: "east"),
  Front: (pos: (+2.0, +1.5), anchor: "west"),
  Right: (pos: (+2.0,  0.0), anchor: "west"),
  Lower: (pos: (+2.0, -1.5), anchor: "west"),
)

// ── Axis triad at V6 (the cuboid corner nearest the viewer) ──────────
// The cube projection itself is unchanged (so the face labels Left/Right
// stay where they were).  Only the triad's +y arrow is flipped from the
// projected-y direction so the triad reads as right-handed — pointing
// the +y axis away from the viewer instead of toward.
#let TRIAD-LEN = 0.45
#let AXIS-X-2D = (0.86603, -0.5)          // projected +x direction
#let AXIS-Y-2D = (0.86603,  0.5)          // triad-only: flipped from (-0.866, -0.5)
#let AXIS-Z-2D = (0.0,      1.0)          // projected +z direction

#align(center + horizon, cetz.canvas(length: 1.5cm, {
  import cetz.draw: *

  // Split back-to-front order into hidden and visible groups, preserving
  // each group's internal painter-order for correct translucency stacking.
  let hidden-names  = data.faces_back_to_front.filter(
    n => not data.faces.at(n).visible)
  let visible-names = data.faces_back_to_front.filter(
    n => data.faces.at(n).visible)

  let draw-face(name) = {
    let face = data.faces.at(name)
    let idxs = face.vertex_indices
    line(
      v(idxs.at(0)), v(idxs.at(1)),
      v(idxs.at(2)), v(idxs.at(3)),
      close: true, fill: face-fill(name, face.visible), stroke: none,
    )
  }

  let draw-face-line(name) = {
    // Bare line from the (column-aligned) label position to the face
    // centroid — no arrowhead.  Hidden-face lines are drawn before the
    // visible face fills, so their inner segments fade behind the
    // translucent fronts.
    let face = data.faces.at(name)
    let lbl-pos = label-overrides.at(name).pos
    let tip = (face.arrow_to.at(0), face.arrow_to.at(1))
    line(lbl-pos, tip, stroke: arrow-stroke)
  }

  // Small 3D "badge" at the end of each label line: a shrunken copy of
  // the face's projected shape, sharing its axis-pair colour.
  let draw-face-marker(name) = {
    let face = data.faces.at(name)
    let idxs = face.vertex_indices
    let c = (face.arrow_to.at(0), face.arrow_to.at(1))
    let shrink-by = 0.18
    let shrunk(idx) = {
      let vi = v(idx)
      (c.at(0) + (vi.at(0) - c.at(0)) * shrink-by,
       c.at(1) + (vi.at(1) - c.at(1)) * shrink-by)
    }
    let fill-col = {
      let (r, g, b) = if name == "Upper" or name == "Lower" {
        (85, 130, 195)
      } else if name == "Front" or name == "Back" {
        (90, 160, 110)
      } else {
        (200, 110, 85)
      }
      rgb(r, g, b, 230)
    }
    line(shrunk(idxs.at(0)), shrunk(idxs.at(1)),
         shrunk(idxs.at(2)), shrunk(idxs.at(3)),
         close: true, fill: fill-col, stroke: 0.7pt + arrow-colour)
  }

  // 1. Hidden face fills (farthest-first painter order within group).
  for name in hidden-names { draw-face(name) }

  // 2. Hidden-face label lines + markers — drawn BEFORE the visible
  //    face fills so their inner segments (and the small badges that
  //    sit at the face centroid) fade behind the translucent fronts.
  for name in hidden-names { draw-face-line(name); draw-face-marker(name) }

  // 3. Hidden edges (dashed) — same "behind the translucent front" idea.
  for edge in data.edges {
    if not edge.visible {
      let v0 = v(edge.vertices.at(0))
      let v1 = v(edge.vertices.at(1))
      line(v0, v1, stroke: hidden-edge)
    }
  }

  // 4. Visible face fills on top — translucently cover the hidden arrows
  //    and edges that lie inside the cube silhouette.
  for name in visible-names { draw-face(name) }

  // 5. Visible-face label lines + markers — fully in front of everything.
  for name in visible-names { draw-face-line(name); draw-face-marker(name) }

  // 6. Visible edges — solid black on top of everything structural.
  for edge in data.edges {
    if edge.visible {
      let v0 = v(edge.vertices.at(0))
      let v1 = v(edge.vertices.at(1))
      line(v0, v1, stroke: visible-edge)
    }
  }

  // 7. Axis triad at V6 — the corner nearest the viewer.  Short arrows
  //    in the projected +x, +y, +z directions, with italic-math labels.
  let triad-origin = v(6)
  let triad-stroke = 0.9pt + arrow-colour
  let triad-mark = (end: ">", fill: arrow-colour)
  let triad-arrow(dir-2d, label-offset-factor, lbl, lbl-anchor) = {
    let tip = (
      triad-origin.at(0) + TRIAD-LEN * dir-2d.at(0),
      triad-origin.at(1) + TRIAD-LEN * dir-2d.at(1),
    )
    line(triad-origin, tip, stroke: triad-stroke, mark: triad-mark)
    let lbl-pos = (
      triad-origin.at(0) + label-offset-factor * TRIAD-LEN * dir-2d.at(0),
      triad-origin.at(1) + label-offset-factor * TRIAD-LEN * dir-2d.at(1),
    )
    content(lbl-pos,
            text(fill: arrow-colour, size: 10pt, style: "italic", lbl),
            anchor: lbl-anchor)
  }
  triad-arrow(AXIS-X-2D, 1.30, $x$, "north-west")
  triad-arrow(AXIS-Y-2D, 1.30, $y$, "north-east")
  triad-arrow(AXIS-Z-2D, 1.22, $z$, "south")

  // 8. Face labels last — always on top; they live outside the cube.
  for (name, override) in label-overrides.pairs() {
    content(override.pos,
            text(fill: black, size: 10pt, name),
            anchor: override.anchor)
  }
}))
