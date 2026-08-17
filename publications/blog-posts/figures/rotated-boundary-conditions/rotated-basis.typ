// Rotated boundary conditions: where the basis changes, and where in the
// solver the rotation lives.
//
// Thesis: rotating the degrees of freedom leaves the discrete problem in a
// MIXED basis, but the obligation is CONTAINED -- it lives in the velocity
// block and its multigrid substructure, and the Schur/pressure machinery
// wrapping it never handles a rotated vector.
//
// House style follows publications/blog-posts/figures/arrays-sync-flow.typ
// (cetz 0.3.4, hex colours, helper-function pattern).

#import "@preview/cetz:0.3.4"

#set page(width: auto, height: auto, margin: 16pt)
#set text(size: 10pt)

#cetz.canvas({
  import cetz.draw: *

  // Colours
  let cart-fg = rgb("#4a7bf7")
  let cart-bg = rgb("#dce8fc")
  let rot-fg = rgb("#e57373")
  let rot-bg = rgb("#fce4ec")
  let plain-bg = rgb("#f2f2f0")
  let mg-bg = rgb("#fdf1f4")
  let schur-fg = rgb("#49a87c")   // the un-rotated half
  let schur-bg = rgb("#e8f4e8")
  let ink = rgb("#1a1a1a")
  let muted = rgb("#7a7a7a")
  let hair = rgb("#d0d0d0")

  // ======================================================================
  // LEFT: a free surface, meshed. Geometry (nodes, Delaunay triangles,
  // surface frames) comes from generate-rotated-basis.py via JSON -- the
  // skill's rule, and the reason every node is now in the triangulation.
  // An earlier version joined nodes by a distance threshold and left some out.
  // ======================================================================
  let data = json("rotated-basis-data.json")
  let vtx = data.vertices
  let at(i) = (vtx.at(i).at(0), vtx.at(i).at(1))
  let is-surface(i) = data.surface.contains(i)

  // triangles first, then the surface, then the nodes: painter's algorithm
  for tri in data.triangles {
    line(at(tri.at(0)), at(tri.at(1)), at(tri.at(2)), close: true,
      stroke: (paint: hair, thickness: 0.55pt))
  }

  line(..data.curve.map(q => (q.at(0), q.at(1))),
    stroke: (paint: ink, thickness: 1.2pt))

  // A rotated frame at every constrained node: n the outward surface normal,
  // t the tangent. Both follow the node, which is the whole point.
  for (k, f) in data.frames.enumerate() {
    let p = (f.p.at(0), f.p.at(1))
    let n = f.n
    let tg = f.t
    let ln = 0.72
    let lt = 0.46
    line(p, (p.at(0) + ln * n.at(0), p.at(1) + ln * n.at(1)),
      mark: (end: ">", scale: 0.4), stroke: (paint: rot-fg, thickness: 1.1pt))
    line(p, (p.at(0) + lt * tg.at(0), p.at(1) + lt * tg.at(1)),
      mark: (end: ">", scale: 0.4), stroke: (paint: rot-fg, thickness: 1.1pt))
    if k == 4 {
      content((p.at(0) + 1.20 * ln * n.at(0), p.at(1) + 1.20 * ln * n.at(1)),
        text(fill: rot-fg, size: 9pt, $n$))
      content((p.at(0) + 1.55 * lt * tg.at(0) + 0.24 * n.at(0),
               p.at(1) + 1.55 * lt * tg.at(1) + 0.24 * n.at(1)),
        text(fill: rot-fg, size: 9pt, $t$))
    }
  }

  // the Cartesian frame at one interior node -- identical at every other one,
  // which is what makes the surface the odd one out
  let ip = at(12)   // an interior node with room around it
  line(ip, (ip.at(0) + 0.66, ip.at(1)), mark: (end: ">", scale: 0.4),
    stroke: (paint: cart-fg, thickness: 1.1pt))
  line(ip, (ip.at(0), ip.at(1) + 0.66), mark: (end: ">", scale: 0.4),
    stroke: (paint: cart-fg, thickness: 1.1pt))
  content((ip.at(0) + 0.88, ip.at(1)), text(fill: cart-fg, size: 9pt, $x$))
  content((ip.at(0), ip.at(1) + 0.88), text(fill: cart-fg, size: 9pt, $y$))

  for i in range(vtx.len()) {
    if is-surface(i) {
      circle(at(i), radius: 0.13, fill: rot-fg,
        stroke: (paint: rot-fg, thickness: 1pt))
    } else {
      circle(at(i), radius: 0.11, fill: cart-bg,
        stroke: (paint: cart-fg, thickness: 1pt))
    }
  }

  content((0, 3.05), text(weight: "bold", size: 10pt, fill: ink,
    "A free surface has no preferred direction"))

  circle((-3.3, -3.55), radius: 0.13, fill: rot-fg, stroke: (paint: rot-fg))
  content((-3.05, -3.55), anchor: "west",
    text(size: 9pt, fill: ink, [surface node --- solve for $(v_n, v_t)$, hold $v_n$]))
  circle((-3.3, -4.1), radius: 0.11, fill: cart-bg,
    stroke: (paint: cart-fg, thickness: 1pt))
  content((-3.05, -4.1), anchor: "west",
    text(size: 9pt, fill: ink, [interior node --- solve for $(v_x, v_y)$]))

  line((4.35, -4.9), (4.35, 3.3), stroke: (paint: hair, thickness: 0.8pt))

  // ======================================================================
  // RIGHT: where the rotation lives. Two blocks ABUTTING, not nested: the
  // velocity solve is rotated, the Schur/pressure solve is not, and the
  // single un-rotation on the boundary between them feeds both the Schur
  // solve and everything outside.
  // ======================================================================
  let panel(tl, br, title, subtitle, bg, edge) = {
    import cetz.draw: *
    rect(tl, br, fill: bg, stroke: (paint: edge, thickness: 1pt), radius: 5pt)
    content((tl.at(0) + 0.30, tl.at(1) - 0.36), anchor: "west",
      text(weight: "bold", size: 9.5pt, fill: edge, title))
    if subtitle != none {
      // cetz content() lays out at natural width and spills over the rect.
      // Box it to the panel's own width so the text wraps inside the border.
      // The canvas default is 1cm per unit, so the arithmetic is direct.
      let tw = (br.at(0) - tl.at(0) - 0.60) * 1cm
      content((tl.at(0) + 0.30, tl.at(1) - 0.90), anchor: "north-west",
        box(width: tw, text(size: 8.5pt, fill: muted, subtitle)))
    }
  }

  content((10.3, 3.05), text(weight: "bold", size: 10pt, fill: ink,
    "Where the rotation lives"))

  // -- the rotated half ---------------------------------------------------
  panel((4.9, 1.65), (11.75, -3.70), [Velocity solve --- rotated],
    [$hat(A) = Q^T A Q$, #h(0.7em) $hat(b) = Q^T b$, #h(0.7em)
     $v_n$ held strongly at surface nodes], rot-bg, rot-fg)

  panel((5.35, -0.15), (11.50, -3.25), "Multigrid",
    [the transfers are the only further obligation], mg-bg, rot-fg)

  let mgrow(y, lhs, rhs) = {
    import cetz.draw: *
    content((5.65, y), anchor: "west", text(size: 9pt, fill: ink, lhs))
    content((8.05, y), anchor: "west", text(size: 9pt, fill: rot-fg, rhs))
  }
  mgrow(-1.85, [prolongation], [$P -> Q^T P$])
  mgrow(-2.40, [coarse operators], [inherit $Q$ via $R A P$])
  mgrow(-2.93, [coarse solve], [SVD (rigid rotations)])

  // -- the un-rotated half, abutting --------------------------------------
  panel((13.35, 1.65), (17.85, -1.35), [Fieldsplit / Schur solve],
    [pressure and constraints. Isotropic, and carrying no boundary
     condition of this kind, so it never sees a rotated vector.],
    schur-bg, schur-fg)

  // -- one un-rotation on the boundary, feeding both ----------------------
  // The junction sits in the gap BETWEEN the two blocks, low enough to clear
  // the velocity block's own subtitle -- one un-rotation, two consumers.
  let jx = 12.55
  let jy = -0.40
  line((11.75, jy), (jx, jy), stroke: (paint: rot-fg, thickness: 1pt))
  circle((jx, jy), radius: 0.075, fill: rot-fg, stroke: (paint: rot-fg))
  content((jx, jy + 0.45), text(size: 9pt, fill: rot-fg, $v = Q hat(v)$))

  // branch 1: into the Schur solve, which needs it un-rotated
  line((jx, jy), (13.29, jy), mark: (end: ">", scale: 0.4),
    stroke: (paint: rot-fg, thickness: 1pt))

  // branch 2: out to everything else
  line((jx, jy), (jx, -2.70), stroke: (paint: rot-fg, thickness: 1pt))
  line((jx, -2.70), (13.20, -2.70), mark: (end: ">", scale: 0.4),
    stroke: (paint: rot-fg, thickness: 1pt))
  content((13.33, -2.70), anchor: "west", text(size: 8.5pt, fill: rot-fg,
    [and to everything outside ---\ output, advection, the surface update]))

  // convention, stated -- a reader who assumes the transpose reads the
  // whole figure backwards
  line((4.75, -5.05), (18.1, -5.05), stroke: (paint: hair, thickness: 0.8pt))
  content((4.75, -5.55), anchor: "west", text(size: 9pt, fill: ink,
    [Convention: the columns of $Q$ are the nodal frame, so $hat(v) = Q^T v$,
     and $Q = I$ at every unconstrained node.]))
})
