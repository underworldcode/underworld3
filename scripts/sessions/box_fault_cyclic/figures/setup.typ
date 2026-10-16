// Model set-up: a surface-breaking strike-slip fault in a 3-D viscoelastic
// block under a harmonic end load (the box toy model of the tidal cycle).
// House style: cetz-figures skill. Compile: typst compile --format png --ppi 220 setup.typ setup.png
#import "@preview/cetz:0.3.4"

#set page(width: auto, height: auto, margin: 10pt)
#set text(size: 9pt)

// ── Colours (hex, named by role) ──────────────────────────────────────
#let edge-colour   = rgb("#1f3a6b")          // navy — block edges
#let hidden-colour = rgb("#9aa5b8")          // grey-blue — hidden edges
#let face-fill     = rgb(31, 58, 107, 14)    // faint navy wash
#let fixed-fill    = rgb(31, 58, 107, 70)    // darker wash — the fixed face
#let fault-colour  = rgb("#c2410c")          // rust — the fault
#let fault-fill    = rgb(194, 65, 12, 60)
#let later-colour  = rgb("#7c3aed")          // violet — faults added later
#let later-fill    = rgb(124, 58, 237, 28)
#let load-colour   = rgb("#059669")          // emerald — applied tractions
#let slip-colour   = rgb("#1f3a6b")
#let shear-colour  = rgb("#059669")
#let normal-colour = rgb("#c2410c")
#let grey          = rgb("#6b7280")

// ── Oblique (cabinet) projection: x right, y receding, z up ───────────
#let proj(p) = (p.at(0) + 0.42 * p.at(1), p.at(2) + 0.26 * p.at(1))
#let P(x, y, z) = proj((x, y, z))

// block (model units) and fault: vertical, normal y, strike x, breaking the surface
#let Lx = 3.2
#let Ly = 1.8
#let H  = 1.0
#let yf = 0.85
#let xa = 1.05
#let xb = 2.35
#let w  = 0.6
// a later fault, drawn faintly
#let yf2 = 1.35
#let xa2 = 1.6
#let xb2 = 2.7

#let seg(a, b, col, th: 0.8pt, dash: none) = {
  import cetz.draw: *
  line(a, b, stroke: (paint: col, thickness: th, dash: dash))
}
#let arrow(a, b, col, th: 1.0pt) = {
  import cetz.draw: *
  line(a, b, stroke: (paint: col, thickness: th), mark: (end: ">", fill: col, scale: 0.6))
}

#let panel-a = cetz.canvas(length: 1.9cm, {
  import cetz.draw: *

  // 1. hidden edges
  seg(P(0, Ly, 0), P(Lx, Ly, 0), hidden-colour, dash: "dashed")
  seg(P(0, Ly, 0), P(0, Ly, H), hidden-colour, dash: "dashed")
  seg(P(0, 0, 0), P(0, Ly, 0), hidden-colour, dash: "dashed")

  // 2. the later fault (faint), then the fault — both inside the block
  line(P(xa2, yf2, H), P(xb2, yf2, H), P(xb2, yf2, H - w), P(xa2, yf2, H - w), close: true,
       fill: later-fill, stroke: (paint: later-colour, thickness: 0.7pt, dash: "dotted"))
  line(P(xa, yf, H), P(xb, yf, H), P(xb, yf, H - w), P(xa, yf, H - w), close: true,
       fill: fault-fill, stroke: (paint: fault-colour, thickness: 0.9pt, dash: "dashed"))

  // 3. visible faces: top, front (y = 0), right (x = Lx); fixed face at x = 0 shows through
  line(P(0, 0, H), P(Lx, 0, H), P(Lx, Ly, H), P(0, Ly, H), close: true,
       fill: face-fill, stroke: (paint: edge-colour, thickness: 0.9pt))
  line(P(0, 0, 0), P(Lx, 0, 0), P(Lx, 0, H), P(0, 0, H), close: true,
       fill: face-fill, stroke: (paint: edge-colour, thickness: 0.9pt))
  line(P(Lx, 0, 0), P(Lx, Ly, 0), P(Lx, Ly, H), P(Lx, 0, H), close: true,
       fill: face-fill, stroke: (paint: edge-colour, thickness: 0.9pt))
  // the fixed face x = 0, hatched by short strokes along its visible edges
  for k in range(0, 7) {
    let z = H * k / 6
    seg(P(0, 0, z), P(0, 0.22, z + 0.05), edge-colour, th: 0.6pt)
  }
  content(P(0, 0.05, 0.5 * H), text(fill: edge-colour, size: 8pt)[fixed\ $bold(u) = 0$], anchor: "east", padding: 4pt)

  // surface traces (solid) over the top wash: later fault faint, the fault strong
  seg(P(xa2, yf2, H), P(xb2, yf2, H), later-colour, th: 1.0pt, dash: "dotted")
  seg(P(xa, yf, H), P(xb, yf, H), fault-colour, th: 1.8pt)

  // 4. slip sense on the surface (right-lateral in the first half cycle)
  let xm = 0.5 * (xa + xb)
  arrow(P(xm - 0.25, yf - 0.17, H), P(xm + 0.25, yf - 0.17, H), slip-colour)
  arrow(P(xm + 0.25, yf + 0.17, H), P(xm - 0.25, yf + 0.17, H), slip-colour)

  // 5. fault dimensions and label
  seg(P(xa, yf, H - w - 0.08), P(xb, yf, H - w - 0.08), grey, th: 0.5pt)
  content(P(xm, yf, H - w - 0.08), text(fill: grey, $ell$), anchor: "north", padding: 2pt)
  seg(P(xb + 0.08, yf, H), P(xb + 0.08, yf, H - w), grey, th: 0.5pt)
  content(P(xb + 0.08, yf, H - 0.5 * w), text(fill: grey, $w$), anchor: "west", padding: 2pt)
  content(P(xm, yf, H - 0.55 * w),
          text(fill: fault-colour, size: 8pt)[strike-slip fault\ Coulomb $mu$, $T_b$],
          anchor: "center")
  content(P(xb2 + 0.05, yf2, H - 0.3 * w), text(fill: later-colour, size: 7.5pt)[further faults\ (later)], anchor: "west", padding: 2pt)

  // 6. the harmonic end load on x = Lx: normal and horizontal shear tractions
  arrow(P(Lx, 0.5 * Ly, 0.55 * H), P(Lx + 0.5, 0.5 * Ly, 0.55 * H), load-colour, th: 1.2pt)
  content(P(Lx + 0.55, 0.5 * Ly, 0.55 * H), text(fill: load-colour, $sigma_(x x)(t)$), anchor: "west")
  arrow(P(Lx, 0.25 * Ly, 0.18 * H), P(Lx, 0.70 * Ly, 0.18 * H), load-colour, th: 1.2pt)
  content(P(Lx, 0.72 * Ly, 0.18 * H), text(fill: load-colour, $sigma_(x y)(t)$), anchor: "north-west", padding: 1pt)

  // 7. axes
  let o = P(-0.55, 0, -0.35)
  arrow(o, proj((-0.55 + 0.45, 0, -0.35)), grey)
  content(proj((-0.55 + 0.5, 0, -0.35)), text(fill: grey, $hat(x)$), anchor: "west")
  arrow(o, proj((-0.55, 0.6, -0.35)), grey)
  content(proj((-0.55, 0.7, -0.35)), text(fill: grey, $hat(y)$), anchor: "south-west")
  arrow(o, proj((-0.55, 0, -0.35 + 0.45)), grey)
  content(proj((-0.55, 0, -0.35 + 0.5)), text(fill: grey, $hat(z)$), anchor: "south")

  // 8. boundary annotations
  content(P(0.5 * Lx, Ly, H), text(fill: edge-colour, size: 8pt)[free surface], anchor: "south", padding: 2pt)
  content(P(0.5 * Lx, 0, 0), text(fill: edge-colour, size: 8pt)[base: free-slip], anchor: "north", padding: 3pt)
  content(P(0.30 * Lx, 0, 0.32 * H), text(fill: edge-colour, size: 8pt)[$y$ faces: traction-free\ (or $u_x = u_z = 0$)], anchor: "center")
  content(P(0.74 * Lx, 0, 0), text(fill: edge-colour, size: 8pt)[Maxwell block, Wallace number $"Wa" = T_"orbit" \/ T_M$], anchor: "north-west", padding: 3pt)
})

// ── Panel (b): the load cycle on the driven face ──────────────────────
#let panel-b = cetz.canvas(length: 1.0cm, {
  import cetz.draw: *
  let W = 5.0
  let A = 1.1
  let phase = 80.0          // deg: the normal traction leads the shear (placeholder: stripe value)
  let n = 120
  seg((0, -A - 0.3), (0, A + 0.3), grey, th: 0.6pt)
  arrow((0, 0), (W + 0.3, 0), grey, th: 0.6pt)
  content((W + 0.35, 0), text(fill: grey, $t \/ T_"orbit"$), anchor: "west")
  for k in range(0, 5) {
    let x = W * k / 4
    seg((x, -0.06), (x, 0.06), grey, th: 0.6pt)
  }
  content((W, -0.1), text(fill: grey, size: 8pt)[1], anchor: "north")
  let pts-s = range(0, n + 1).map(i => (W * i / n, A * calc.sin(2 * calc.pi * i / n)))
  let pts-n = range(0, n + 1).map(i => (W * i / n, A * calc.sin(2 * calc.pi * i / n + phase * calc.pi / 180)))
  line(..pts-s, stroke: (paint: shear-colour, thickness: 1.2pt))
  line(..pts-n, stroke: (paint: normal-colour, thickness: 1.2pt, dash: "dashed"))
  seg((W + 0.5, A + 0.05), (W + 1.0, A + 0.05), shear-colour, th: 1.2pt)
  content((W + 1.1, A + 0.05), text(fill: shear-colour, size: 8pt)[$sigma_(x y)$ shear, $S sin(omega t)$], anchor: "west")
  seg((W + 0.5, A - 0.35), (W + 1.0, A - 0.35), normal-colour, th: 1.2pt, dash: "dashed")
  content((W + 1.1, A - 0.35), text(fill: normal-colour, size: 8pt)[$sigma_(x x)$ normal, $N sin(omega t + phi.alt)$], anchor: "west")
  let xs = W * 0.25
  let xn = W * (0.25 - phase / 360)
  seg((xn, -A - 0.15), (xs, -A - 0.15), grey, th: 0.6pt)
  seg((xn, -A - 0.22), (xn, -A - 0.08), grey, th: 0.6pt)
  seg((xs, -A - 0.22), (xs, -A - 0.08), grey, th: 0.6pt)
  content((0.5 * (xn + xs), -A - 0.2), text(fill: grey, size: 8pt)[$phi.alt$], anchor: "north", padding: 2pt)
  content((W * 0.25, -A * 0.45), text(fill: grey, size: 7.5pt)[right-lateral\ half cycle], anchor: "center")
  content((W * 0.75, A * 0.5), text(fill: grey, size: 7.5pt)[left-lateral\ half cycle], anchor: "center")
})

#grid(columns: 2, column-gutter: 14pt, align: bottom,
  [#panel-a #v(2pt) #text(size: 8pt)[(a) block, boundary conditions and the fault]],
  [#panel-b #v(2pt) #text(size: 8pt)[(b) the harmonic end load: normal and shear tractions with a phase lag $phi.alt$]],
)
