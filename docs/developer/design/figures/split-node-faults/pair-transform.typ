// The pair transform. Thesis: one orthogonal 2·dim block per coincident
// DOF pair converts (v+, v-) to mean and jump components in the fault
// frame; the no-opening constraint and the friction law each occupy
// exactly their own rotated row(s), and everything else stays free.
#import "@preview/cetz:0.3.4"

#set page(width: auto, height: auto, margin: 12pt)
#set text(size: 10pt)

#let fault-col = rgb("#c62828")
#let mean-bg = rgb("#dce8fc")
#let jump-n-bg = rgb("#fef3c7")
#let jump-t-bg = rgb("#fce4ec")
#let row-stroke = rgb("#8899aa")

#cetz.canvas(length: 1.35cm, {
  import cetz.draw: *

  let row(y, fill, label, role, name: none) = {
    rect((3.1, y - 0.3), (6.4, y + 0.3), fill: fill,
         stroke: (paint: row-stroke, thickness: 0.5pt), radius: 3pt)
    content((4.75, y), label)
    content((8.55, y), align(left, role))
  }

  // ---- left: the coincident pair with the fault frame --------------------
  // fault trace through the pair
  line((-0.4, 0.6), (2.0, 1.4), stroke: (paint: fault-col,
       thickness: 1.0pt))
  line((-0.4, 0.44), (2.0, 1.24), stroke: (paint: fault-col,
       thickness: 1.0pt, dash: "densely-dashed"))
  // the pair (drawn slightly apart; coincident in reality)
  circle((0.8, 1.0), radius: 0.05, fill: fault-col, stroke: none)
  circle((0.8, 0.84), radius: 0.05, fill: white,
         stroke: (paint: fault-col, thickness: 0.9pt))
  content((0.28, 1.28), text(size: 9pt)[$v^+$])
  content((0.3, 0.52), text(size: 9pt)[$v^-$])
  // frame at the pair
  line((0.8, 1.0), (0.42, 2.0), stroke: (thickness: 0.8pt), mark: (end: ">"))
  content((0.28, 2.2), $hat(n)$)
  line((0.8, 1.0), (1.9, 1.44), stroke: (thickness: 0.8pt), mark: (end: ">"))
  content((2.14, 1.56), $hat(t)_1$)
  content((1.62, 0.92), text(size: 8.5pt, fill: rgb("#555555"))[($hat(t)_2$ in 3-D)])

  content((0.8, -0.4), align(center, text(size: 9pt)[coincident pair
    (2·dim DOFs), \ no shared unknowns]))

  // arrow to the rotated rows
  line((2.25, 0.65), (2.95, 0.65), stroke: (thickness: 0.9pt),
       mark: (end: ">"))
  content((2.6, 1.0), text(size: 9pt)[$Q$])

  // ---- right: the rotated rows -------------------------------------------
  content((4.75, 2.6), align(center)[rotated unknowns (per pair)])
  content((8.55, 2.6), align(center)[role])

  row(1.9, mean-bg, [$(v^+ + v^-) dot hat(n) slash sqrt(2)$],
      text(size: 9pt)[free — bulk momentum])
  row(1.1, mean-bg, [$(v^+ + v^-) dot hat(t) slash sqrt(2)$],
      text(size: 9pt)[free — bulk momentum])
  row(0.3, jump-n-bg, [$[v] dot hat(n) slash sqrt(2)$],
      text(size: 9pt)[constrained $= 0$ — no opening; \ reaction $arrow.r sigma_n$])
  row(-0.7, jump-t-bg, [$[v] dot hat(t) slash sqrt(2) = V slash sqrt(2)$],
      text(size: 9pt)[the law: $integral_Gamma tau(V, sigma_n, theta)
        thin delta V$; \ tangent $2 (diff tau slash diff V) M$ per iterate])

  content((4.75, -1.65), align(center, text(size: 9pt)[
    frictionless: slip row simply FREE (zero shear traction) \
    3-D: two slip rows, collinear traction
    $bold(tau) = tau(|V|) hat(V)$]))
})
