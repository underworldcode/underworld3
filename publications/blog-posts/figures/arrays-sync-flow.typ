#import "@preview/cetz:0.3.4"

#set page(width: auto, height: auto, margin: 16pt)
#set text(size: 10pt)

#cetz.canvas({
  import cetz.draw: *

  // Colours
  let green-bg = rgb("#e8f4e8")
  let green-fg = rgb("#49a87c")
  let yellow-bg = rgb("#fef3c7")
  let yellow-fg = rgb("#d9960a")
  let blue-bg = rgb("#dce8fc")
  let blue-fg = rgb("#4a7bf7")
  let blue-node = rgb("#b8d4f8")
  let pink-bg = rgb("#fce4ec")
  let pink-fg = rgb("#e57373")
  let pink-node = rgb("#f8bbd0")
  let label-bg = rgb("#e8e8e8")
  let label-fg = rgb("#999999")

  // Helper: rounded box with text
  let node(pos, label, fill: white, stroke: black, name: none, width: 3.2, height: 1.2) = {
    rect(
      (pos.at(0) - width / 2, pos.at(1) - height / 2),
      (pos.at(0) + width / 2, pos.at(1) + height / 2),
      fill: fill, stroke: stroke, radius: 6pt, name: name,
    )
    content(pos, align(center, label))
  }

  // Helper: region box
  let region(tl, br, label, fill: white, stroke: black) = {
    rect(tl, br, fill: fill, stroke: stroke, radius: 8pt)
    content(
      ((tl.at(0) + br.at(0)) / 2, tl.at(1) - 0.3),
      text(weight: "bold", size: 9pt, label),
    )
  }

  // Helper: edge label pill
  let elabel(pos, label) = {
    rect(
      (pos.at(0) - 0.85, pos.at(1) - 0.22),
      (pos.at(0) + 0.85, pos.at(1) + 0.22),
      fill: label-bg, stroke: label-fg + 0.5pt, radius: 4pt,
    )
    content(pos, text(size: 8pt, label))
  }

  // Regions
  region((-0.5, 1.2), (3.5, -1.6), "User Space", fill: green-bg, stroke: green-fg)
  region((6.5, 1.2), (10.8, -1.6), " ", fill: yellow-bg, stroke: yellow-fg)
  region((14, 2.0), (23.5, -2.4), "PETSc", fill: blue-bg, stroke: blue-fg)
  region((16, -3.2), (22, -5.6), "Neighbours (MPI)", fill: pink-bg, stroke: pink-fg)

  // Nodes
  node((1.5, -0.2), [`.data` / `.array`\ (NumPy)], fill: rgb("#c8e6c8"), stroke: green-fg, name: "user")
  node((8.65, -0.2), [`NDArray_With_Callback`\ `id(_lvec)` check], fill: rgb("#fde68a"), stroke: yellow-fg, name: "cache", width: 3.6)
  node((16.5, -0.2), [Local Vector\ (with ghosts)], fill: blue-node, stroke: blue-fg, name: "lvec")
  node((21.0, -0.2), [Global Vector\ (owned DOFs)], fill: blue-node, stroke: blue-fg, name: "gvec")
  node((19.0, -4.4), [Ghost Exchange], fill: pink-node, stroke: pink-fg, name: "mpi", width: 2.6, height: 1.0)

  // Arrows: user → cache → lvec
  line((3.1, -0.2), (6.85, -0.2), mark: (end: ">", fill: black), stroke: 0.8pt)
  elabel((5.0, 0.3), "callback")

  line((10.45, -0.2), (14.9, -0.2), mark: (end: ">", fill: black), stroke: 0.8pt)
  elabel((12.7, 0.3), "pack")

  // Arrows: lvec → gvec (top arc)
  bezier(
    (17.5, 0.6), (20.0, 0.6),
    (18.2, 1.5), (19.3, 1.5),
    mark: (end: ">", fill: black), stroke: 0.8pt,
  )
  elabel((18.75, 1.55), "localToGlobal")

  // Arrows: gvec → lvec (bottom arc)
  bezier(
    (20.0, -1.0), (17.5, -1.0),
    (19.3, -1.9), (18.2, -1.9),
    mark: (end: ">", fill: black), stroke: 0.8pt,
  )
  elabel((18.75, -1.95), "globalToLocal")

  // Arrows: lvec ↔ mpi (dashed)
  line(
    (16.0, -0.8), (17.8, -3.9),
    mark: (end: ">", fill: black), stroke: (dash: "dashed", thickness: 0.6pt),
  )
  elabel((16.0, -2.4), "scatter")

  line(
    (20.2, -3.9), (17.2, -0.8),
    mark: (end: ">", fill: black), stroke: (dash: "dashed", thickness: 0.6pt),
  )
  elabel((20.0, -2.4), "fill ghosts")
})
