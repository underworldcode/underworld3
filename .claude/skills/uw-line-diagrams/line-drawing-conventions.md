# Line Drawings for Articles: Conventions and a Practical Recipe

A working reference for engineering-style line diagrams (with a fluid-dynamics slant), covering graphical conventions, production specifications, and tooling.

---

## 1. Line hierarchy: weight and style carry meaning

The formal lineage is ISO 128 ("Basic conventions for lines"), which fixes a small vocabulary of basic line types — continuous, dashed, dashed-spaced, long-dashed dotted, long-dashed double-dotted, dotted, etc. — together with rules for leader lines ([ISO 128-2:2022](https://cdn.standards.iteh.ai/samples/83355/10bb39d36fc34caeb80ecd25347ddb0c/ISO-128-2-2022.pdf)). The American counterpart is ASME Y14.2, "Line Conventions and Lettering" ([ASME](https://www.asme.org/codes-standards/find-codes-standards/y14-2-line-conventions-lettering)). Journals do not enforce these standards on authors, but textbook line drawings inherit their logic: **the reader should be able to classify any line by weight and dash pattern alone.**

A robust three-weight, four-style scheme:

| Role | Line style | Typical weight |
|---|---|---|
| Physical boundaries: walls, bodies, solid surfaces | Solid | Heavy (1–1.5 pt) |
| Fluid interfaces, free surfaces, geometry that matters | Solid | Medium (0.5–1 pt) |
| Streamlines, contour lines, construction/reference geometry | Solid, thin | Light (0.25–0.5 pt) |
| Hidden or idealized geometry; streamlines shown "behind" a body | Dashed | Medium |
| Control-volume boundary; computational domain | Dashed or dash-dot, closed curve | Medium |
| Centre lines, axes of symmetry | Long-dash-dot (the classic "centre line") | Light–medium |
| Dimension/leader lines | Thin solid with arrowheads or dots | Light |

Rules of thumb:

- Keep at most **three distinct weights**; anything finer is read as noise after reduction.
- Use dash patterns **consistently within one paper** — a dash-dot line should mean the same thing in Figure 2 as in Figure 12.
- Distinguish line *roles* by style, not by colour alone (see §4).

## 2. Fluid-dynamics conventions specifically

There is no ISO-style formal standard for fluid schematics; the conventions below are the strong disciplinary habits distilled from standard textbooks (White, Kundu & Cohen, Panton, Bar-Meir) and lecture notes:

**Streamlines vs vectors vs trajectories.** Streamlines are thin, continuous curves tangent to the local velocity, drawn sparsely and given arrowheads only occasionally — a wall-to-wall thicket of arrowheads reads as noise. Velocity *vectors* are straight arrows anchored at points; if their lengths are not proportional to a real scale, say "schematic" in the caption. Keep the distinction between streamline, pathline, and streakline visually honest (dashed is often used for pathlines/trajectories).

**Arrow classes.** Give each physical quantity its own arrow treatment. One worked example worth copying comes from Bar-Meir's *Basics of Fluid Mechanics*, which applies a consistent scheme across all its figures: blue for motion (velocity), green for forces/pressure/torque, black for dimensions and distances, red for energy/heat/work ([Basics of Fluid Mechanics](https://madar-ju.com/storage/images/files/file_1738976438FOFYp.pdf)) — a textbook habit rather than a formal standard, but exactly the kind of consistency reviewers notice. Whatever scheme you choose, forces and fluxes should be drawn heavier than streamlines, and every arrow class should be legible in grayscale.

**Control volumes.** A closed dashed or dash-dot boundary, with mass/momentum flux arrows crossing it (in and out) and outward normals marked where the argument depends on them. The control surface should be *precisely defined* — an unambiguous closed surface fixed in a stated frame ([Caltech Fluidbook](http://brennen.caltech.edu/fluidbook/basicfluiddynamics/massconservation/controlvolume.pdf)).

**Boundary layers and thin features.** Exaggerate the thickness and say so ("boundary-layer thickness exaggerated for clarity"); never let a reader mistake the drawing for a scale rendition.

**Inlets/outlets.** Arrows crossing the domain boundary, labelled with the relevant quantities — \(u\), \(Q\), \(\dot m\), \(p\), \(T\), Re — so the schematic and the governing equations line up term by term.

**Environment.** Show coordinate axes, gravity \(\mathbf g\), rotation, and section planes (with section arrows, as in ISO/ASME sectioning) explicitly. A reader should never have to infer the frame.

**Texture.** Use hatching (45° thin lines) for solids/cross-sections, and leave fluid regions untextured or lightly tinted — the classic engineering-drawing convention that makes walls read as walls.

## 3. Production specifications (what journals actually check)

Draw the figure **vector** (PDF/EPS, or SVG via Inkscape) at the journal's final column width. Key numbers, current as of 2026 — these are common publisher targets (Nature, Elsevier, PLOS, ASME, PNAS, MAA), not universal rules; the destination journal's own instructions always win:

| Parameter | Specification | Source |
|---|---|---|
| Line weight | 0.25–1 pt at final size (Nature); below 0.25 pt may vanish in print | [Scientific Figure](https://scientificfigure.com/blog/how-to-make-scientific-figures) |
| Line weight | 0.5–1.5 pt; below 0.5 pt reproduces poorly (ASME proceedings) | [ASME](https://www.asme.org/publications-submissions/proceedings/formatting-the-paper/dealing-with-graphics) |
| Line weight | ≥ 0.5 pt; note that TikZ's default stroke (~0.3–0.4 pt) is "printable but very thin" | [MAA figure instructions](https://maa.org/wp-content/uploads/2025/10/Revised_Figure_Instructions2025.pdf) |
| Text size after reduction | 5–7 pt (Nature), 6 pt min (ASME, Elsevier subscripts), 8 pt min (PLOS) | [Scientific Figure](https://scientificfigure.com/blog/how-to-make-scientific-figures), [ASME](https://www.asme.org/publications-submissions/proceedings/formatting-the-paper/dealing-with-graphics) |
| Column widths | 89 mm (Nature) / 90 mm (Elsevier) single; 183–190 mm full width | [Scientific Figure](https://scientificfigure.com/blog/how-to-make-scientific-figures) |
| Formats | Vector (EPS/PDF) for line art, schematics, graphs; avoid low-resolution PNG exports of drawn content | [Scientific Figure](https://scientificfigure.com/blog/how-to-make-scientific-figures) |
| Raster fallback | If line art must be rasterized: 1000 dpi (Elsevier), 1000–1200 ppi (PNAS) | [Scientific Figure](https://scientificfigure.com/blog/how-to-make-scientific-figures), [PNAS](https://www.pnas.org/pb-assets/authors/digitalart-1675347574760.pdf) |
| Fonts | One sans-serif (Helvetica/Arial) across all figures; embedded; outlined only as the last step | [Scientific Figure](https://scientificfigure.com/blog/how-to-make-scientific-figures) |
| Grayscale | Check the grayscale conversion before submission; print proceedings are often B&W | [ASME](https://www.asme.org/publications-submissions/proceedings/formatting-the-paper/dealing-with-graphics) |

Practical habits:

- **Design at final size.** Draw at exactly 90 mm (or 183 mm) wide, then print at 100% and inspect on paper — screens flatter thin lines ([Scientific Figure](https://scientificfigure.com/blog/how-to-make-scientific-figures)).
- **Direct labels beat legends.** Label \(u\), \(p\), \(\mathbf F\) on the drawing with leader lines; reserve the caption for what cannot be drawn.
- **Match the manuscript's math.** Use the same italic/upright conventions for symbols in figures as in the text (\(u\) italic for variables, upright for units/operators).
- **Avoid 3D perspective** unless the geometry demands it; textbook style is orthographic or mildly oblique schematic.
- Every arrow, symbol, and abbreviation must be explained — ICMJE and most publishers require the figure to be self-explanatory from the legend ([Scientific Figure](https://scientificfigure.com/blog/how-to-make-scientific-figures)).

## 4. Colour, when you use it

- Colour must be a *secondary* channel: every colour-coded distinction needs a redundant encoding (line style, dash pattern, direct label) so the figure survives grayscale and colour-vision deficiency (~8% of men) ([Scientific Figure](https://scientificfigure.com/blog/how-to-make-scientific-figures)).
- Never encode data in red-vs-green alone; magenta/green or blue/orange are the standard safe pairs.
- For continuous fields, use perceptually uniform colormaps (Crameri's scientific colour maps), which keep ordering under grayscale conversion ([Scientific Figure](https://scientificfigure.com/blog/how-to-make-scientific-figures)).
- In a line *diagram* (as opposed to a data plot), restrained flat tints for fluid/solid regions plus black linework age best — this is the classic textbook look.

## 5. Toolchain: LaTeX/TikZ and Typst/CeTZ

Both ecosystems below express the same conventions; pick the one matching your manuscript and keep the style definitions in a shared file so every figure inherits them.

**LaTeX / TikZ-PGF.** The gold standard for reproducible, journal-quality schematics; fonts and math automatically match the manuscript. Izaak Neutelings' TikZ.net hosts a large, well-crafted **Fluid Dynamics category** (laminar/turbulent flow, Bernoulli/Venturi, buoyancy, surface tension...) with downloadable `.tex` sources and Overleaf links — an excellent starting point to steal styles from ([tikz.net fluid dynamics](https://tikz.net/fluid_dynamics_laminar/)). Note the house conventions in those figures: `very thick` round-capped vectors for velocity, distinct heavy styles for forces, LaTeX arrowheads, dashed construction lines, and light flat tints for water.

**Typst / CeTZ.** [CeTZ](https://github.com/cetz-package/cetz) ("CeTZ, ein Typst Zeichenpaket") is the TikZ-equivalent drawing package for Typst — a TikZ/Processing-inspired API with relative coordinates, TikZ-style anchors, and an auto-resizing canvas ([GitHub](https://github.com/cetz-package/cetz)). It maps naturally onto every convention in this guide:

- **Line weights** use plain Typst strokes: `stroke: 1.2pt`, or `stroke: (paint: red, thickness: 1pt, cap: "round")` ([CeTZ styling docs](https://cetz-package.github.io/docs/basics/styling/)).
- **Dash patterns** use Typst's predefined names — `"dashed"`, `"dotted"`, `"dash-dotted"` (plus `densely-`/`loosely-` variants) or explicit arrays like `(10pt, 5pt, "dot", 5pt)` for centre-line and control-volume styles ([Typst stroke docs](https://typst.app/docs/reference/visualize/stroke/)).
- **Arrowheads** are *marks*: `mark: (end: "stealth", fill: black, scale: 0.8)`, with shorthands `">"` (triangle) and `">>"` (stealth), and options for `harpoon`, `flip`, `reverse`, `length`, `width` — a direct analogue of TikZ's `arrows.meta` ([CeTZ marks docs](https://cetz-package.github.io/docs/basics/marks/)).
- **Global styles** via `set-style(...)`, the counterpart of `\tikzset` ([CeTZ styling docs](https://cetz-package.github.io/docs/basics/styling/)).
- **Hatching** needs one extra package: build a diagonal-line pattern with `modpattern` (or Typst's `tiling`) and pass it as `fill:` ([CeTZ issue #805](https://github.com/cetz-package/cetz/issues/805), [Typst forum](https://forum.typst.app/t/is-there-a-way-to-do-hatched-in-filling-like-this-in-typst-cetz/2467)).
- Gotcha: since CeTZ 0.4.2 the line's stroke styling (e.g. its dash pattern) is also applied to its arrowhead; override with `mark: (..., stroke: (dash: none))` ([Typst forum](https://forum.typst.app/t/how-to-get-different-stroke-styles-for-a-line-and-its-mark-in-cetz-0-4-2/8264)).
- For node-and-arrow diagrams (flow charts, force diagrams), [Fletcher](https://typst.app/universe/package/fletcher/) builds on CeTZ and is the standard choice ([Typst Universe](https://typst.app/universe/package/tiptoe/)).

**A shared gallery for both.** [janosh/diagrams](https://github.com/janosh/diagrams) is a browsable collection of 140+ MIT-licensed scientific diagrams in physics, chemistry and ML, each downloadable as PDF/SVG/PNG **with its source in either TikZ `.tex` or CeTZ `.typ` form** — ideal for seeing the same conventions expressed in both languages ([GitHub](https://github.com/janosh/diagrams)).

**Inkscape.** Best for interactive vector drawing and cleanup; exports PDF/EPS; keep the SVG as source. Use for figures that start as sketches rather than code.

**matplotlib/Julia.** Right tool when geometry is *computed* (streamline fields from actual model output); export PDF (vector) rather than PNG; strip chartjunk and re-letter in the manuscript font.

**Recommended workflow for a Markdown/Typst/Myst pipeline:**

1. Keep a `figures/` directory with *source* (`.tex`/`.svg`/`.typ`) and a build step exporting `.pdf` (LaTeX) or `.svg` (web/Myst).
2. Define one shared style file (line weights, arrow styles, colours, font sizes) imported by every figure — consistency across figures is what publishers and reviewers actually notice.
3. Version figures with the manuscript; regenerate from source at submission time so the PDF and the source never drift.

### A minimal TikZ style block to start from (LaTeX)

```latex
% \usetikzlibrary{arrows.meta,patterns}
% line weights: 0.5pt / 0.8pt / 1.2pt at 90mm column width
\tikzset{
  boundary/.style = {line width=1.2pt},                     % walls, bodies
  interface/.style = {line width=0.8pt},                     % free surface
  streamline/.style = {line width=0.5pt},                    % flow lines
  hidden/.style    = {line width=0.8pt, dashed},
  cvline/.style    = {line width=0.8pt, dash pattern=on 5pt off 2pt on 1pt off 2pt},
  axis/.style      = {line width=0.5pt, dash pattern=on 8pt off 2pt on 1pt off 2pt},
  vvec/.style      = {-{Latex[length=3.5pt,width=2.5pt]}, line width=1.0pt, line cap=round},
  force/.style     = {-{Latex[length=4pt,width=3pt]},   line width=1.4pt, line cap=round},
  dim/.style       = {{Latex[length=3pt,width=2.5pt]}-{Latex[length=3pt,width=2.5pt]},
                     line width=0.5pt},
  hatch/.style     = {pattern=north east lines},             % 45-degree hatching
}
```

### A minimal CeTZ style block to start from (Typst)

```typst
#import "@preview/cetz:0.5.2": canvas
#import "@preview/modpattern:0.1.0": modpattern

// shared figure styles — same roles as the TikZ block above
#let styles = (
  boundary:  (stroke: 1.2pt),                                  // walls, bodies
  interface: (stroke: 0.8pt),                                  // free surface
  streamline:(stroke: 0.5pt),                                  // flow lines
  hidden:    (stroke: (thickness: 0.8pt, dash: "dashed")),
  cvline:    (stroke: (thickness: 0.8pt, dash: (5pt, 2pt, 1pt, 2pt))),
  axis:      (stroke: (thickness: 0.5pt, dash: (8pt, 2pt, 1pt, 2pt))),
  vvec:      (stroke: (thickness: 1.0pt, cap: "round"),
              mark: (end: "stealth", fill: black, scale: 0.8)),
  force:     (stroke: (thickness: 1.4pt, cap: "round"),
              mark: (end: "stealth", fill: black, scale: 1.0)),
  dim:       (stroke: 0.5pt,
              mark: (symbol: ">", fill: black, scale: 0.6)),   // both ends
)

// 45-degree hatching for solids/cross-sections
#let hatch = modpattern((3pt, 3pt),
  std.line(start: (0%, 0%), end: (100%, 100%), stroke: 0.4pt))

// usage: everything inside the canvas inherits Typst's text/math fonts
#canvas(length: 1cm, {
  import cetz.draw: *
  rect((0, 0), (4, 1), ..styles.boundary, fill: hatch)   // hatched wall
  line((0.5, 1.2), (2.5, 1.2), ..styles.vvec)              // velocity vector
  content((2.6, 1.2), $u$)                                 // label in math mode
})
```

The two blocks are deliberately parallel: `boundary`/`interface`/`streamline` weights, the dash patterns for hidden lines, control volumes, and centre lines, and the arrow classes (`vvec`, `force`, `dim`) match one-to-one, so a figure can be prototyped in either system and translated mechanically.

## 6. Pre-submission checklist

- [ ] Drawn as vector (PDF/EPS/SVG), not raster; text not outlined until the final export
- [ ] Designed at final column width (90 mm / 183 mm); printed and inspected at 100%
- [ ] All line weights ≥ 0.5 pt at final size; at most three weights
- [ ] All text ≥ 6 pt (journal-dependent; 8 pt is safer) after reduction; one font throughout
- [ ] Every line role distinguishable in grayscale and without colour
- [ ] Streamlines/vectors/forces/dimensions each have a distinct arrow style; all explained in the caption
- [ ] Control volume closed and unambiguous; axes, gravity, normals shown
- [ ] Exaggerations (boundary layers, aspect ratios) declared in the caption
- [ ] Symbols and notation match the manuscript text
- [ ] Source files under version control; figure regenerable by a single build command
- [ ] Destination journal's current figure instructions checked — its spec wins over any general guidance here

---

### Sources

- [ISO 128-2:2022 — Basic conventions for lines](https://cdn.standards.iteh.ai/samples/83355/10bb39d36fc34caeb80ecd25347ddb0c/ISO-128-2-2022.pdf)
- [ASME Y14.2 — Line Conventions and Lettering](https://www.asme.org/codes-standards/find-codes-standards/y14-2-line-conventions-lettering)
- [ASME proceedings — Dealing with Graphics](https://www.asme.org/publications-submissions/proceedings/formatting-the-paper/dealing-with-graphics)
- [MAA Figure Instructions](https://maa.org/wp-content/uploads/2025/10/Revised_Figure_Instructions2025.pdf)
- [Scientific Figure — How to Make Scientific Figures for Journals (2026)](https://scientificfigure.com/blog/how-to-make-scientific-figures)
- [PNAS Digital Art Guidelines](https://www.pnas.org/pb-assets/authors/digitalart-1675347574760.pdf)
- [Bar-Meir, Basics of Fluid Mechanics — figure colour scheme](https://madar-ju.com/storage/images/files/file_1738976438FOFYp.pdf)
- [Caltech Fluidbook — Control Volume](http://brennen.caltech.edu/fluidbook/basicfluiddynamics/massconservation/controlvolume.pdf)
- [TikZ.net Fluid Dynamics examples (Izaak Neutelings)](https://tikz.net/fluid_dynamics_laminar/)
- [CeTZ — a Typst drawing package (GitHub)](https://github.com/cetz-package/cetz)
- [CeTZ documentation — styling](https://cetz-package.github.io/docs/basics/styling/) and [marks](https://cetz-package.github.io/docs/basics/marks/)
- [Typst stroke reference — dash patterns](https://typst.app/docs/reference/visualize/stroke/)
- [modpattern — hatched fills for CeTZ](https://github.com/cetz-package/cetz/issues/805)
- [janosh/diagrams — TikZ and CeTZ scientific diagram gallery](https://github.com/janosh/diagrams)
