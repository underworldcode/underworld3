# Presentations System — Development Notes

## Quick Start

```bash
cd docs/presentations
./build.sh              # or just: quarto render
open _build/index.html  # view results
```

## Structure

```
presentations/
├── _quarto.yml                    # Project config (outputs to _build/)
├── _extensions/
│   └── underworld/uw3_slides/     # Custom theme extension
│       ├── _extension.yml
│       └── uw3.scss               # UW3 maroon theme
├── build.sh                       # Build script (called by docs/build-docs.sh)
├── images/                        # Presentation images
├── index.qmd                      # Slides index page
├── underworld3-intro.qmd          # Main presentation (skeleton)
└── uw3-introduction.qmd           # Alternative presentation
```

## Build Integration

- **Standalone**: Run `./build.sh` or `quarto render` from this directory
- **With docs**: Run `docs/build-docs.sh` which calls `presentations/build.sh` first
- **Output**: `presentations/_build/`

## Important: No self-contained

Do NOT add `self-contained: true` to `_quarto.yml` — it creates giant HTML files that break GitHub and browsers. Only use for small demos.

## Theme Details

The `uw3_slides` extension provides:
- Primary colour: #883344 (UW3 maroon)
- Fonts: Jost (text), JetBrains Mono (code)
- Left-aligned slides
- Mermaid diagram support
- Chalkboard plugin enabled

## Presentation Content

`uw3-introduction.qmd` covers the full UW3 overview:
1. Why Underworld3 (landscape, design philosophy)
2. Architecture (with Mermaid diagrams)
3. Core concepts (meshes, variables, symbolic expressions)
4. Solvers (Poisson, Stokes)
5. Time evolution (advection-diffusion)
6. Materials & particles (swarms)
7. Units & scaling
8. AI-friendly patterns
9. BatBot vision
10. Roadmap

## Adding New Presentations

1. Create `new-presentation.qmd` with YAML header:
   ```yaml
   ---
   title: "Your Title"
   format:
     uw3_slides-revealjs:
       slide-number: true
   ---
   ```

2. Add to `index.qmd` links

3. Run `./build.sh`

## Reference Implementation

Pattern based on `~/+Github/EMSC-QuartoBook-Course/WebSlides/`
