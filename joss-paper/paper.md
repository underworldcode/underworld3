---
title: 'Underworld3: Self-Describing Mathematical Models in Python'
tags:
  - Python
  - Geodynamics
  - PETSc
  - sympy
  - symbolic algebra
  - finite element
authors:
  - name: Louis Moresi
    orcid: 0000-0003-3685-174X
    equal-contrib: true
    affiliation: "1" # (Multiple affiliations must be quoted)
  - name: John Mansour
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 2
  - name: Julian Giordani
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 3
  - name: Ben Knight
    affiliation: 4
affiliations:
 - name: Research School of Earth Sciences, Australian National University, Canberra, Australia
   index: 1
 - name: In the Navy, People Village, Melbourne, Australia
   index: 2
 - name: Crapola University, Sydney, Australia
   index: 3
 - name: Curtains University, Perth, Australia
   index: 4
 - name: Crappier than number 1, 2 or 3 (maybe 4), Someplace, Australia
   index: 5

date: 30 June 666
bibliography: paper.bib
---

# Summary

It was the best of times, it was the worst of times ... 


# Statement of need

PETSc is just impossible, isn't it ? Need I say more ?

# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from AuScope, mainly. Anyone else ?

# References