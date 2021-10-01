#! /usr/bin/env bash

# This will build the slides, then the book

cd Lectures
source build_slides.sh 
cd ..

# This is where we can build LaTeX beamer content to pdfs
# pdfs should be build to Lectures/static_slides/PDFs

# Now build the remaining book

jupyter-book build . 

# This is best done by hand so it updates the slides even 
# if there is no work to be done in rebuilding the book 

cp -r Lectures/static_slides/slideshows _build/html
cp -r Lectures/static_slides/PDFs _build/html

mkdir -p _build/html/Figures/Movies
cp -r Figures/Movies _build/html/Figures
cp -r Figures/PDFs _build/html/Figures

# mkdir -p _build/html/Exercises/Resources
# cp -r Exercises/Resources _build/html/Exercises
