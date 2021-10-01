#! /usr/bin/env bash

# This will build all the reveal.md files it finds in the root directory
# These files are also ignored by the jupyterbook script


if ! command -v reveal-md &> /dev/null
then
    npm install -g reveal-md 
fi

mkdir -p static_slides/slideshows
mkdir -p static_slides/PDFs

reveal-md  --static static_slides/slideshows \
           --theme css/anu.css --glob '**/*.reveal.md' \
           --separator '<--o-->' \
           --vertical-separator '<--v-->' \
           --static-dirs movies,images  
