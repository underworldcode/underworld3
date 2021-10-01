# Jupyterbook setup for github pages

After you have cloned the repository, make sure that you have enabled GitHub Pages on the gh-pages branch
from the repository Settings / Options page. If the gh-pages branch is missing, just go ahead and create it. 

The _config.yml file will need to be edited to reflect the location of the repository

```
    repository:
    url         : https://github.com/underworld-geodynamics-cloud/self-managing-jupyterhub  # Online location of this book
    branch      : master  # Which branch of the repository should be used when creating links (optional)
```

and, if you intend to set up and use the jupyterhub (as well as or instead of binder.org), set the url of the landing point of the hub. 

```
launch_buttons:
  jupyterhub_url: "https://demon.underworldcloud.org"  # The URL for your JupyterHub. 
  binderhub_url:  "https://mybinder.org"  # The URL of the BinderHub (e.g., https://mybinder.org)
  notebook_interface: "classic" # "jupyterlab" or "classic"
```

If you do not intend to use the jupyterhub, then you ought to disable the github actions that run it. 
To do this, you could simply rename these files (e.g. change the extension)

```
.github/workflows/health_check.yml
.github/workflows/install_tljh.yml
.github/workflows/update_packages.yml
```


## Some notes about jupyterbook

Sphinx admonitions that are available:

 -  default
 -  note
 -  attention
 -  caution
 -  warning
 -  danger
 -  error 
 -  hint 
 -  tip     
 -  important

Style sheet information [on github](https://github.com/pydata/pydata-sphinx-theme/blob/master/pydata_sphinx_theme/static/css/theme.css)