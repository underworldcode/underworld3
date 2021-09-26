# Example markdown notebook

`jupyter-book` uses `jupytext` to make markdown notebooks. These documents are much easier to render and their cell-output is not part of the revision control system. 

If you want to launch these notebooks in binder or a jupyterhub, it makes sense to choose the *classic* notebook option in the configuration of the launch links (at least until jupyter-lab becomes a bit more friendly towards jupytext).

```yaml
launch_buttons:
  jupyterhub_url: "https://myhub.mydomain"  # The URL for your JupyterHub. 
  binderhub_url:  "https://mybinder.org"    # The URL of the BinderHub 
  notebook_interface: "classic" # "jupyterlab" or "classic"
``` 


```python
import numpy as np
```

```python
%%timeit

A = np.zeros((1000,1000))
for i in range(0,1000):
    for j in range(0,1000):
        A[i,j] = 2.0
        
```

```python
%%timeit

B = np.zeros((1000,1000))
B[:,:] = 2.0
```

```python
%%timeit

C = np.zeros((1000,1000))
C[...] = 2.0
```

```python
%%timeit

D = 2.0 *  np.ones((1000,1000))
```

```python
%%timeit
L = []
for i in range(0,1000):
    L.append([])
    for j in range(0,1000):
        L[-1].append(2.0)
       
```

## Important note !

The output cells in this notebook will not be part of the book **unless it is executed during the build**. `ipynb` notebooks are rendered in the state that you left them when editing !

```python

```
