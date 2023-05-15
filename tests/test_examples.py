import subprocess
import pytest
import ntpath
from inspect import getsourcefile

wdir = ntpath.dirname(getsourcefile(lambda:0))+"/../examples/"

# get ipynb scripts to test
#import glob
#scripts = [pytest.param(path, id=ntpath.basename(path)) for path in sorted(glob.glob(wdir+"/*.py"))]

scripts = [
           "Jupyterbook/Notebooks/Examples-StokesFlow/Ex_stokes_sinkingBlock_benchmark.py",
          ]

@pytest.mark.parametrize('script', scripts)
def test_script_execution(script):
    subprocess.run(["pytest", script], check=True)
