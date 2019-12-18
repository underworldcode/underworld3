FROM underworldcode/carbonite:petsc_3.12
RUN pip3 uninstall -y petsc4py
RUN git clone https://github.com/underworldcode/petsc4py /tmp/petsc4py
WORKDIR /tmp/petsc4py
RUN pip3 install .
COPY . /tmp/uw3
WORKDIR /tmp/uw3/python
RUN pip3 install .