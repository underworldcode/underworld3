FROM underworldcode/carbonite:petsc_dev
COPY . /tmp/uw3
WORKDIR /tmp/uw3
RUN pip3 install .
RUN pip3 install sympy