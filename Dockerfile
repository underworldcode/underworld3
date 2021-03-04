FROM underworldcode/carbonite:latest
COPY . /tmp/uw3
WORKDIR /tmp/uw3
RUN pip3 install .
