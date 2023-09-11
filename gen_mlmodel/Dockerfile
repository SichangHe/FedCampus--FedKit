# Try to reproduce an environment from February 2021.
FROM continuumio/conda-ci-linux-64-python3.8

RUN sudo chmod -R 777 /opt/conda
RUN conda install coremltools==4.1
CMD ["python3", "-c", "from coremltools.converters import keras as keras_converter"]
