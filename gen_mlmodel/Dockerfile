# Try to reproduce an environment from February 2021.
FROM debian:10

# Install Miniconda.
RUN apt-get update && apt-get install -y curl \
    && curl -L https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh -o conda.sh \
    && chmod +x conda.sh \
    && ./conda.sh -b -p /opt/conda

# Create and activate a Conda environment
RUN /opt/conda/bin/conda create -n myenv python=3.8 \
    && echo "source activate myenv" > ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

RUN source activate myenv
RUN conda install coremltools==4.1
CMD ["python3", "-c", "from coremltools.converters import keras as keras_converter"]
