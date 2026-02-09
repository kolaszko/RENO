FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel as base

WORKDIR /app

# Copy project files
COPY . /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libsparsehash-dev \
    && rm -rf /var/lib/apt/lists/*

RUN conda install -y -c conda-forge ninja filelock git pip
RUN /opt/conda/bin/pip install torchac open3d rosbags

FROM base as executor
# Set build environment variables for torchsparse
ENV MAX_JOBS=4
ENV FORCE_CUDA=1
ENV CUDA_HOME=/usr/local/cuda-12.4
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 9.0+PTX"

RUN cd third_party/torchsparse && python setup.py install

# Run the compression script
CMD ["/bin/bash", "-c", "source /app/env.sh && python compress_and_rewrite_bag.py"]
