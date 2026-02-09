FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel as base

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libsparsehash-dev \
    ca-certificates openssl libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN /opt/conda/bin/pip install ninja==1.13.0 filelock==3.20.1 tox==4.32.0 gitpython
RUN /opt/conda/bin/pip install torchac==0.9.3 open3d==0.19.0 rosbags==0.11.0

FROM base as executor
# Set build environment variables for torchsparse
ENV MAX_JOBS=4
ENV FORCE_CUDA=1
ENV CUDA_HOME=/usr/local/cuda-12.4
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 9.0+PTX"

# Copy only torchsparse for installation
COPY third_party/torchsparse /app/third_party/torchsparse
RUN cd third_party/torchsparse && python setup.py install

# Copy the rest of the application code last
COPY . /app

# Run the compression script
CMD ["/bin/bash", "-c", "source /app/env.sh && python compress_and_rewrite_bag.py"]
