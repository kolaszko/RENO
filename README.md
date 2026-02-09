# RENO: Real-Time Neural Compression for 3D LiDAR Point Clouds

[![arXiv](https://img.shields.io/badge/Arxiv-2503.12382-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2503.12382)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

## 🧩 Local additions

- **ROS bag compression pipeline** via [compress_and_rewrite_bag.py](compress_and_rewrite_bag.py), supporting ROS1 and ROS2 bags. Output is written next to the input bag in a `reno/<bag_name>_posq<POSQ>/` folder, along with `statistics.json`.
- **Docker execution** for GPU-enabled, reproducible runs (see Docker usage below).
- **Helper scripts**: [single_run.sh](single_run.sh) for one bag and [batch_run.sh](batch_run.sh) for multiple bags.

### 🐳 Docker usage (ROS bag compression)

**Prerequisites:** NVIDIA GPU + NVIDIA Container Toolkit.

1. Place your bags under `../datasets/` (host). They are mounted to `/datasets` in the container.
2. Run with environment variables:

```bash
docker compose up --build \
    --remove-orphans
```

```bash
BAG_PATH=/datasets/example.bag \
POSQ=16 \
TOPIC=/velodyne_points \
MAX_MESSAGES=100 \
docker compose up --build
```

**Notes:**
- `TOPIC` and `MAX_MESSAGES` are optional.
- Outputs are saved under `/datasets/reno/<bag_name>_posq<POSQ>/` on the host.

You can also use the helpers:

```bash
./single_run.sh /datasets/example.bag 16 /velodyne_points 100
./batch_run.sh
```

## ✨ Introduction
This repository is the offical PyTorch implementation of our paper *RENO: Real-Time Neural Compression for 3D LiDAR Point Clouds*.

**Abstract:** Despite the substantial advancements demonstrated by learning-based neural models in the LiDAR Point Cloud Compression (LPCC) task, realizing real-time compression—an indispensable criterion for numerous industrial applications—remains a formidable challenge. This paper proposes RENO, the first real-time neural codec for 3D LiDAR point clouds, achieving superior performance with a lightweight model. RENO skips the octree construction and directly builds upon the multiscale sparse tensor representation. Instead of the multi-stage inferring, RENO devises sparse occupancy codes, which exploit cross-scale correlation and derive voxels' occupancy in a one-shot manner, greatly saving processing time. Experimental results demonstrate that the proposed RENO achieves real-time coding speed, 10 fps at 14-bit depth on a desktop platform (e.g., one RTX 3090 GPU) for both encoding and decoding processes, while providing 12.25% and 48.34% bit-rate savings compared to G-PCCv23 and Draco, respectively, at a similar quality. RENO model size is merely 1MB, making it attractive for practical applications. The source code is available at https://github.com/NJUVISION/RENO.

## ⚙️ Environment Setup

```bash
conda create -n reno python=3.10
conda activate reno

# Install pytorch (older versions such as pytorch 1.10 should also be compatible)
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia

# Install torchsparse
# You may need to ensure that the CUDA version displayed by nvcc -V matches the CUDA version in the environment
apt-get install libsparsehash-dev
git clone https://github.com/mit-han-lab/torchsparse.git && cd torchsparse
python setup.py install

# Install other dependencies
pip install torchac open3d numpy==1.*
```

## 😀 Usage

The pre-trained weights are located in the `./model/` directory. These weights can be directly utilized for the encoding&decoding of point cloud geometry.

### Compression

We use `posQ` to adjust the compression rate. 

- For the original-scale point cloud with floating-point coordinates, such as the raw data in KITTI Detection and SemanticKITTI, the following commands can be used for compression. When the `posQ` parameter is set to 16, the point cloud precision is 14-bits, whereas a `posQ` value of 64 results in a precision of 12-bits. In our paper, we use `posQ` values of `{8, 16, 32, 64, 128, 256, 512}`.

```bash
python compress.py \
    --input_glob='./data/kittidet_examples/*.ply' \
    --output_folder='./data/kittidet_compressed/' \
    --ckpt='./model/KITTIDetection/ckpt.pt' \
    --posQ=16
```

- For pre-quantized point clouds, please set `is_data_pre_quantized=True` as following. Make sure that your pre-quantized coordinates are all positive values.

```bash
python compress.py \
    --input_glob='./data/ford_vox1mm_examples/*.ply' \
    --output_folder='./data/ford_vox1mm_compressed/' \
    --is_data_pre_quantized=True \
    --ckpt='./model/Ford/ckpt.pt' \
    --posQ=16
```


### Decompression

- For the original-scale point cloud, use the following:

```bash
python decompress.py \
    --input_glob='./data/kittidet_compressed/*.bin' \
    --output_folder='./data/kittidet_decompressed/' \
    --ckpt='./model/KITTIDetection/ckpt.pt'
```

- For pre-quantized point clouds, make sure that `is_data_pre_quantized` is set to `True`:

```bash
python decompress.py \
    --input_glob='./data/ford_vox1mm_compressed/*.bin' \
    --output_folder='./data/ford_vox1mm_decompressed/' \
    --is_data_pre_quantized=True \
    --ckpt='./model/Ford/ckpt.pt'
```

### Evaluation

```bash
chmod +x ./third_party/pc_error_d
```

```bash
python eval.py \
    --input_glob='./data/kittidet_examples/*.ply' \
    --decompressed_path='./data/kittidet_decompressed' \
    --pcc_metric_path='./third_party/pc_error_d' \
    --resolution=59.70
```

```bash
python eval.py \
    --input_glob='./data/ford_vox1mm_examples/*.ply' \
    --decompressed_path='./data/ford_vox1mm_decompressed' \
    --pcc_metric_path='./third_party/pc_error_d' \
    --resolution=30000
```

## ⏳ Training From Scratch

- The following command can be used to train on the KITTI Detection dataset:

```bash
python train.py \
    --training_data='../KITTI_detection/training/velodyne/*.bin' \
    --model_save_folder='./model/KITTIDetection' \
    --valid_samples='../KITTI_detection/ImageSets/train.txt'
```

- Please set `is_data_pre_quantized=True` for the point clouds which have been quantized to 18-bits.

```bash
python train.py \
    --training_data='../Ford_01_q_1mm/*.ply' \
    --model_save_folder='./model/Ford' \
    --is_data_pre_quantized=True
```

## 🌊 Citation

If you find this work useful, we would appreciate a citation:

```
@inproceedings{you2025reno,
  title={Reno: Real-time neural compression for 3d lidar point clouds},
  author={You, Kang and Chen, Tong and Ding, Dandan and Asif, M Salman and Ma, Zhan},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={22172--22181},
  year={2025}
}
```
