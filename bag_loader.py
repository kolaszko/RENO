#!/usr/bin/env python3
"""
Script to load a ROS bag file and compress/decompress PointCloud2 messages.
Requires: rosbags, torch, torchsparse, torchac

Uses a PointCloudAutoencoder to compress and decompress point clouds in-memory.
"""

import sys
import os
import struct
import numpy as np
from pathlib import Path
import torch
import torchac
from torchsparse import SparseTensor
from torchsparse.nn import functional as F

try:
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore
except ImportError as e:
    print(f"Error: Missing required package - {e}")
    print("Install with: pip install rosbags")
    sys.exit(1)

from network import Network
import kit.op as op
import kit.io as io


class PointCloudAutoencoder:
    """
    Wraps neural network-based compression and decompression for point clouds.
    Handles device setup, model loading, and in-memory compression/decompression.
    """
    
    def __init__(self, checkpoint_path, channels=32, kernel_size=3, pos_quantization=16):
        """
        Initialize the autoencoder.
        
        Args:
            checkpoint_path (str): Path to the trained model checkpoint
            channels (int): Number of channels in the network
            kernel_size (int): Kernel size for convolutions
            pos_quantization (float): Position quantization step
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.channels = channels
        self.kernel_size = kernel_size
        self.posQ = pos_quantization
        
        # Setup convolution config
        conv_config = F.conv_config.get_default_conv_config()
        conv_config.kmap_mode = "hashmap"
        F.conv_config.set_global_conv_config(conv_config)
        
        # Load network
        self.net = Network(channels=channels, kernel_size=kernel_size)
        self.net.load_state_dict(torch.load(checkpoint_path))
        self.net.to(self.device).eval()
        
        # Warm up
        self._warmup()
    
    def _warmup(self):
        """Warm up the network with random input."""
        with torch.no_grad():
            random_coords = torch.randint(low=0, high=2048, size=(2048, 3)).int().to(self.device)
            self.net(SparseTensor(
                coords=torch.cat((random_coords[:, 0:1]*0, random_coords), dim=-1),
                feats=torch.ones((2048, 1))
            ).to(self.device))
    
    def compress(self, xyz):
        """
        Compress a point cloud and return the decompressed result.
        
        Args:
            xyz (np.ndarray): Point cloud coordinates of shape (N, 3)
            
        Returns:
            np.ndarray: Decompressed point cloud of shape (N, 3)
        """
        with torch.no_grad():
            # Preprocess coordinates
            xyz_tensor = torch.tensor(xyz / 0.001 + 131072, device=self.device)
            xyz_tensor = torch.round(xyz_tensor / self.posQ).int()
            N = xyz_tensor.shape[0]
            xyz_tensor = torch.cat((xyz_tensor[:, 0:1]*0, xyz_tensor), dim=-1).int()
            feats = torch.ones((xyz_tensor.shape[0], 1), dtype=torch.float, device=self.device)
            x = SparseTensor(coords=xyz_tensor, feats=feats)
            
            # Preprocessing phase
            data_ls = []
            while True:
                x = self.net.fog(x)
                data_ls.append((x.coords.clone(), x.feats.clone()))
                if x.coords.shape[0] < 64:
                    break
            data_ls = data_ls[::-1]
            
            # Compression phase
            byte_stream_ls = []
            for depth in range(len(data_ls) - 1):
                x_C, x_O = data_ls[depth]
                gt_x_up_C, gt_x_up_O = data_ls[depth + 1]
                gt_x_up_C, gt_x_up_O = op.sort_CF(gt_x_up_C, gt_x_up_O)
                
                x_F = self.net.prior_embedding(x_O.int()).view(-1, self.channels)
                x = SparseTensor(coords=x_C, feats=x_F)
                x = self.net.prior_resnet(x)
                x_up_C, x_up_F = self.net.fcg(x_C, x_O, x.feats)
                x_up_C, x_up_F = op.sort_CF(x_up_C, x_up_F)
                
                x_up_F = self.net.target_embedding(x_up_F, x_up_C)
                x_up = SparseTensor(coords=x_up_C, feats=x_up_F)
                x_up = self.net.target_resnet(x_up)
                
                # Occupancy prediction
                gt_x_up_O_s0 = torch.remainder(gt_x_up_O, 16)
                gt_x_up_O_s1 = torch.div(gt_x_up_O, 16, rounding_mode='floor')
                x_up_O_prob_s0 = self.net.pred_head_s0(x_up.feats)
                x_up_O_prob_s1 = self.net.pred_head_s1(
                    x_up.feats + self.net.pred_head_s1_emb(gt_x_up_O_s0[:, 0].long())
                )
                x_up_O_prob = torch.cat((x_up_O_prob_s0, x_up_O_prob_s1), dim=0)
                gt_x_up_O = torch.cat((gt_x_up_O_s0, gt_x_up_O_s1), dim=0)
                
                # Entropy coding
                x_up_O_cdf = torch.cat((x_up_O_prob[:, 0:1]*0, x_up_O_prob.cumsum(dim=-1)), dim=-1)
                x_up_O_cdf = torch.clamp(x_up_O_cdf, min=0, max=1)
                x_up_O_cdf_norm = op._convert_to_int_and_normalize(x_up_O_cdf, True)
                x_up_O_cdf_norm = x_up_O_cdf_norm.cpu()
                gt_x_up_O = gt_x_up_O[:, 0].to(torch.int16).cpu()
                
                half_num_gt_occ = gt_x_up_O.shape[0] // 2
                byte_stream_s0 = torchac.encode_int16_normalized_cdf(
                    x_up_O_cdf_norm[:half_num_gt_occ], gt_x_up_O[:half_num_gt_occ]
                )
                byte_stream_s1 = torchac.encode_int16_normalized_cdf(
                    x_up_O_cdf_norm[half_num_gt_occ:], gt_x_up_O[half_num_gt_occ:]
                )
                byte_stream_ls.append(byte_stream_s0)
                byte_stream_ls.append(byte_stream_s1)
            
            base_x_coords, base_x_feats = data_ls[0]
            base_x_coords = base_x_coords[:, 1:].cpu().numpy()
            base_x_feats = base_x_feats.cpu().numpy().astype(np.uint8)
            
            # Decompression phase
            base_x_coords_t = torch.tensor(base_x_coords, device=self.device)
            base_x_feats_t = torch.tensor(base_x_feats.reshape(-1, 1), device=self.device)
            x_dec = SparseTensor(
                coords=torch.cat((base_x_feats_t*0, base_x_coords_t), dim=-1),
                feats=base_x_feats_t
            )
            
            for byte_stream_idx in range(0, len(byte_stream_ls), 2):
                byte_stream_s0 = byte_stream_ls[byte_stream_idx]
                byte_stream_s1 = byte_stream_ls[byte_stream_idx + 1]
                
                x_O = x_dec.feats.int()
                x_dec.feats = self.net.prior_embedding(x_O).view(-1, self.channels)
                x_dec = self.net.prior_resnet(x_dec)
                
                x_up_C, x_up_F = self.net.fcg(x_dec.coords, x_O, x_F=x_dec.feats)
                x_up_C, x_up_F = op.sort_CF(x_up_C, x_up_F)
                x_up_F = self.net.target_embedding(x_up_F, x_up_C)
                x_up = SparseTensor(coords=x_up_C, feats=x_up_F)
                x_up = self.net.target_resnet(x_up)
                
                # Occupancy decoding
                x_up_O_prob_s0 = self.net.pred_head_s0(x_up.feats)
                x_up_O_cdf_s0 = torch.cat((x_up_O_prob_s0[:, 0:1]*0, x_up_O_prob_s0.cumsum(dim=-1)), dim=-1)
                x_up_O_cdf_s0 = torch.clamp(x_up_O_cdf_s0, min=0, max=1)
                x_up_O_cdf_s0_norm = op._convert_to_int_and_normalize(x_up_O_cdf_s0, True)
                x_up_O_cdf_s0_norm = x_up_O_cdf_s0_norm.cpu()
                x_up_O_s0 = torchac.decode_int16_normalized_cdf(x_up_O_cdf_s0_norm, byte_stream_s0).to(self.device)
                
                x_up_O_prob_s1 = self.net.pred_head_s1(
                    x_up.feats + self.net.pred_head_s1_emb(x_up_O_s0.long())
                )
                x_up_O_cdf_s1 = torch.cat((x_up_O_prob_s1[:, 0:1]*0, x_up_O_prob_s1.cumsum(dim=-1)), dim=-1)
                x_up_O_cdf_s1 = torch.clamp(x_up_O_cdf_s1, min=0, max=1)
                x_up_O_cdf_s1_norm = op._convert_to_int_and_normalize(x_up_O_cdf_s1, True)
                x_up_O_cdf_s1_norm = x_up_O_cdf_s1_norm.cpu()
                x_up_O_s1 = torchac.decode_int16_normalized_cdf(x_up_O_cdf_s1_norm, byte_stream_s1).to(self.device)
                
                x_up_O = x_up_O_s1 * 16 + x_up_O_s0
                x_dec = SparseTensor(coords=x_up_C, feats=x_up_O.unsqueeze(-1))
            
            # Inverse transform
            scan = self.net.fcg(x_dec.coords, x_dec.feats)
            scan = (scan[:, 1:] * self.posQ - 131072) * 0.001
            return scan.float().cpu().numpy()


def load_and_compress_pointcloud(bag_path, autoencoder, topic=None, max_messages=None, result_folder='./data/bag_results/'):
    """
    Load a ROS bag file and compress PointCloud2 messages using the autoencoder.
    Saves both original and decompressed point clouds as PLY files.
    
    Args:
        bag_path (str or Path): Path to the .bag file or directory
        autoencoder (PointCloudAutoencoder): Initialized autoencoder instance
        topic (str): Specific topic to filter. If None, processes all PointCloud2 messages.
        max_messages (int): Maximum number of messages to process. If None, process all.
        result_folder (str): Directory to save PLY files
    """
    bag_path = Path(bag_path)
    if not bag_path.exists():
        print(f"Error: Bag file not found: {bag_path}")
        sys.exit(1)
    
    os.makedirs(result_folder, exist_ok=True)
    
    print(f"Loading bag file: {bag_path}")
    if bag_path.is_file():
        print(f"File size: {bag_path.stat().st_size / (1024**2):.2f} MB\n")
    
    try:
        typestore = get_typestore(Stores.LATEST)
        with AnyReader([bag_path], default_typestore=typestore) as reader:
            print("Available topics:")
            topic_info = {}
            for connection in reader.connections:
                if connection.topic not in topic_info:
                    topic_info[connection.topic] = {
                        'type': connection.msgtype,
                        'count': 0
                    }
                topic_info[connection.topic]['count'] += 1
            
            for topic_name, info in sorted(topic_info.items()):
                print(f"  {topic_name}: {info['type']} ({info['count']} messages)")
            print()
            
            # Filter for PointCloud2 messages
            pointcloud_topics = {}
            for topic_name, info in topic_info.items():
                if 'PointCloud2' in info['type']:
                    pointcloud_topics[topic_name] = info['count']
            
            if not pointcloud_topics:
                print("No PointCloud2 messages found in the bag file.")
                return
            
            print(f"Found {len(pointcloud_topics)} PointCloud2 topic(s):")
            for topic_name, count in pointcloud_topics.items():
                print(f"  {topic_name}: {count} messages")
            print()
            
            # Process messages
            message_count = 0
            for connection, timestamp, rawdata in reader.messages():
                if topic and connection.topic != topic:
                    continue
                if 'PointCloud2' not in connection.msgtype:
                    continue
                if max_messages and message_count >= max_messages:
                    break
                
                msg = reader.deserialize(rawdata, connection.msgtype)
                try:
                    # Extract point coordinates
                    points = []
                    for i in range(msg.height * msg.width):
                        point_data = msg.data[i * msg.point_step:(i + 1) * msg.point_step]
                        x, y, z = struct.unpack('fff', point_data[:12])
                        points.append((x, y, z))
                    
                    xyz = np.array(points)
                    print(f"Compressing message {message_count + 1} with {xyz.shape[0]} points...")
                    
                    # Save original point cloud
                    topic_name_clean = connection.topic.replace('/', '_')
                    orig_ply_path = os.path.join(result_folder, f"{topic_name_clean}_{timestamp}_original.ply")
                    io.save_ply_ascii_geo(xyz, orig_ply_path)
                    
                    # Compress and decompress using autoencoder
                    decompressed_xyz = autoencoder.compress(xyz)
                    
                    # Save decompressed point cloud
                    dec_ply_path = os.path.join(result_folder, f"{topic_name_clean}_{timestamp}_decompressed.ply")
                    io.save_ply_ascii_geo(decompressed_xyz, dec_ply_path)
                    
                    print(f"Saved original: {orig_ply_path}")
                    print(f"Saved decompressed: {dec_ply_path}\n")
                    
                except Exception as e:
                    print(f"Error extracting/compressing/decompressing points: {e}\n")
                
                message_count += 1
            
            print(f"{'='*70}")
            print(f"Processed {message_count} PointCloud2 message(s)")
            print(f"{'='*70}")
    
    except Exception as e:
        print(f"Error reading bag file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Compress and decompress pointclouds using NN from bag files."
    )
    parser.add_argument("input_file", type=str, help="Path to the .bag file")
    parser.add_argument("--topic", type=str, default=None, help="Specific topic to filter")
    parser.add_argument("--max-messages", type=int, default=None, help="Maximum number of messages to process")
    parser.add_argument("--output-folder", type=str, default='./data/bag_results/', help="Output folder for PLY files")
    parser.add_argument("--checkpoint", type=str, default='./model/KITTIDetection/ckpt.pt', help="Path to model checkpoint")
    args = parser.parse_args()

    # Initialize autoencoder
    autoencoder = PointCloudAutoencoder(
        checkpoint_path=args.checkpoint,
        channels=32,
        kernel_size=3,
        pos_quantization=16
    )
    
    # Load and compress
    load_and_compress_pointcloud(
        args.input_file,
        autoencoder=autoencoder,
        topic=args.topic,
        max_messages=args.max_messages,
        result_folder=args.output_folder
    )
