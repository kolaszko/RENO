#!/usr/bin/env python3
"""
Script to load a ROS bag file, compress PointCloud2 messages, and write them back to a new bag file.
Supports both ROS1 and ROS2 bag files.

Requirements: rosbags, torch, torchsparse, torchac
"""

import sys
import os
import struct
import time
import json
import numpy as np
from pathlib import Path
import torch
import torchac
from torchsparse import SparseTensor
from torchsparse.nn import functional as F

try:
    from rosbags.highlevel import AnyReader
    from rosbags.rosbag1 import Writer as Writer1
    from rosbags.rosbag2 import Writer as Writer2
    from rosbags.typesys import Stores, get_typestore
    from rosbags.typesys.base import TypesysError
except ImportError as e:
    print(f"Error: Missing required package - {e}")
    print("Install with: pip install rosbags")
    sys.exit(1)

from network import Network
import kit.op as op


class PointCloudCompressor:
    """
    Wraps neural network-based compression and decompression for point clouds.
    Handles device setup, model loading, and in-memory compression/decompression.
    """
    
    def __init__(self, checkpoint_path, channels=32, kernel_size=3, pos_quantization=16):
        """
        Initialize the compressor.
        
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
    
    def compress_and_decompress(self, xyz):
        """
        Compress a point cloud and return the decompressed result.
        
        Args:
            xyz (np.ndarray): Point cloud coordinates of shape (N, 3)
            
        Returns:
            np.ndarray: Decompressed point cloud of shape (M, 3)
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


def extract_points_from_pointcloud2(msg):
    """
    Extract XYZ points from a PointCloud2 message.
    
    Args:
        msg: PointCloud2 message
        
    Returns:
        np.ndarray: Array of shape (N, 3) containing XYZ coordinates
    """
    # Convert data to bytes if it's a memoryview or numpy array
    if isinstance(msg.data, np.ndarray):
        data_bytes = msg.data.tobytes()
    elif isinstance(msg.data, memoryview):
        data_bytes = bytes(msg.data)
    else:
        data_bytes = msg.data
    
    points = []
    for i in range(msg.height * msg.width):
        point_data = data_bytes[i * msg.point_step:(i + 1) * msg.point_step]
        x, y, z = struct.unpack('fff', point_data[:12])
        points.append([x, y, z])
    return np.array(points)


def create_pointcloud2_message(xyz, original_msg):
    """
    Create a PointCloud2 message from XYZ coordinates, preserving the original message format.
    
    Args:
        xyz (np.ndarray): Array of shape (N, 3) containing XYZ coordinates
        original_msg: Original PointCloud2 message to copy metadata from
        
    Returns:
        PointCloud2 message with compressed points
    """
    
    # Keep only XYZ fields with proper offsets for compressed point cloud
    new_fields = []
    offset = 0
    for field in original_msg.fields:
        if field.name in ['x', 'y', 'z']:
            # Create new field with updated offset
            new_field = type(field)(
                name=field.name,
                offset=offset,
                datatype=field.datatype,
                count=field.count
            )
            new_fields.append(new_field)
            offset += 4  # Each float is 4 bytes
    
    # Convert to uint8 view
    num_points = xyz.shape[0]
    point_step = 12  # 3 floats × 4 bytes = 12 bytes per point
    data_bytes = xyz.reshape(-1).view(np.uint8)
    
    # Create a new message with XYZ fields only
    new_msg = type(original_msg)(
        header=original_msg.header,
        height=1,  # Unorganized point cloud
        width=num_points,
        fields=new_fields,
        is_bigendian=False,
        point_step=point_step,
        row_step=num_points * point_step,
        data=data_bytes,
        is_dense=False
    )
    
    return new_msg


def compress_and_rewrite_bag(input_bag_path, output_bag_path, compressor, 
                              topic_filter=None, max_messages=None, output_version=None,
                              posq=None, save_statistics=True):
    """
    Load a ROS bag file, compress PointCloud2 messages, and write to a new bag file.
    Supports both ROS1 and ROS2 bag files.
    
    Args:
        input_bag_path (str or Path): Path to the input .bag file or directory
        output_bag_path (str or Path): Path to the output .bag file or directory
        compressor (PointCloudCompressor): Initialized compressor instance
        topic_filter (str or list): Specific topic(s) to compress. If None, compress all PointCloud2 messages.
        max_messages (int): Maximum number of messages to process. If None, process all.
        output_version (int): Output bag version. 2 for ROS1, 8/9 for ROS2. Auto-detected if None.
        posq (int): Position quantization parameter for statistics
        save_statistics (bool): Whether to save statistics to a JSON file
    """
    input_bag_path = Path(input_bag_path)
    output_bag_path = Path(output_bag_path)
    
    if not input_bag_path.exists():
        print(f"Error: Input bag file not found: {input_bag_path}")
        sys.exit(1)
    
    # Ensure output directory exists
    output_bag_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Input bag file: {input_bag_path}")
    if input_bag_path.is_file():
        print(f"Input file size: {input_bag_path.stat().st_size / (1024**2):.2f} MB")
    print(f"Output bag file: {output_bag_path}\n")
    
    # Convert topic_filter to list if it's a string
    if isinstance(topic_filter, str):
        topic_filter = [topic_filter]
    
    try:
        typestore = get_typestore(Stores.LATEST)
        
        with AnyReader([input_bag_path], default_typestore=typestore) as reader:
            # Detect bag format
            is_ros2 = input_bag_path.is_dir() or 'metadata.yaml' in [f.name for f in input_bag_path.parent.glob('*')]
            
            # Auto-detect output version if not specified
            if output_version is None:
                output_version = 9 if is_ros2 else 2
            
            print(f"Detected bag format: {'ROS2' if is_ros2 else 'ROS1'}")
            print(f"Output bag version: {output_version}\n")
            
            # Get bag info
            print("Available topics in input bag:")
            topic_info = {}
            connection_map = {}  # Map topic to connection
            skipped_topics = set()  # Track topics with unknown types
            
            for connection in reader.connections:
                # Check if the message type is known
                try:
                    # Try to get the type definition to verify it exists
                    typestore.types.get(connection.msgtype, None)
                    
                    if connection.topic not in topic_info:
                        topic_info[connection.topic] = {
                            'type': connection.msgtype,
                            'count': 0
                        }
                        connection_map[connection.topic] = connection
                    topic_info[connection.topic]['count'] += 1
                except (TypesysError, KeyError) as e:
                    # Skip topics with unknown message types
                    if connection.topic not in skipped_topics:
                        skipped_topics.add(connection.topic)
                        print(f"  Skipping topic '{connection.topic}': Type '{connection.msgtype}' is unknown")
            
            for topic_name, info in sorted(topic_info.items()):
                print(f"  {topic_name}: {info['type']} ({info['count']} messages)")
            
            if skipped_topics:
                print(f"\nSkipped {len(skipped_topics)} topic(s) with unknown message types.")
            print()
            
            # Identify PointCloud2 topics
            pointcloud_topics = set()
            for topic_name, info in topic_info.items():
                if 'PointCloud2' in info['type']:
                    if topic_filter is None or topic_name in topic_filter:
                        pointcloud_topics.add(topic_name)
            
            if not pointcloud_topics:
                print("No matching PointCloud2 messages found to compress.")
                if topic_filter:
                    print(f"Topic filter: {topic_filter}")
                return
            
            print(f"Will compress {len(pointcloud_topics)} PointCloud2 topic(s):")
            for topic_name in sorted(pointcloud_topics):
                print(f"  {topic_name}")
            print()
            
            # Choose appropriate writer based on output version
            WriterClass = Writer2 if output_version >= 8 else Writer1
            writer_kwargs = {'version': output_version} if output_version >= 8 else {}
            
            # Write output bag
            with WriterClass(output_bag_path, **writer_kwargs) as writer:
                # Add all connections, skipping ones with unknown types
                connection_id_map = {}  # Map original connection id to new connection id
                
                for connection in reader.connections:
                    # Skip connections with unknown types
                    if connection.topic in skipped_topics:
                        continue
                        
                    try:
                        new_conn = writer.add_connection(
                            connection.topic,
                            connection.msgtype,
                            typestore=typestore
                        )
                        connection_id_map[connection.id] = new_conn
                    except (TypesysError, KeyError) as e:
                        print(f"Warning: Cannot add connection for topic '{connection.topic}': {e}")
                        skipped_topics.add(connection.topic)
                        continue
                
                # Process all messages
                processed_count = 0
                compressed_count = 0
                compression_times = []  # Track compression times
                original_point_counts = []  # Track original point counts
                compressed_point_counts = []  # Track compressed point counts
                per_message_stats = []  # Track per-message statistics
                
                for connection, timestamp, rawdata in reader.messages():
                    # Skip messages from topics with unknown types
                    if connection.topic in skipped_topics or connection.id not in connection_id_map:
                        continue
                    
                    # Check if we've reached max_messages
                    if max_messages and compressed_count >= max_messages:
                        # Deserialize and re-serialize remaining messages
                        try:
                            msg = reader.deserialize(rawdata, connection.msgtype)
                            new_rawdata = typestore.serialize_ros1(msg, connection.msgtype)
                            writer.write(connection_id_map[connection.id], timestamp, new_rawdata)
                        except (TypesysError, KeyError) as e:
                            print(f"Warning: Cannot deserialize message from topic '{connection.topic}': {e}")
                        continue
                    
                    # Check if this is a PointCloud2 message to compress
                    if connection.topic in pointcloud_topics and 'PointCloud2' in connection.msgtype:
                        try:
                            # Deserialize message
                            msg = reader.deserialize(rawdata, connection.msgtype)
                            
                            # Extract points
                            xyz = extract_points_from_pointcloud2(msg)
                            original_points = xyz.shape[0]
                            print(f"Compressing {connection.topic} (msg {compressed_count + 1}): {original_points} points...", end=' ')
                            
                            # Compress and decompress with timing
                            start_time = time.time()
                            compressed_xyz = compressor.compress_and_decompress(xyz)
                            compression_time = time.time() - start_time
                            compressed_points = compressed_xyz.shape[0]
                            
                            compression_times.append(compression_time)
                            original_point_counts.append(original_points)
                            compressed_point_counts.append(compressed_points)
                            
                            # Track per-message statistics
                            per_message_stats.append({
                                'message_index': compressed_count + 1,
                                'topic': connection.topic,
                                'timestamp': timestamp,
                                'original_points': original_points,
                                'compressed_points': compressed_points,
                                'compression_ratio': original_points / compressed_points if compressed_points > 0 else 0,
                                'compression_time': compression_time
                            })
                            
                            print(f"→ {compressed_points} points ({compression_time:.4f}s)")
                            
                            # Create new message with compressed points
                            new_msg = create_pointcloud2_message(compressed_xyz, msg)
                            
                            # Serialize and write
                            new_rawdata = typestore.serialize_ros1(new_msg, connection.msgtype)
                            writer.write(connection_id_map[connection.id], timestamp, new_rawdata)
                            
                            compressed_count += 1
                            
                        except Exception as e:
                            print(f"\nWarning: Error compressing message from {connection.topic}: {e}")
                            print("Writing original message instead.")
                            try:
                                msg = reader.deserialize(rawdata, connection.msgtype)
                                new_rawdata = typestore.serialize_ros1(msg, connection.msgtype)
                                writer.write(connection_id_map[connection.id], timestamp, new_rawdata)
                            except (TypesysError, KeyError) as e:
                                print(f"Warning: Cannot deserialize/write original message from topic '{connection.topic}': {e}")
                    else:
                        # Deserialize and re-serialize non-PointCloud2 messages
                        try:
                            msg = reader.deserialize(rawdata, connection.msgtype)
                            new_rawdata = typestore.serialize_ros1(msg, connection.msgtype)
                            writer.write(connection_id_map[connection.id], timestamp, new_rawdata)
                        except (TypesysError, KeyError) as e:
                            print(f"Warning: Cannot deserialize message from topic '{connection.topic}': {e}")
                    
                    processed_count += 1
            
            print(f"\n{'='*70}")
            print(f"Processed {processed_count} total messages")
            print(f"Compressed {compressed_count} PointCloud2 messages")
            
            # Calculate and display timing statistics
            statistics = {
                'input_bag': str(input_bag_path),
                'output_bag': str(output_bag_path),
                'total_messages_processed': processed_count,
                'pointcloud_messages_compressed': compressed_count,
                'posq_parameter': posq,
                'compression_settings': {
                    'channels': compressor.channels,
                    'kernel_size': compressor.kernel_size,
                    'pos_quantization': compressor.posQ
                }
            }
            
            if compression_times:
                mean_time = np.mean(compression_times)
                std_time = np.std(compression_times)
                total_time = np.sum(compression_times)
                
                # Point count statistics
                total_original_points = np.sum(original_point_counts)
                total_compressed_points = np.sum(compressed_point_counts)
                mean_original_points = np.mean(original_point_counts)
                mean_compressed_points = np.mean(compressed_point_counts)
                
                print(f"\nCompression/Decompression Timing Statistics:")
                print(f"  Total time: {total_time:.4f}s")
                print(f"  Mean time per message: {mean_time:.4f}s")
                print(f"  Std deviation: {std_time:.4f}s")
                print(f"  Min time: {np.min(compression_times):.4f}s")
                print(f"  Max time: {np.max(compression_times):.4f}s")
                
                print(f"\nPoint Count Statistics:")
                print(f"  Total original points: {total_original_points}")
                print(f"  Total compressed points: {total_compressed_points}")
                print(f"  Mean original points per message: {mean_original_points:.2f}")
                print(f"  Mean compressed points per message: {mean_compressed_points:.2f}")
                print(f"  Overall compression ratio: {total_original_points / total_compressed_points:.4f}" if total_compressed_points > 0 else "  Overall compression ratio: N/A")
                
                # Add to statistics dictionary
                statistics['timing'] = {
                    'total_time': total_time,
                    'mean_time': mean_time,
                    'std_time': std_time,
                    'min_time': float(np.min(compression_times)),
                    'max_time': float(np.max(compression_times))
                }
                statistics['point_counts'] = {
                    'total_original_points': int(total_original_points),
                    'total_compressed_points': int(total_compressed_points),
                    'mean_original_points': mean_original_points,
                    'mean_compressed_points': mean_compressed_points,
                    'overall_compression_ratio': float(total_original_points / total_compressed_points) if total_compressed_points > 0 else None
                }
                statistics['per_message_stats'] = per_message_stats
            
            print(f"\nOutput saved to: {output_bag_path}")
            
            # Get file sizes
            input_size_mb = None
            output_size_mb = None
            
            if input_bag_path.exists():
                if input_bag_path.is_file():
                    input_size_mb = input_bag_path.stat().st_size / (1024**2)
                elif input_bag_path.is_dir():
                    input_size_mb = sum(f.stat().st_size for f in input_bag_path.rglob('*') if f.is_file()) / (1024**2)
            
            if output_bag_path.exists():
                if output_bag_path.is_file():
                    output_size_mb = output_bag_path.stat().st_size / (1024**2)
                    print(f"Output file size: {output_size_mb:.2f} MB")
                elif output_bag_path.is_dir():
                    output_size_mb = sum(f.stat().st_size for f in output_bag_path.rglob('*') if f.is_file()) / (1024**2)
                    print(f"Output directory size: {output_size_mb:.2f} MB")
            
            # Add file size info to statistics
            statistics['file_sizes'] = {
                'input_size_mb': input_size_mb,
                'output_size_mb': output_size_mb,
                'size_reduction_ratio': input_size_mb / output_size_mb if (input_size_mb and output_size_mb and output_size_mb > 0) else None
            }
            
            # Save statistics to JSON file if requested
            if save_statistics:
                stats_path = output_bag_path.parent / 'statistics.json'
                try:
                    with open(stats_path, 'w') as f:
                        json.dump(statistics, f, indent=2, default=str)
                    print(f"Statistics saved to: {stats_path}")
                except Exception as e:
                    print(f"Warning: Could not save statistics: {e}")
            
            print(f"{'='*70}")
            
            return statistics
    
    except Exception as e:
        print(f"Error processing bag file: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compress PointCloud2 messages in ROS bag files (ROS1/ROS2 compatible)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_bag", type=str, help="Path to the input .bag file or directory")
    parser.add_argument("output_bag", type=str, help="Path to the output .bag file or directory")
    parser.add_argument("--topic", type=str, nargs='+', default=None, 
                        help="Specific topic(s) to compress (space-separated). If not specified, all PointCloud2 topics are compressed.")
    parser.add_argument("--max-messages", type=int, default=None, 
                        help="Maximum number of PointCloud2 messages to compress")
    parser.add_argument("--checkpoint", type=str, default='./model/KITTIDetection/ckpt.pt', 
                        help="Path to model checkpoint")
    parser.add_argument("--channels", type=int, default=32, help="Neural network channels")
    parser.add_argument("--kernel-size", type=int, default=3, help="Convolution kernel size")
    parser.add_argument("--pos-quantization", type=int, default=16, help="Position quantization scale")
    parser.add_argument("--output-version", type=int, default=None, 
                        help="Output bag version (2 for ROS1, 8 or 9 for ROS2). Auto-detected if not specified.")
    
    args = parser.parse_args()
    
    # Initialize compressor
    print(f"Loading model from: {args.checkpoint}")
    compressor = PointCloudCompressor(
        checkpoint_path=args.checkpoint,
        channels=args.channels,
        kernel_size=args.kernel_size,
        pos_quantization=args.pos_quantization
    )
    print(f"Using device: {compressor.device}\n")
    
    # Setup experiment directory structure
    input_bag_path = Path(args.input_bag)
    
    # Get the directory where the input bag file is located
    if input_bag_path.is_file():
        bag_dir = input_bag_path.parent
        bag_name = input_bag_path.stem  # Get filename without extension
    else:
        bag_dir = input_bag_path.parent
        bag_name = input_bag_path.name
    
    # Create reno directory in the same location as the input bag
    reno_dir = bag_dir / 'reno'
    reno_dir.mkdir(exist_ok=True)
    
    # Create subdirectory based on bag name and posq parameter
    experiment_dir_name = f"{bag_name}_posq{args.pos_quantization}"
    experiment_dir = reno_dir / experiment_dir_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Create output bag path with new naming scheme
    output_bag_name = f"{bag_name}_posq{args.pos_quantization}.bag"
    output_bag_path = experiment_dir / output_bag_name
    
    print(f"Experiment directory: {experiment_dir}")
    print(f"Output bag will be saved as: {output_bag_name}\n")
    
    # Process bag file
    compress_and_rewrite_bag(
        input_bag_path=args.input_bag,
        output_bag_path=str(output_bag_path),
        compressor=compressor,
        topic_filter=args.topic,
        max_messages=args.max_messages,
        output_version=args.output_version,
        posq=args.pos_quantization,
        save_statistics=True
    )
