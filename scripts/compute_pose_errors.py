#!/usr/bin/env python3
"""Script to compute pose errors between two cryoDRGN format pose files.

This script loads two pose pickle files in cryoDRGN format and computes:
- In-plane rotation error (degrees)
- Out-of-plane rotation error (degrees)
- Offset mean error (pixels)

Usage:
    python compute_pose_errors.py poses1.pkl poses2.pkl [--image-size L]
"""

import argparse
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

# Import error computation functions from the existing codebase
from solvar.poses import in_plane_rot_error, offset_mean_error, out_of_plane_rot_error, pose_cryoDRGN2APIRE


def load_cryodrgn_poses(poses_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load poses from a cryoDRGN format pickle file.

    Args:
        poses_file: Path to the poses pickle file

    Returns:
        Tuple of (rotations, offsets) where:
        - rotations: (N, 3, 3) array of rotation matrices
        - offsets: (N, 2) array of translation offsets
    """
    with open(poses_file, "rb") as f:
        poses = pickle.load(f)

    if isinstance(poses, tuple) and len(poses) == 2:
        rotations, offsets = poses
    else:
        raise ValueError(f"Expected poses file to contain tuple of (rotations, offsets), got {type(poses)}")

    # Convert to numpy arrays if they aren't already
    rotations = np.array(rotations)
    offsets = np.array(offsets)

    print(f"Loaded poses from {poses_file}:")
    print(f"  - Rotations shape: {rotations.shape}")
    print(f"  - Offsets shape: {offsets.shape}")

    return rotations, offsets


def parse_pose(input_path: str, output_path: str, image_size: int):

    if input_path.endswith(".pkl"):
        shutil.copy(input_path, output_path)

    elif input_path.endswith(".star"):
        parse_cmd = f"cryodrgn parse_pose_star {input_path} -o {output_path} -D {image_size}"
        subprocess.run(parse_cmd, shell=True, check=True)

    elif input_path.endswith(".cs"):
        parse_cmd = f"cryodrgn parse_pose_csparc {input_path} -o {output_path} -D {image_size}"
        subprocess.run(parse_cmd, shell=True, check=True)

    else:
        raise ValueError(f"Unknown file type of input path {input_path}")


def compute_pose_errors(poses1_file: str, poses2_file: str, global_align: bool = False, image_size: int = None) -> dict:
    """Compute pose errors between two cryoDRGN pose files.

    Args:
        poses1_file: Path to first poses file
        poses2_file: Path to second poses file
        image_size: Image resolution for offset normalization (optional)
        global_align:  Whether to perform global alignment of rotations before computing error

    Returns:
        Dictionary containing error statistics
    """
    # Load both pose files
    print("Loading pose files...")
    rots1, offsets1 = pose_cryoDRGN2APIRE(load_cryodrgn_poses(poses1_file), image_size)
    rots2, offsets2 = pose_cryoDRGN2APIRE(load_cryodrgn_poses(poses2_file), image_size)

    # Check that both files have the same number of poses
    if len(rots1) != len(rots2):
        raise ValueError(f"Pose files have different numbers of poses: {len(rots1)} vs {len(rots2)}")

    if len(offsets1) != len(offsets2):
        raise ValueError(f"Offset files have different numbers of poses: {len(offsets1)} vs {len(offsets2)}")

    print(f"\nComputing pose errors for {len(rots1)} poses...")

    # Convert to torch tensors for error computation
    rots1_tensor = torch.tensor(rots1, dtype=torch.float32)
    rots2_tensor = torch.tensor(rots2, dtype=torch.float32)
    offsets1_tensor = torch.tensor(offsets1, dtype=torch.float32)
    offsets2_tensor = torch.tensor(offsets2, dtype=torch.float32)

    # Compute in-plane rotation error
    print("Computing in-plane rotation errors...")
    in_plane_angles, in_plane_mean, in_plane_median = in_plane_rot_error(
        rots1_tensor, rots2_tensor, global_align=global_align
    )

    # Compute out-of-plane rotation error
    print("Computing out-of-plane rotation errors...")
    out_of_plane_angles, out_of_plane_mean, out_of_plane_median = out_of_plane_rot_error(
        rots1_tensor, rots2_tensor, global_align=global_align
    )

    # Compute offset mean error (in pixels)
    print("Computing offset mean errors...")
    offset_mean = offset_mean_error(offsets1_tensor, offsets2_tensor, L=None)

    # Compute additional statistics
    in_plane_std = np.std(in_plane_angles)
    out_of_plane_std = np.std(out_of_plane_angles)

    # Compute offset statistics (in pixels)
    offset_errors = torch.norm(offsets1_tensor - offsets2_tensor, dim=1)
    offset_std = torch.std(offset_errors).item()
    offset_median = torch.median(offset_errors).item()

    results = {
        "in_plane_rotation": {
            "mean": in_plane_mean,
            "median": in_plane_median,
            "std": in_plane_std,
            "angles": in_plane_angles,
        },
        "out_of_plane_rotation": {
            "mean": out_of_plane_mean,
            "median": out_of_plane_median,
            "std": out_of_plane_std,
            "angles": out_of_plane_angles,
        },
        "offset": {"mean": offset_mean, "median": offset_median, "std": offset_std, "errors": offset_errors.numpy()},
        "image_size": image_size,
        "num_poses": len(rots1),
    }

    return results


def plot_histograms(results: dict, output_path: str = "pose_errors_histogram.png") -> None:
    """Generate histogram subplots of pose errors.

    Args:
        results: Dictionary containing error statistics from compute_pose_errors
        output_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # In-plane rotation errors histogram
    ax1 = axes[0]
    in_plane_angles = results["in_plane_rotation"]["angles"]
    ax1.hist(in_plane_angles, bins=50, edgecolor="black", alpha=0.7)
    ax1.axvline(
        results["in_plane_rotation"]["mean"],
        color="red",
        linestyle="--",
        label=f"Mean: {results['in_plane_rotation']['mean']:.3f}°",
    )
    ax1.axvline(
        results["in_plane_rotation"]["median"],
        color="green",
        linestyle="--",
        label=f"Median: {results['in_plane_rotation']['median']:.3f}°",
    )
    ax1.set_xlabel("In-plane Rotation Error (degrees)")
    ax1.set_ylabel("Frequency")
    ax1.set_yscale("log")
    ax1.set_title("In-plane Rotation Errors")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Out-of-plane rotation errors histogram
    ax2 = axes[1]
    out_of_plane_angles = results["out_of_plane_rotation"]["angles"]
    ax2.hist(out_of_plane_angles, bins=50, edgecolor="black", alpha=0.7, color="orange")
    ax2.axvline(
        results["out_of_plane_rotation"]["mean"],
        color="red",
        linestyle="--",
        label=f"Mean: {results['out_of_plane_rotation']['mean']:.3f}°",
    )
    ax2.axvline(
        results["out_of_plane_rotation"]["median"],
        color="green",
        linestyle="--",
        label=f"Median: {results['out_of_plane_rotation']['median']:.3f}°",
    )
    ax2.set_xlabel("Out-of-plane Rotation Error (degrees)")
    ax2.set_ylabel("Frequency")
    ax2.set_yscale("log")
    ax2.set_title("Out-of-plane Rotation Errors")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Offset errors histogram
    ax3 = axes[2]
    offset_errors = results["offset"]["errors"]
    ax3.hist(offset_errors, bins=50, edgecolor="black", alpha=0.7, color="purple")
    ax3.axvline(
        results["offset"]["mean"], color="red", linestyle="--", label=f"Mean: {results['offset']['mean']:.3f} px"
    )
    ax3.axvline(
        results["offset"]["median"],
        color="green",
        linestyle="--",
        label=f"Median: {results['offset']['median']:.3f} px",
    )
    ax3.set_xlabel("Offset Error (pixels)")
    ax3.set_ylabel("Frequency")
    ax3.set_yscale("log")
    ax3.set_title("Offset Errors")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nHistogram figure saved to: {output_path}")
    plt.close()


def print_results(results: dict) -> None:
    """Print pose error results in a formatted way."""
    print("\n" + "=" * 60)
    print("POSE ERROR ANALYSIS RESULTS")
    print("=" * 60)

    print(f"\nNumber of poses analyzed: {results['num_poses']}")
    if results["image_size"] is not None:
        print(f"Image size for normalization: {results['image_size']} pixels")

    print("\nIN-PLANE ROTATION ERRORS (degrees):")
    print(f"  Mean:   {results['in_plane_rotation']['mean']:.3f}°")
    print(f"  Median: {results['in_plane_rotation']['median']:.3f}°")
    print(f"  Std:    {results['in_plane_rotation']['std']:.3f}°")

    print("\nOUT-OF-PLANE ROTATION ERRORS (degrees):")
    print(f"  Mean:   {results['out_of_plane_rotation']['mean']:.3f}°")
    print(f"  Median: {results['out_of_plane_rotation']['median']:.3f}°")
    print(f"  Std:    {results['out_of_plane_rotation']['std']:.3f}°")

    print("\nOFFSET ERRORS (pixels):")
    print(f"  Mean:   {results['offset']['mean']:.3f} pixels")
    print(f"  Median: {results['offset']['median']:.3f} pixels")
    print(f"  Std:    {results['offset']['std']:.3f} pixels")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Compute pose errors between two cryoDRGN format pose files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compute_pose_errors.py poses1.pkl poses2.pkl
  python compute_pose_errors.py poses1.pkl poses2.pkl --image-size 128
        """,
    )

    parser.add_argument("poses1", help="Path to first poses pickle file")
    parser.add_argument("poses2", help="Path to second poses pickle file")
    parser.add_argument(
        "--image-size",
        "-L",
        type=int,
        default=None,
        help="Image resolution (for reference only, offsets reported in pixels)",
    )
    parser.add_argument(
        "--global-align",
        action="store_true",
        help="Whether to perform global alignment of rotations before computing errors",
    )
    parser.add_argument(
        "--plot-output",
        type=str,
        default=None,
        help="If provided, save histogram figure of pose errors to this path",
    )

    args = parser.parse_args()

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            print("Pasring input files")
            poses1 = os.path.join(temp_dir, "poses1.pkl")
            poses2 = os.path.join(temp_dir, "poses2.pkl")
            parse_pose(args.poses1, poses1, args.image_size)
            parse_pose(args.poses2, poses2, args.image_size)
            # Compute pose errors
            print("Computing pose error")
            results = compute_pose_errors(poses1, poses2, args.global_align, args.image_size)

        # Print results
        print_results(results)

        # Generate histogram figure if output path provided
        if args.plot_output is not None:
            plot_histograms(results, args.plot_output)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
