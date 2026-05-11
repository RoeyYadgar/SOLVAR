#!/usr/bin/env python3
"""Script to match coordinates between two analysis datasets.

This script takes two analysis data paths (pickle files containing dictionaries with 'coords' key)
and a third pickle file containing a numpy array. It finds the closest coordinates in the first
dataset to each point in the numpy array, then returns the corresponding coordinates from the
second dataset.

Usage:
    python match_coordinates.py <analysis_data1.pkl> <analysis_data2.pkl> <query_coords.pkl> [output.pkl]
"""

import argparse
import pickle
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist


def load_analysis_data(filepath: str) -> Dict[str, Any]:
    """Load analysis data from pickle file.

    Args:
        filepath: Path to pickle file containing analysis data

    Returns:
        Dictionary containing analysis data

    Raises:
        FileNotFoundError: If file doesn't exist
        KeyError: If 'coords' key is missing
    """
    try:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Analysis data file not found: {filepath}")

    if "coords" in data:
        coords = data["coords"]
    elif "coords_est" in data:
        coords = data["coords_est"]
    else:
        raise KeyError(f"'coords' or 'coods_est' key not found in analysis data: {filepath}")

    return data, coords


def load_query_coords(filepath: str) -> np.ndarray:
    """Load query coordinates from pickle file.

    Args:
        filepath: Path to pickle file containing numpy array

    Returns:
        Numpy array of query coordinates

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If loaded data is not a numpy array
    """
    try:
        with open(filepath, "rb") as f:
            coords = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Query coordinates file not found: {filepath}")

    if not isinstance(coords, np.ndarray):
        raise ValueError(f"Query coordinates must be a numpy array, got {type(coords)}")

    return coords


def find_closest_coordinates(
    query_coords: np.ndarray, reference_coords: np.ndarray, num_neighbors: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """Find closest coordinates in reference dataset for each query point.

    Args:
        query_coords: Query coordinates (n x dim)
        reference_coords: Reference coordinates (m x dim)

    Returns:
        Tuple of (closest_indices, distances) where:
        - closest_indices: Array of indices in reference_coords for each query point
        - distances: Array of distances to closest points
    """
    if query_coords.shape[1] != reference_coords.shape[1]:
        raise ValueError(
            f"Coordinate dimensions must match: query {query_coords.shape[1]}, "
            f"reference {reference_coords.shape[1]}"
        )

    # Compute pairwise distances between query and reference coordinates
    distances = cdist(query_coords, reference_coords, metric="euclidean")

    # Find closest indices for each query point
    # Use np.argpartition for efficiency to get indices of k nearest neighbors
    closest_indices = np.argpartition(distances, range(num_neighbors), axis=1)[:, :num_neighbors]
    # Sort these k neighbors for each query point based on the actual distances
    row_indices = np.arange(distances.shape[0])[:, None]
    sorted_order = np.argsort(distances[row_indices, closest_indices], axis=1)
    closest_indices = closest_indices[row_indices, sorted_order]
    min_distances = distances[row_indices, closest_indices]

    return closest_indices, min_distances


def match_coordinates(
    analysis_data1_path: str,
    analysis_data2_path: str,
    query_coords_path: str,
    output_path: Optional[str] = None,
    num_neighbors: int = 1,
) -> Dict[str, Any]:
    """Match coordinates between two analysis datasets.

    Args:
        analysis_data1_path: Path to first analysis data (reference)
        analysis_data2_path: Path to second analysis data (target)
        query_coords_path: Path to query coordinates pickle file
        output_path: Optional output path for results

    Returns:
        Dictionary containing matching results
    """
    print(f"Loading analysis data from: {analysis_data1_path}")
    _, coords1 = load_analysis_data(analysis_data1_path)

    print(f"Loading analysis data from: {analysis_data2_path}")
    _, coords2 = load_analysis_data(analysis_data2_path)

    print(f"Loading query coordinates from: {query_coords_path}")
    query_coords = load_query_coords(query_coords_path)

    print(f"Reference coords shape: {coords1.shape}")
    print(f"Target coords shape: {coords2.shape}")
    print(f"Query coords shape: {query_coords.shape}")

    # Find closest coordinates
    print("Finding closest coordinates...")
    closest_indices, distances = find_closest_coordinates(query_coords, coords1, num_neighbors)

    # Get corresponding coordinates from second dataset
    matched_coords = coords2[closest_indices].mean(axis=1)

    # Prepare results
    results = {
        "query_coords": query_coords,
        "reference_coords": coords1,
        "target_coords": coords2,
        "closest_indices": closest_indices,
        "distances": distances,
        "matched_coords": matched_coords,
        "analysis_data1_path": analysis_data1_path,
        "analysis_data2_path": analysis_data2_path,
        "query_coords_path": query_coords_path,
    }

    # Save matched coordinates if output path provided
    if output_path:
        print(f"Saving matched coordinates to: {output_path}")
        with open(output_path, "wb") as f:
            pickle.dump(matched_coords, f)

    # Print summary
    print("\nMatching Summary:")
    print(f"  Query points: {len(query_coords)}")
    print(f"  Reference points: {len(coords1)}")
    print(f"  Target points: {len(coords2)}")
    print(f"  Mean distance to closest: {np.mean(distances):.4f}")
    print(f"  Max distance to closest: {np.max(distances):.4f}")
    print(f"  Min distance to closest: {np.min(distances):.4f}")

    return results


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Match coordinates between two analysis datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python match_coordinates.py data1.pkl data2.pkl query.pkl
    python match_coordinates.py data1.pkl data2.pkl query.pkl output.pkl
        """,
    )

    parser.add_argument("analysis_data1", help="Path to first analysis data pickle file (reference dataset)")
    parser.add_argument("analysis_data2", help="Path to second analysis data pickle file (target dataset)")
    parser.add_argument("query_coords", help="Path to query coordinates pickle file (numpy array)")
    parser.add_argument(
        "output", nargs="?", help="Optional output path for matched coordinates pickle file (numpy array)"
    )
    parser.add_argument(
        "--num_neighbors",
        type=int,
        default=1000,
        help="Number of nearest neighbors to match (default: 1)",
    )

    args = parser.parse_args()

    try:
        match_coordinates(args.analysis_data1, args.analysis_data2, args.query_coords, args.output, args.num_neighbors)

        print("\nMatching completed successfully!")
        if args.output:
            print(f"Matched coordinates saved to: {args.output}")
        else:
            print("Matched coordinates returned (not saved to file)")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
