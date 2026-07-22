"""Unit tests for depth plane calibration."""
import numpy as np

from utils.mpi.depth_plane import (
    classify_segment_depth,
    depth_tolerance,
    mpi_depth_partition,
    placement_depth_z_star,
)


def test_classify_segment_depth_bands():
    z, tau = 100.0, 5.0
    assert classify_segment_depth(110.0, z, tau) == "front"
    assert classify_segment_depth(102.0, z, tau) == "same"
    assert classify_segment_depth(90.0, z, tau) == "behind"


def test_mpi_partition_uses_depth_max():
    depth = np.full((64, 64), 200.0, dtype=np.float32)
    assert mpi_depth_partition(80, depth) == [(0, 80), (80, 200)]


def test_placement_depth_median_in_mask():
    depth = np.zeros((32, 32), dtype=np.float32)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[10:20, 10:20] = 1
    depth[10:20, 10:20] = 42.0
    assert placement_depth_z_star(depth, mask) == 42.0


def test_depth_tolerance_capped_when_slider_far_from_bbox():
    depth = np.full((64, 64), 50.0, dtype=np.float32)
    mask = np.ones((64, 64), dtype=np.uint8)
    tau = depth_tolerance(depth, mask, z_star=215.0)
    assert tau <= 25.0
