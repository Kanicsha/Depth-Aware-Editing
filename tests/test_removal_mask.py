"""Unit tests for layered SAM removal mask helper."""
import os

import numpy as np
from utils.mpi.removal_mask import (
    anydoor_edit_mask,
    full_bbox_removal_mask,
    genfill_mode,
    patch_image_dict_anydoor_hint,
    resolve_genfill_removal_mask,
    sam_intersection_removal_mask,
    sam_layered_removal_masks,
)


def _fake_image_dict(placement_mask: np.ndarray) -> dict:
    hint = np.zeros((*placement_mask.shape, 4), dtype=np.float32)
    hint[:, :, -1] = (placement_mask > 0).astype(np.float32)
    return {
        "hint": hint,
        "extra_sizes": [512, 512, 512, 512],
        "tar_box_yyxx_crop": [0, 512, 0, 512],
    }


def _layered_fixture():
    labels = np.zeros((512, 512), dtype=np.int32)
    labels[100:180, 100:180] = 1
    labels[120:200, 120:200] = 2
    placement = np.zeros((512, 512), dtype=np.uint8)
    placement[80:220, 80:220] = 1

    depth = np.full((512, 512), 80.0, dtype=np.float32)
    depth[labels == 1] = 100.0
    depth[labels == 2] = 50.0

    image_dict = _fake_image_dict(placement)
    same, front, behind, debug = sam_layered_removal_masks(
        labels,
        image_dict,
        depth_map=depth,
        z_star=100.0,
        tau=5.0,
        border_floor_exclude_ratio=0.99,
    )
    return image_dict, same, front, behind, debug


def test_same_segment_marked_at_bbox_intersection_only():
    labels = np.zeros((512, 512), dtype=np.int32)
    labels[100:200, 100:200] = 1
    labels[400:480, 400:480] = 2

    placement = np.zeros((512, 512), dtype=np.uint8)
    placement[150:250, 150:250] = 1

    depth = np.full((512, 512), 100.0, dtype=np.float32)
    depth[labels == 1] = 100.0

    same, front, behind, debug = sam_layered_removal_masks(
        labels,
        _fake_image_dict(placement),
        depth_map=depth,
        z_star=100.0,
        tau=5.0,
        border_floor_exclude_ratio=0.99,
    )

    assert 1 in debug["kept_same_segment_ids"]
    assert same[175, 175] == 255
    assert same[110, 110] == 0
    assert same[440, 440] == 0


def test_behind_segment_excluded_from_removal():
    image_dict, same, front, behind, debug = _layered_fixture()

    assert 1 in debug["kept_same_segment_ids"]
    assert 2 in debug["kept_behind_segment_ids"]
    assert behind[150, 150] == 255
    assert same[150, 150] == 0
    assert int((same > 0).sum()) > 0


def test_front_segment_in_front_mask_not_removal():
    labels = np.zeros((512, 512), dtype=np.int32)
    labels[100:180, 100:180] = 1
    placement = np.zeros((512, 512), dtype=np.uint8)
    placement[80:220, 80:220] = 1

    depth = np.full((512, 512), 100.0, dtype=np.float32)
    depth[labels == 1] = 110.0

    same, front, behind, debug = sam_layered_removal_masks(
        labels,
        _fake_image_dict(placement),
        depth_map=depth,
        z_star=100.0,
        tau=5.0,
        border_floor_exclude_ratio=0.99,
    )

    assert 1 in debug["kept_front_segment_ids"]
    assert 1 not in debug.get("kept_same_segment_ids", [])
    assert int((same > 0).sum()) == 0
    assert front[140, 140] == 255


def test_wrapper_returns_legacy_keys():
    labels = np.zeros((512, 512), dtype=np.int32)
    labels[100:200, 100:200] = 1
    placement = np.zeros((512, 512), dtype=np.uint8)
    placement[50:250, 50:250] = 1

    removal, debug = sam_intersection_removal_mask(
        labels,
        _fake_image_dict(placement),
        depth_map=np.full((512, 512), 100.0),
        depth_value=100,
        border_floor_exclude_ratio=0.99,
    )
    assert "kept_segment_ids" in debug
    assert removal[150, 150] == 255


def test_full_bbox_removal_covers_placement_minus_front():
    placement = np.zeros((512, 512), dtype=np.uint8)
    placement[100:300, 150:350] = 1
    front = np.zeros((512, 512), dtype=np.uint8)
    front[120:180, 160:220] = 255
    image_dict = _fake_image_dict(placement)

    removal = full_bbox_removal_mask(image_dict, (512, 512), front_mask=front)

    assert removal[200, 200] == 255
    assert removal[150, 180] == 0
    assert removal[50, 50] == 0


def test_genfill_mode_defaults_same_intersection():
    old = os.environ.pop("GENFILL_MODE", None)
    try:
        assert genfill_mode() == "same_intersection"
    finally:
        if old is not None:
            os.environ["GENFILL_MODE"] = old


def test_genfill_mode_aliases():
    cases = {
        "c3": "same_and_behind_intersection",
        "e2": "behind_intersection",
        "f3": "f3",
        "same_only": "same_intersection",
        "full_bbox": "full_bbox",
    }
    old = os.environ.get("GENFILL_MODE")
    for raw, expected in cases.items():
        os.environ["GENFILL_MODE"] = raw
        assert genfill_mode() == expected
    if old is None:
        os.environ.pop("GENFILL_MODE", None)
    else:
        os.environ["GENFILL_MODE"] = old


def test_resolve_c3_same_and_behind_union():
    image_dict, same, front, behind, _ = _layered_fixture()
    removal, meta = resolve_genfill_removal_mask(
        "same_and_behind_intersection",
        image_dict,
        (512, 512),
        same,
        behind,
        front,
    )
    assert meta["genfill_mode"] == "same_and_behind_intersection"
    assert removal[150, 150] == 255
    assert int((removal > 0).sum()) > int((same > 0).sum())


def test_resolve_e2_behind_only():
    image_dict, same, front, behind, _ = _layered_fixture()
    removal, meta = resolve_genfill_removal_mask(
        "behind_intersection",
        image_dict,
        (512, 512),
        same,
        behind,
        front,
    )
    assert meta["genfill_mode"] == "behind_intersection"
    assert removal[150, 150] == 255
    assert int((same > 0).sum()) > 0
    assert int((removal > 0).sum()) > 0


def test_resolve_f3_same_only_and_anydoor_hint_patch():
    image_dict, same, front, behind, _ = _layered_fixture()
    removal, meta = resolve_genfill_removal_mask(
        "f3",
        image_dict,
        (512, 512),
        same,
        behind,
        front,
    )
    assert meta["anydoor_hint_excludes_behind"] is True
    assert removal[150, 150] == 0
    assert int((removal > 0).sum()) > 0

    edit = anydoor_edit_mask(image_dict, (512, 512), behind_mask=behind)
    assert edit[150, 150] == 0
    assert edit[100, 100] > 0

    patched = patch_image_dict_anydoor_hint(image_dict, edit)
    assert patched["hint"][150, 150, -1] == 0
    assert patched["hint"][100, 100, -1] > 0
