"""Unit tests for occlusion-aware compositing."""
import numpy as np

from utils.mpi.composite import occlusion_aware_composite, reference_alpha_from_dict


def _image_dict_with_alpha(ref_alpha_crop: np.ndarray) -> dict:
    h, w = ref_alpha_crop.shape[:2]
    hint = np.zeros((512, 512, 4), dtype=np.float32)
    hint[:, :, -1] = 1.0
    return {
        "hint": hint,
        "extra_sizes": [h, w, h, w],
        "tar_box_yyxx_crop": [0, h, 0, w],
        "ref_alpha_crop": ref_alpha_crop,
    }


def test_reference_alpha_maps_holes_to_full_image():
    ref_alpha = np.zeros((100, 100), dtype=np.uint8)
    ref_alpha[40:60, 40:60] = 255
    image_dict = _image_dict_with_alpha(ref_alpha)
    full_alpha = reference_alpha_from_dict(image_dict, (100, 100))
    assert full_alpha is not None
    assert full_alpha[50, 50] > 0.9
    assert full_alpha[10, 10] < 0.1


def test_occlusion_composite_shows_original_through_alpha_holes():
    h, w = 120, 120
    original = np.full((h, w, 3), 200, dtype=np.uint8)
    original[:, :] = (255, 0, 0)
    working = original.copy()
    anydoor = original.copy()
    anydoor[:, :] = (0, 255, 0)

    ref_alpha = np.zeros((h, w), dtype=np.uint8)
    ref_alpha[30:90, 30:90] = 255
    ref_alpha[50:70, 50:70] = 0  # hole in object alpha

    image_dict = _image_dict_with_alpha(ref_alpha)
    out = occlusion_aware_composite(
        original,
        working,
        anydoor,
        image_dict,
        save_dir=None,
    )
    assert np.allclose(out[60, 60], (255, 0, 0), atol=2)
    assert np.allclose(out[35, 35], (0, 255, 0), atol=2)
