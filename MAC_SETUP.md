# Mac (Apple Silicon) Setup — Depth-Aware Editing

This guide walks through running Depth-Aware Editing on an Apple Silicon Mac (e.g. M5 Max) using the Mac/MPS patch.

## Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4/M5)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- Git
- Hugging Face account (for model downloads)

## 1. Clone the repository

```bash
cd ~
git clone https://github.com/rishubhpar/Depth-Aware-Editing.git
cd Depth-Aware-Editing
```

If you already have a fork, clone your fork instead and ensure the Mac/MPS patch is applied.

## 2. Create a Mac Python environment

Do **not** use `environment.yml` as-is — it targets Linux + CUDA.

```bash
conda create -n depthedit python=3.9 -y
conda activate depthedit
pip install -r requirements-mac.txt
```

Verify MPS is available:

```bash
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
```

Expected output: `MPS: True`

Optional — enable CPU fallback for unsupported MPS ops:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

Add that line to `~/.zshrc` if you use it regularly.

## 3. Download model weights

```bash
mkdir -p weights
bash download_weights.sh
```

Verify:

```bash
ls -lh weights/
```

You should see:

- `depth_anything_metric_depth_indoor.pt`
- `depth_anything_vitl14.pth`
- `dinov2_vitg14_pretrain.pth`
- `epoch=1-step=8687.ckpt`

Confirm DINOv2 path in `configs/anydoor.yaml`:

```yaml
weight: ./weights/dinov2_vitg14_pretrain.pth
```

## 4. Hugging Face login

The patch uses `Manojb/stable-diffusion-2-1-base` and `facebook/sam-vit-huge`:

```bash
pip install huggingface_hub
huggingface-cli login
```

First run will download several GB of models.

## 5. Run object-placement Gradio demo

```bash
conda activate depthedit
export PYTORCH_ENABLE_MPS_FALLBACK=1
python gradio_demo_op.py
```

Open the URL Gradio prints (usually `http://127.0.0.1:7860`).

**Workflow:**

1. Upload a background image and draw a mask (bbox where the object goes)
2. Upload a reference object image (RGBA with transparent background)
3. Click **Analyze Depth & SAM** — review depth plots and the SAM removal mask preview
4. Adjust depth slider; optionally toggle **Enable clean-plate (Colligo GenFill)**
5. Click **Generate Object Placement**

### Clean-plate (Colligo GenFill) — optional

The Gradio demo can remove intersecting objects inside the placement bbox **before** AnyDoor using Adobe Colligo `v2/images/fill`. This requires corporate network access to the stage endpoint.

Set environment variables (add to `~/.zshrc` if you use this regularly), **or enter host/token directly in the Gradio UI**:

```bash
export COLLIGO_HOST="https://clio-imaging-colligov2-stage.corp.ethos851-stage-or2.ethos.adobe.net"
export COLLIGO_TOKEN="your-bearer-token"
```

Notes:

- Use the **stage root URL** (no `/debug` suffix). The client strips `/debug` automatically if present.
- **VPN:** you must be on the Adobe corporate VPN to reach the stage host.
- If clean-plate is off or `COLLIGO_TOKEN` is unset, standard AnyDoor runs on the original photo (no Telea).
- **Tune Colligo without restarting Gradio:** edit `utils/colligo/runtime_config.json` (similarity, detailLevel) and click Analyze/Generate again.
- **Layered clean-plate** (when GenFill succeeds): SAM segments are classified FRONT / SAME / BEHIND at the depth slider plane; SAME segments are fully GenFill-removed; final image composites cleaned bg + AnyDoor object + FRONT segments restored on top.
- Artifacts per run: `results/object_placement/runs/<timestamp>_*/` — `layer_debug.png`, `removal_mask.png`, `front_mask.png`, `clean_plate_full.jpg`, `composite_layer*.jpg`, `final_composite.jpg`.
- **Null-text inversion** runs fresh every Generate by default (fixes stale cross-image cache). Optional disk cache: `export NULL_TEXT_DISK_CACHE=1`. Speed tunables: `NULL_TEXT_DDIM_STEPS=50`, `NULL_TEXT_INNER_STEPS=5` (default; was 50×5 inner loop). Delete stale cache: `rm examples/Gradio/null_embed/gradio_inference_*`.

**Gate 1 CLI smoke test** (crop + manual mask, no Gradio):

```bash
python scripts/colligo_fill_cli.py path/to/crop.jpg path/to/mask.png path/to/out.jpg
```

**Gate 3 A/B compare** (baseline vs clean-plate on a crop):

```bash
python scripts/clean_plate_ab_compare.py bg.jpg ref.png bbox_mask.png ./ab_out --depth 50
```

### Model caching (AnyDoor, DINOv2, SAM, SD, etc.)

These models are **not image-specific** — they are loaded **once per Gradio process** and reused for every Analyze/Generate:

| Model | Where stored on disk | When loaded |
|-------|----------------------|-------------|
| AnyDoor checkpoint | `weights/epoch=1-step=8687.ckpt` (~16 GB) | Gradio startup |
| DINOv2 ViT-g | `weights/dinov2_vitg14_pretrain.pth` (~4 GB) | Inside AnyDoor |
| Depth Anything (HF) | `~/.cache/huggingface/` | First import of `preprocess.py` |
| SAM ViT-H | `~/.cache/huggingface/` | First import of `preprocess.py` |
| SD 2.1 base | `~/.cache/huggingface/` | `FeatureGuidance` at startup |

**Image-specific** (recomputed each run): null-text inversion embeds, depth/SAM *inference* on your crop, Colligo clean-plate.

Warm all global models once (downloads missing HF weights on first run):

```bash
python scripts/warm_global_models.py
```

Optional: keep HF/Torch caches inside the repo instead of `~/.cache/`:

```bash
export DEPTH_EDIT_MODEL_CACHE=1
./run_gradio.sh
```

Or launch Gradio normally — it still reuses `~/.cache/huggingface` by default (no re-download after first run).

Run artifacts and terminal logs are saved per click under `results/object_placement/runs/<timestamp>_<stage>/` (`run.log`, `run_manifest.json`).

## 6. CLI inference

```bash
python run_inference_object_placement.py
python inference_scene_composition.py
```

Edit paths and parameters in those scripts as described in `readme.md`.

## Known limitations

| Issue | Notes |
|-------|-------|
| `gradio_demo_sc.py` | Not patched; may still hardcode CUDA. Use `inference_scene_composition.py` instead. |
| Speed | Much slower than a 24 GB NVIDIA GPU; first run also downloads models. |
| Memory | DINOv2 ViT-g + AnyDoor + SD-2.x is heavy. Close other apps if you OOM. |
| SAM / depth pipelines | Run on CPU in the patch (expected, slower). |
| xformers / triton | Not installed on Mac (Linux/CUDA only). |

## Troubleshooting

| Error | Fix |
|-------|-----|
| `float64 not supported on MPS` | Search for remaining float64 usage; use float32 |
| `CUDA not available` | Grep for `.cuda()` and replace with `.to(device)` |
| HF 401 / model not found | Run `huggingface-cli login` |
| OOM | Reduce resolution; run one pipeline at a time |

Quick sanity check:

```bash
python -c "import torch; print('device:', 'mps' if torch.backends.mps.is_available() else 'cpu')"
python gradio_demo_op.py
```
