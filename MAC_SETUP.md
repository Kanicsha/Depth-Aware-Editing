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
3. Use depth plots to set the depth slider
4. Click Generate

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
