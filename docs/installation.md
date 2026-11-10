# Installation

> Source of truth: `install.py`, `README.md` (Installation section),
> `requirements.txt`, `pyproject.toml`, `.gitmodules`.

GaussianWrapping is a CUDA-heavy research codebase: it compiles **six native
extensions** (four Gaussian rasterizer variants, Simple KNN, Fused SSIM) plus a
C++/CGAL **tetrahedral triangulation** tool, **nvdiffrast**, and **warp-patch-ncc**.
The bundled `install.py` automates all of this for a Conda environment.

## Requirements

| Requirement | Value |
|-------------|-------|
| OS | Linux (CUDA toolchain assumed at `/usr/local/cuda-<version>`) |
| Python | 3.9 (Conda env `gaussian_wrapping`) |
| CUDA | 11.8 or 12.1 (`--cuda_version` choice of `install.py`; 12.1 recommended for RTX 4090) |
| PyTorch | 2.3.1 (+ torchvision 0.18.1, torchaudio 2.3.1, `pytorch-cuda`, `mkl=2023.1.0`) |
| Build tools | `cmake`, `gmp`, `cgal` (installed into the env by `install.py` for `tetra_triangulation`) |
| Package manager | Conda (the script shells out to `conda install`) |

## Step-by-step install

```bash
git clone --recurse-submodules https://github.com/diego1401/GaussianWrapping
cd GaussianWrapping
conda create -n gaussian_wrapping python=3.9 -y
conda activate gaussian_wrapping

# Set CUDA paths (adjust version as needed)
CUDA_VERSION=12.1   # or 11.8 (12.1 is best for 4090RTX)

export CPATH=/usr/local/cuda-${CUDA_VERSION}/targets/x86_64-linux/include:$CPATH
export LD_LIBRARY_PATH=/usr/local/cuda-${CUDA_VERSION}/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-${CUDA_VERSION}/bin:$PATH

python install.py --cuda_version ${CUDA_VERSION}
```

`--cuda_version` accepts `11.8` or `12.1` (default `11.8`). It selects the
`pytorch-cuda=<version>` build **and** the matching PyG wheel index
(`https://data.pyg.org/whl/torch-2.1.0+cu<version>.html`).

> The CUDA environment variables must be exported **before** `install.py` runs:
> the `tetra_triangulation` step invokes `cmake`, which needs the CUDA paths.

## What `install.py` does, in order

| Step | Command | Purpose |
|------|---------|---------|
| 1 | `conda install -y pip=22.3.1` | Pin pip |
| 2 | `conda install -y pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=<v> mkl=2023.1.0 -c pytorch -c nvidia` | PyTorch stack |
| 3 | `pip install -r requirements.txt` | Python dependencies (table below) |
| 4 | `pip install --no-build-isolation submodules/diff-gaussian-rasterization_ms` | Mini-Splatting2 rasterizer |
| 5 | `pip install --no-build-isolation submodules/diff-gaussian-rasterization` | RaDe-GS rasterizer |
| 6 | `pip install --no-build-isolation submodules/diff-gaussian-rasterization_ours` | "Ours" (median-depth) rasterizer |
| 7 | `pip install --no-build-isolation submodules/diff-gaussian-rasterization_sof` | SOF rasterizer |
| 8 | `pip install --no-build-isolation submodules/simple-knn` | Simple KNN (initial scaling) |
| 9 | `pip install --no-build-isolation submodules/fused-ssim` | Fused SSIM loss |
| 10 | `cd submodules/tetra_triangulation` → `conda install cmake`, `gmp`, `cgal` → `cmake . -DCMAKE_POLICY_VERSION_MINIMUM=3.5` → `make` → `pip install -e .` | Delaunay tetrahedralization used by mesh extraction (`-DCMAKE_POLICY_VERSION_MINIMUM=3.5` needed for RTX 4090-era CMake) |
| 11 | `cd submodules/nvdiffrast` → `pip install --no-build-isolation -e .` | Differentiable rasterizer (mesh rendering) |
| 12 | `pip install --no-build-isolation submodules/Geometry-Grounded-Gaussian-Splatting/submodules/warp-patch-ncc` | NCC patch operator used by multiview regularization |
| 13 | `pip install torch_geometric` + `pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu<v>.html` | Torch Geometric stack |

## Python dependencies (`requirements.txt`)

| Package | Pinned version | Role |
|---------|----------------|------|
| open3d | 0.19.0 | Mesh I/O / geometry processing |
| trimesh | 4.6.8 | Mesh I/O |
| scikit-image | 0.24.0 | Image utilities |
| opencv-python | 4.11.0.86 | Image utilities |
| plyfile | 1.1 | PLY point-cloud/mesh I/O |
| tqdm | 4.67.1 | Progress bars |
| wandb | (latest) | Optional experiment tracking |
| einops | (latest) | Tensor reshaping |
| scikit-learn | (latest) | Misc. ML utilities |
| dacite | (latest) | Config parsing helpers |

`pyproject.toml` repeats these as `>=` constraints (`open3d>=0.19.0`,
`trimesh>=4.6.8`, …), so `pip install -e .` works without the pinned file.

## Installing the package itself

After the submodules are built, install the package (editable install keeps the
YAML configs and console scripts importable):

```bash
pip install -e .
```

This registers two console scripts (from `pyproject.toml`):

| Script | Target | Status |
|--------|--------|--------|
| `gw-train` | `gaussian_wrapping.train:main` | Works |
| `gw-render` | `gaussian_wrapping.render:main` | **Broken** — `render.py` defines no `main()`; use `python -m gaussian_wrapping.render` instead (see [cli-reference.md](cli-reference.md)) |

## Git submodules (`.gitmodules`)

| Submodule | Path | Used by |
|-----------|------|---------|
| [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2.git) | `submodules/Depth-Anything-V2` | `--depth_order` regularizer (learned monocular depth prior) |
| [nvdiffrast](https://github.com/NVlabs/nvdiffrast.git) | `submodules/nvdiffrast` | Mesh rendering (MILo, texture refinement) |
| [TNTUniScanEvals](https://github.com/diego1401/TNTUniScanEvals.git) | `gaussian_wrapping/eval/TNTUniScanEvals` | Tanks and Temples evaluation; also imported by `primal_adaptive_meshing_extraction.py` for ground-truth bounding volumes |
| [Geometry-Grounded-Gaussian-Splatting](https://github.com/HKUST-SAIL/Geometry-Grounded-Gaussian-Splatting.git) | `submodules/Geometry-Grounded-Gaussian-Splatting` | Provides `warp-patch-ncc` (multiview NCC regularization) |

If you cloned without `--recurse-submodules`, run:

```bash
git submodule update --init --recursive
```

## Verification

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "from diff_gaussian_rasterization import _C; print('radegs rasterizer OK')"   # may differ per variant
python -c "import fused_ssim; print('fused_ssim OK')"
python -c "from gaussian_wrapping.api import TrainingConfig, run_training; print('API OK')"
gw-train --help
```

A training smoke test (tiny budget):

```bash
gw-train -s <DATASET> -m /tmp/smoke --iterations 100 --densify_until_iter 50
```

See [training-guide.md](training-guide.md) for what the run produces.

## Common install problems

| Symptom | Cause | Fix |
|---------|-------|-----|
| `cmake` fails in `tetra_triangulation` | CUDA paths not exported, or new CMake policy error | Export `CPATH`/`LD_LIBRARY_PATH`/`PATH` first; keep `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` (already in `install.py`) |
| `ModuleNotFoundError: diff_gaussian_rasterization` | Rasterizer submodule not built | Re-run the matching `pip install --no-build-isolation submodules/...` line |
| `ModuleNotFoundError: fused_ssim` | Fused SSIM not built | `pip install --no-build-isolation submodules/fused-ssim` |
| PyG wheel install fails | Wheel index mismatch with CUDA version | Use the exact `-f https://data.pyg.org/whl/torch-2.1.0+cu<118|121>.html` index matching `--cuda_version` |
| `ImportError` on `gaussian_wrapping.eval.TNTUniScanEvals` | Submodule not initialized | `git submodule update --init --recursive` |
| `wandb` prompts for login at training start | WandB installed but not configured | `wandb disabled`, or leave `--wandb_project` unset (training skips WandB when project/entity are `None`) |
