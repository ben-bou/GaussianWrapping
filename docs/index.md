# GaussianWrapping — Documentation Index

GaussianWrapping is a 3D Gaussian Splatting (3DGS) research codebase that trains
Gaussian scenes and wraps them into high-quality meshes. It combines:

- **3DGS training** with several rasterizers (RaDe-GS, a median-depth "ours"
  rasterizer, Mini-Splatting2, SOF),
- **regularizers** that shape geometry: a learnable *normal field*, multiview
  NCC-style consistency, mesh-in-the-loop (MILo) optimization, depth-normal
  consistency, and an optional learned depth-order prior (Depth-Anything-V2),
- **mesh extraction** via two pipelines: *pivot-based* Delaunay/marching-tetrahedra
  extraction and *Primal Adaptive Meshing* (PAM),
- **benchmarks** for Tanks and Temples, DTU, MipNeRF360 and NeRF Synthetic.

The package version is `1.0.0` (`pyproject.toml`, `gaussian_wrapping.__version__`).

> This documentation set was generated from the source code in this repository.
> Every default, flag and file name below was transcribed from the code — when
> in doubt, the linked source file is the authority.

## Documentation map

| Document | Contents |
|----------|----------|
| [installation.md](installation.md) | Environment, dependencies, CUDA rasterizer submodules, step-by-step install, verification |
| [cli-reference.md](cli-reference.md) | Every CLI flag of every entry point (`gw-train`, `convert`, mesh extraction, evaluation, metrics, …) |
| [api-reference.md](api-reference.md) | Python API: `TrainingConfig` (all 78 fields), `run_training`, `TrainingResult`, `GaussianWrappingScene` |
| [training-guide.md](training-guide.md) | What a training run does end to end: losses, densification, regularizer scheduling, outputs, checkpoints, WandB, GUI |
| [regularization-configs.md](regularization-configs.md) | Every YAML config under `gaussian_wrapping/configs/`, key by key, traced to the code that consumes it |
| [mesh-extraction.md](mesh-extraction.md) | Pivot-based extraction and Primal Adaptive Meshing pipelines, bounding volumes, Blender add-on, post-processing |
| [evaluation.md](evaluation.md) | Benchmarks (TNT, DTU, MipNeRF360, NeRF Synthetic), rendering and geometric metrics, per-script I/O |
| [architecture.md](architecture.md) | Module-by-module internals: scene, renderers, regularizers, submodules, data flow diagram |

## Repository layout

```
GaussianWrapping/
├── install.py                     # Conda/CUDA environment bootstrap (see installation.md)
├── pyproject.toml                 # Package metadata; console scripts gw-train / gw-render
├── requirements.txt               # Pinned Python dependencies
├── .gitmodules                    # Depth-Anything-V2, nvdiffrast, TNTUniScanEvals, GGGS
├── submodules/                    # CUDA rasterizers + native extensions (populated by git clone --recurse-submodules)
└── gaussian_wrapping/             # The Python package
    ├── train.py                   # Training loop + CLI (gw-train)
    ├── api.py                     # Public Python API (TrainingConfig / run_training)
    ├── render.py                  # Novel-view rendering of a trained model
    ├── convert.py                 # COLMAP RAW -> COLMAP bin conversion (ROOT_CAM etc.)
    ├── metrics.py                 # PSNR / SSIM / LPIPS over rendered vs. GT images
    ├── pivot_based_mesh_extraction.py
    ├── primal_adaptive_meshing_extraction.py
    ├── texture_mesh.py / mesh_decimate.py / evaluate_dtu_mesh.py
    ├── arguments/__init__.py      # ModelParams / OptimizationParams / PipelineParams
    ├── configs/                   # YAML regularization configs (see regularization-configs.md)
    ├── scene/                     # Scene, GaussianModel, cameras, COLMAP loaders
    ├── gaussian_renderer/         # Rasterizer wrappers + network GUI
    ├── regularization/            # Normal field, multiview, MILo, SDF, snapping
    ├── densification/ extraction/ functional/ utils/ blender/ eval/
    └── scripts/                   # End-to-end benchmark & train+extract wrappers
```

## Quick starts

### Train + extract a mesh (end-to-end wrapper)

```bash
# RaDe-GS rasterizer (best-looking meshes, a bit slower)
python gaussian_wrapping/scripts/train_and_extract_gw_radegs.py \
    -s <PATH_TO_COLMAP_DATASET> \
    -m <OUTPUT_DIR>

# "Ours" median-depth rasterizer (faster, better metrics)
python gaussian_wrapping/scripts/train_and_extract_gw_ours.py \
    -s <PATH_TO_COLMAP_DATASET> \
    -m <OUTPUT_DIR> \
    -r 2
```

### Train only, via the CLI

```bash
gw-train -s <DATASET> -m <OUTPUT> --rasterizer radegs
# equivalent: python -m gaussian_wrapping.train -s <DATASET> -m <OUTPUT>
```

All flags: [cli-reference.md](cli-reference.md). What happens during the run:
[training-guide.md](training-guide.md).

### Train only, via the Python API

```python
from gaussian_wrapping.api import TrainingConfig, run_training

config = TrainingConfig(
    source_path="/data/my_scene",
    model_path="/output/my_scene",
    rasterizer="radegs",
)
result = run_training(config)   # -> TrainingResult(scene, gaussians, ...)
```

All 78 `TrainingConfig` fields: [api-reference.md](api-reference.md).

## Downstream consumer

The companion repository `GW_experiments` (TransLab benchmark orchestration)
calls this package exclusively through `gaussian_wrapping.api` — see its
`docs/configuration-reference.md` for the YAML schema that maps 1:1 onto
`TrainingConfig`.
