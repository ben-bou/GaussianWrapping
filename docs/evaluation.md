# Evaluation

This guide covers benchmarking Gaussian Wrapping: geometric evaluation of extracted meshes (Tanks and Temples, DTU) and rendering-based novel view synthesis metrics (MipNeRF 360, and mesh-based NVS). It lists the prerequisites, the exact commands, what each script does, and the files it produces.

> Source of truth: `README.md` (Benchmarks), `gaussian_wrapping/eval/tnt/run.py`, `gaussian_wrapping/eval/tnt/cull_mesh.py`, `gaussian_wrapping/eval/tnt/evaluation.py`, `gaussian_wrapping/eval/tnt/registration.py`, `gaussian_wrapping/eval/tnt/compute_bbox_for_mesh.py`, `gaussian_wrapping/eval/dtu/eval.py`, `gaussian_wrapping/evaluate_dtu_mesh.py`, `gaussian_wrapping/metrics.py`, `gaussian_wrapping/render.py`, `gaussian_wrapping/eval/TNTUniScanEvals/` (git submodule), `.gitmodules`, `gaussian_wrapping/scripts/benchmark_tnt_gw_ours.py`, `gaussian_wrapping/scripts/benchmark_mip360_gw_ours.py`.

## Contents

1. [Overview: metric families](#1-overview-metric-families)
2. [Prerequisites](#2-prerequisites)
3. [Tanks and Temples](#3-tanks-and-temples)
4. [DTU](#4-dtu)
5. [MipNeRF 360](#5-mipnerf-360)
6. [Mesh-based novel view synthesis](#6-mesh-based-novel-view-synthesis)
7. [NeRF Synthetic](#7-nerf-synthetic)
8. [Rendering-based metrics](#8-rendering-based-metrics)
9. [Inputs and outputs per script](#9-inputs-and-outputs-per-script)
10. [Related documentation](#10-related-documentation)

## 1. Overview: metric families

| Family | Metrics | Scripts | Requires |
|---|---|---|---|
| Rendering-based (novel view synthesis) | PSNR, SSIM, LPIPS | `render.py` + `metrics.py` (LPIPS via `gaussian_wrapping/lpipsPyTorch`, VGG backbone) | Trained model, test split |
| Geometric (TNT) | Precision, recall, F-score | `eval/TNTUniScanEvals/uniform_sampling_eval.py`, `eval/TNTUniScanEvals/virtual_scan_sampling_eval.py`, `eval/tnt/run.py` | TNT ground truth (GT ply, crop JSON, alignment, COLMAP SfM log) |
| Geometric (DTU) | Mean d2s / s2d distances (accuracy / completeness) | `evaluate_dtu_mesh.py` + `eval/dtu/eval.py` | DTU evaluation data (ObsMask, STL points, Plane) |

## 2. Prerequisites

1. **Datasets.** Tanks and Temples scenes in COLMAP format (one subdirectory per scene, e.g. `Barn/`, `Caterpillar/`), MipNeRF 360 scenes (`bicycle/`, `bonsai/`, `counter/`, `garden/`, `kitchen/`, `room/`, `stump/`), DTU scans, NeRF Synthetic scenes.
2. **TNT ground truth.** For each scene, the ground-truth directory must contain `<Scene>.ply`, `<Scene>.json`, `<Scene>_trans.txt`, and `<Scene>_COLMAP_SfM.log`.
3. **TNTUniScanEvals submodule.** The uniform-sampling and virtual-scan scripts live in the git submodule `gaussian_wrapping/eval/TNTUniScanEvals` (see `.gitmodules`, URL `https://github.com/diego1401/TNTUniScanEvals.git`). Clone with `--recurse-submodules` or run `git submodule update --init` and follow that directory's `README.md` for dataset setup and installation.
4. **Trained models and meshes.** Produced by the end-to-end scripts (see [Mesh extraction](mesh-extraction.md) and [Training guide](training-guide.md)).
5. **DTU evaluation data.** `evaluate_dtu_mesh.py` expects `--DTU` to point at a directory containing `Calibration/cal18/pos_*.txt`, and `eval/dtu/eval.py` expects `<dataset_dir>/ObsMask/ObsMask<scan>_10.mat`, `<dataset_dir>/ObsMask/Plane<scan>.mat`, and `<dataset_dir>/Points/stl/stl<scan:03>_total.ply`.

## 3. Tanks and Temples

> **Note on `_post_` meshes:** The meshes that are evaluated with the following metrics are passed through a post-processing procedure similar to [GGGS](https://github.com/HKUST-SAIL/Geometry-Grounded-Gaussian-Splatting) and [PGSR](https://zju3dv.github.io/PGSR/). This procedure yields in practice better metrics. However, we find that it can sometimes remove objects on the scene (e.g chandelier in the Meeting Room scene). Thus, for the textured meshes we recommend using non post-processed meshes.

To reproduce the full Tanks and Temples results, run the end-to-end benchmark scripts from the project root. Each script trains all 6 scenes (`Barn`, `Caterpillar`, `Courthouse`, `Ignatius`, `Meetingroom`, `Truck`), extracts meshes, and runs all three evaluations (uniform sampling, virtual scan sampling, and legacy TNT):

```bash
# Ours rasterizer
python gaussian_wrapping/scripts/benchmark_tnt_gw_ours.py \
    --data_dir <PATH_TO_TNT_SCENES> \
    --gt_dir <PATH_TO_TNT_GT> \
    --output_dir <OUTPUT_DIR>

# RaDeGS rasterizer
python gaussian_wrapping/scripts/benchmark_tnt_gw_radegs.py \
    --data_dir <PATH_TO_TNT_SCENES> \
    --gt_dir <PATH_TO_TNT_GT> \
    --output_dir <OUTPUT_DIR>
```

`<PATH_TO_TNT_SCENES>` should contain one subdirectory per scene (`Barn/`, `Caterpillar/`, etc.) in COLMAP format. `<PATH_TO_TNT_GT>` should contain the ground truth files for each scene (`<Scene>.ply`, `<Scene>.json`, `<Scene>_trans.txt`, `<Scene>_COLMAP_SfM.log`).

Per-scene results are written to `<OUTPUT_DIR>/<Scene>/`, with evaluation outputs in `eval_uniform/`, `eval_virtual_scan/`, and `eval_legacy/` subdirectories.

The benchmark scripts run, per scene: `train_and_extract_gw_ours.py` (or the RaDe-GS variant) with `-r 2`, then evaluate `mesh_ours_2pivots_post.ply` (or `mesh_exact_computation_2pivots_searched_post.ply`) against `<gt_dir>/<Scene>/` with the three commands below.

### 3.1 Uniform Sampling Evaluation

Evaluates a mesh or point cloud against the ground truth using surface-based sampling and a 3-step ICP alignment pipeline.

```bash
python gaussian_wrapping/eval/TNTUniScanEvals/uniform_sampling_eval.py \
    --dataset-dir <path/to/scene_dir> \
    --traj-path <path/to/scene_dir>/<Scene>_COLMAP_SfM.log \
    --ply-path <path/to/mesh.ply> \
    --out-dir <output_dir>
```

### 3.2 Virtual Scan Sampling Evaluation

Evaluates a mesh by rendering depth from each training camera and projecting to world-space point clouds. Requires a COLMAP dataset and a CUDA-enabled GPU.

```bash
python gaussian_wrapping/eval/TNTUniScanEvals/virtual_scan_sampling_eval.py \
    -s <path/to/colmap_dataset> \
    -r 2 \
    --dataset-dir <path/to/scene_dir> \
    --traj-path <path/to/scene_dir>/<Scene>_COLMAP_SfM.log \
    --ply-path <path/to/mesh.ply> \
    --out-dir <output_dir>
```

Both scripts output **precision**, **recall**, and **F-score** to the console, along with precision/recall curve plots. Pass `--save_point_clouds` to also write color-coded error clouds. Full documentation and the standalone toolbox are available at [diego1401/TNTUniScanEvals](https://github.com/diego1401/TNTUniScanEvals.git).

### 3.3 Legacy TNT evaluation (`eval/tnt/run.py`)

The original TanksAndTemples toolbox evaluation, used by the benchmark scripts as the third evaluation:

```bash
python gaussian_wrapping/eval/tnt/run.py \
    --dataset-dir <path/to/scene_dir> \
    --traj-path <path/to/scene_dir>/<Scene>_COLMAP_SfM.log \
    --ply-path <path/to/mesh.ply> \
    --out-dir <output_dir>
```

What `run.py` does:

1. Looks up the per-scene distance threshold `dTau` in `eval/tnt/config.py` (`scenes_tau_dict`).
2. Loads the reconstruction PLY, adds face-center points to the vertex cloud, and loads the GT point cloud.
3. Aligns the estimated trajectory to the reference COLMAP SfM log (`registration.py`: `trajectory_alignment`, `registration_vol_ds`, `registration_unif` — three ICP refinement iterations).
4. Crops both clouds with the selection polygon volume from `<Scene>.json`.
5. Computes precision/recall/F-score at `dTau / 2` (`evaluation.py`: `EvaluateHisto`).

Outputs in `--out-dir`: `<Scene>.precision.ply`, `<Scene>.recall.ply`, precision/recall curve plots (`PR_<Scene>_@d_th_*.png/pdf` via `plot.py`), and the console report of `distance tau`, `precision`, `recall`, `f-score`.

### 3.4 Supporting TNT scripts

| Script | Role |
|---|---|
| `eval/tnt/cull_mesh.py` | Library module (no CLI): `Mesher` renders depth maps of a mesh with `pyrender` and culls unseen geometry; `get_traj` reads `.npy`/`.json`/`.log` trajectories. The legacy TNT `run.sh` workflow culls the mesh before evaluation. |
| `eval/tnt/registration.py` | ICP-based trajectory alignment and registration used by `run.py`. |
| `eval/tnt/evaluation.py` | `EvaluateHisto`: precision/recall histograms, F-score, and color-coded error clouds. |
| `eval/tnt/compute_bbox_for_mesh.py` | Utility that prints the axis-aligned bounding box of the GT point cloud after trajectory alignment (for building crop volumes). |
| `eval/tnt/config.py` | `scenes_tau_dict`: per-scene `dTau` thresholds (Barn 0.01, Caterpillar 0.005, Church 0.025, Courthouse 0.025, Ignatius 0.003, Meetingroom 0.01, Truck 0.005). |

## 4. DTU

The README lists the DTU benchmark as "Coming soon", but the evaluation code is present in the repository:

1. `evaluate_dtu_mesh.py` — loads the trained model, culls and aligns the mesh, then calls `eval/dtu/eval.py`:

```bash
python gaussian_wrapping/evaluate_dtu_mesh.py \
    -s <PATH_TO_DTU_SCAN> \
    -m <OUTPUT_DIR> \
    --DTU <PATH_TO_DTU_EVAL_DATA>
```

What it does:

1. Loads DTU cameras from `<DTU>/Calibration/cal18/pos_{i:03d}.txt` (64 projection matrices).
2. Loads the mesh from `<model_path>/recon_post.ply` (hard-coded name).
3. Culls the mesh by projecting vertices into each camera frustum (dilated `gt_mask`).
4. Fits the scale to the DTU camera positions and exports `<model_path>/recon_aligned.ply`.
5. Runs `python eval/dtu/eval.py --data <aligned_mesh> --scan <scan> --mode mesh --dataset_dir <DTU> --vis_out_dir <model_path>/vis`, where `<scan>` is parsed from the last component of `-s` (e.g. `scan24` → `24`).

2. `eval/dtu/eval.py` — DTUeval-python style evaluation:

```bash
python gaussian_wrapping/eval/dtu/eval.py \
    --data <PATH_TO_ALIGNED_MESH.ply> \
    --scan <SCAN_ID> \
    --mode mesh \
    --dataset_dir <PATH_TO_DTU_EVAL_DATA> \
    --vis_out_dir <OUTPUT_DIR>
```

What it does: samples points from the mesh (`--downsample_density`, default `0.2`), masks them with `ObsMask<scan>_10.mat`, computes mean distance from data to STL (`mean_d2s`, accuracy) and from STL to data (`mean_s2d`, completeness) with `--max_dist` (default `20`), and prints the overall score (their mean).

Outputs in `--vis_out_dir`: `vis_<scan:03>_d2s.ply`, `vis_<scan:03>_s2d.ply` (color-coded error clouds), and `results.json` with `mean_d2s`, `mean_s2d`, `overall`.

## 5. MipNeRF 360

To reproduce the full MipNeRF 360 results, run the end-to-end benchmark scripts from the project root. Each script trains the scenes, extracts meshes, and evaluates novel view synthesis quality:

```bash
# Ours rasterizer
python gaussian_wrapping/scripts/benchmark_mip360_gw_ours.py \
    --data_dir <PATH_TO_MIP360_SCENES> \
    --output_dir <OUTPUT_DIR>

# RaDeGS rasterizer
python gaussian_wrapping/scripts/benchmark_mip360_gw_radegs.py \
    --data_dir <PATH_TO_MIP360_SCENES> \
    --output_dir <OUTPUT_DIR>
```

`<PATH_TO_MIP360_SCENES>` should contain one subdirectory per scene (`bicycle/`, `bonsai/`, `counter/`, `garden/`, `kitchen/`, `room/`, `stump/`) in COLMAP format.

Per-scene results are written to `<OUTPUT_DIR>/<scene>/`. Rendered images and metrics (PSNR / SSIM / LPIPS) are saved in the standard Gaussian Splatting output structure. Each scene runs three steps: `train_and_extract_gw_ours.py` (or the RaDe-GS variant), then `render.py`, then `metrics.py -m <OUTPUT_DIR>/<scene>`.

## 6. Mesh-based novel view synthesis

We evaluate mesh-based novel view synthesis on the MipNeRF360 and Tanks and Temples datasets. Use the training and mesh extraction scripts described in [Mesh extraction](mesh-extraction.md) to produce meshes for these datasets, then refer to [MILo](https://github.com/Anttwo/MILo.git) for the evaluation pipeline.

The repository also ships `gaussian_wrapping/blender/milo_addon.py` ("Mesh-based Editing Tool for MILo"), a Blender add-on that can import MILo meshes, edit them, and export a render package (`<milo_dir>/blender/packages/<scene>.json`) containing camera, mesh, and armature data for rendering edited scenes with Gaussians.

## 7. NeRF Synthetic

> **Note:** We only launch the training and extraction with the RaDe-GS rasterizer since it leads to smoother meshes. There are no metrics in this script. We provide it just as an example of our method being applied to synthetic scenes.

```bash
python gaussian_wrapping/scripts/launch_nerf_synthetic.py \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --gpu_device "${GPU_DEVICE}"
```

The script trains and extracts a mesh per scene with `train_and_extract_gw_radegs.py --isosurface_value 0.2 --no_postprocess`, then runs PAM (`primal_adaptive_meshing_extraction.py`) on `mesh_exact_computation_2pivots_transmittance_threshold_0.7_searched.ply` to produce `<scene>_pam.ply`, and finally runs texture refinement on the PAM mesh.

## 8. Rendering-based metrics

### 8.1 Rendering a trained model (`render.py`)

`gaussian_wrapping/render.py` renders the trained Gaussians for the train and test splits:

```bash
python gaussian_wrapping/render.py \
    -s <PATH_TO_COLMAP_DATASET> \
    -m <OUTPUT_DIR> \
    --rasterizer ours
```

Key flags: `--iteration` (default `-1`, i.e. the latest checkpoint), `--skip_train`, `--skip_test`, `--rasterizer` (`radegs` by default, `ours` recommended for the benchmark scripts).

`render.py` uses `get_combined_args` (from `gaussian_wrapping/arguments/__init__.py`), which merges the CLI with the `cfg_args` file stored in the model directory at training time: any argument not given on the command line (e.g. `-s`, `--sh_degree`, `--white_background`) is taken from `<model_dir>/cfg_args`, so the model is reconstructed exactly as it was trained.

Outputs, per split (`train`/`test`), under `<model_dir>/<split>/ours_<iteration>/`:

- `renders/` — `00000.png`, `00001.png`, ... rendered images.
- `gt/` — the corresponding ground-truth images.

### 8.2 Computing metrics (`metrics.py`)

```bash
python gaussian_wrapping/metrics.py -m <OUTPUT_DIR>
```

`metrics.py` accepts multiple model paths (`-m` is `nargs="+"`), reads `<model_dir>/test/<method>/renders` and `<model_dir>/test/<method>/gt`, and computes per-image SSIM (`utils/loss_utils.py`), PSNR (`utils/image_utils.py`), and LPIPS (`gaussian_wrapping/lpipsPyTorch`, VGG backbone: `lpips(img1, img2, net_type='vgg')`). It prints the per-scene means and writes:

- `<model_dir>/results.json` — `{"SSIM": ..., "PSNR": ..., "LPIPS": ...}` per method.
- `<model_dir>/per_view.json` — per-image metric values.

## 9. Inputs and outputs per script

| Script | Inputs | Outputs |
|---|---|---|
| `scripts/benchmark_tnt_gw_ours.py` / `benchmark_tnt_gw_radegs.py` | `--data_dir` (COLMAP TNT scenes), `--gt_dir` (TNT GT), `--output_dir` | `<output_dir>/<Scene>/` trained model + `mesh_ours_2pivots_post.ply` (ours) or `mesh_exact_computation_2pivots_searched_post.ply` (RaDe-GS); `eval_uniform/`, `eval_virtual_scan/`, `eval_legacy/` |
| `eval/TNTUniScanEvals/uniform_sampling_eval.py` | `--dataset-dir`, `--traj-path`, `--ply-path`, `--out-dir` | Console precision/recall/F-score, P/R curve plots, optional `--save_point_clouds` error clouds |
| `eval/TNTUniScanEvals/virtual_scan_sampling_eval.py` | Same as above plus `-s` (COLMAP dataset), `-r` | Same as above |
| `eval/tnt/run.py` | `--dataset-dir`, `--traj-path`, `--ply-path`, `--out-dir` | `<Scene>.precision.ply`, `<Scene>.recall.ply`, P/R curve plots, console F-score |
| `evaluate_dtu_mesh.py` | `-s`, `-m`, `--DTU` | `<model_path>/recon_aligned.ply`, `<model_path>/vis/` results |
| `eval/dtu/eval.py` | `--data`, `--scan`, `--mode`, `--dataset_dir`, `--vis_out_dir` | `vis_<scan:03>_d2s.ply`, `vis_<scan:03>_s2d.ply`, `results.json` (`mean_d2s`, `mean_s2d`, `overall`) |
| `scripts/benchmark_mip360_gw_ours.py` / `benchmark_mip360_gw_radegs.py` | `--data_dir`, `--output_dir` | `<output_dir>/<scene>/` trained model, mesh, `train/ours_<iter>/` and `test/ours_<iter>/` renders, `results.json`, `per_view.json` |
| `render.py` | `-s`, `-m`, `--rasterizer`, `--iteration`, `--skip_train`, `--skip_test` | `<model_dir>/<split>/ours_<iteration>/renders/*.png` and `gt/*.png` |
| `metrics.py` | `-m <model_dir> [<model_dir> ...]` | `<model_dir>/results.json`, `<model_dir>/per_view.json` |
| `scripts/launch_nerf_synthetic.py` | `--data_dir`, `--output_dir`, `--gpu_device` | Per-scene trained model, `mesh_exact_computation_2pivots_transmittance_threshold_0.7_searched.ply`, `<scene>_pam.ply`, texture-refined mesh |

## 10. Related documentation

- [Mesh extraction](mesh-extraction.md) — how the evaluated meshes are produced.
- [Training guide](training-guide.md) — training runs and the model directory layout.
- [CLI reference](cli-reference.md) — full flag tables for every script.
- [Regularization configs](regularization-configs.md) — depth-order and other regularization configs.
- [API reference](api-reference.md) — module-level documentation.
- [Architecture](architecture.md) — how the components fit together.
