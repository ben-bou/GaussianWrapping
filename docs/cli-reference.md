# GaussianWrapping CLI Reference

This document is the complete command-line reference for every entry point of the
`gaussian_wrapping` package: the training and rendering CLIs, the COLMAP converter,
the mesh extraction and texturing tools, the metric scripts, and the Tanks and
Temples / DTU evaluation toolbox. Every flag is listed with its exact type, default
value, shorthand (if any), and behavior as implemented in the code. The shared
`ParamGroup` mechanism and the `get_combined_args` merge helper are documented in
their own sections.

> Source of truth: `gaussian_wrapping/arguments/__init__.py`,
> `gaussian_wrapping/train.py`, `gaussian_wrapping/render.py`,
> `gaussian_wrapping/convert.py`,
> `gaussian_wrapping/pivot_based_mesh_extraction.py`,
> `gaussian_wrapping/primal_adaptive_meshing_extraction.py`,
> `gaussian_wrapping/evaluate_dtu_mesh.py`, `gaussian_wrapping/mesh_decimate.py`,
> `gaussian_wrapping/texture_mesh.py`, `gaussian_wrapping/metrics.py`,
> `gaussian_wrapping/eval/tnt/run.py`, `gaussian_wrapping/eval/tnt/cull_mesh.py`,
> `gaussian_wrapping/eval/tnt/compute_bbox_for_mesh.py`,
> `gaussian_wrapping/eval/tnt/evaluation.py`,
> `gaussian_wrapping/eval/tnt/registration.py`,
> `gaussian_wrapping/eval/dtu/eval.py`, and `pyproject.toml`.

## Contents

1. [gw-train](#1-gw-train)
2. [gw-render](#2-gw-render)
3. [convert](#3-convert)
4. [pivot_based_mesh_extraction](#4-pivot_based_mesh_extraction)
5. [primal_adaptive_meshing_extraction](#5-primal_adaptive_meshing_extraction)
6. [evaluate_dtu_mesh](#6-evaluate_dtu_mesh)
7. [mesh_decimate](#7-mesh_decimate)
8. [texture_mesh](#8-texture_mesh)
9. [metrics](#9-metrics)
10. [Tanks and Temples toolbox (eval/tnt)](#10-tanks-and-temples-toolbox-evaltnt)
11. [get_combined_args and cfg_args](#11-get_combined_args-and-cfg_args)
12. [See also](#12-see-also)

---

## 1. gw-train

```bash
python -m gaussian_wrapping.train -s <scene_path> -m <output_path> [flags]
# equivalent console script (see pyproject.toml [project.scripts]):
gw-train -s <scene_path> -m <output_path> [flags]
```

Trains a GaussianWrapping scene: joint optimization of 3D Gaussians, a learnable
per-Gaussian normal field, and (optionally) an in-the-loop mesh, with multiview and
depth-normal regularization. `gw-train` maps to `gaussian_wrapping.train:main`;
`main(argv=None)` parses `sys.argv[1:]` unless an argv list is passed programmatically
(the library API in [api-reference.md](api-reference.md) reuses the same `training()`
function directly).

Flags come from four places: the three shared `ParamGroup`s registered in
`arguments/__init__.py` (`ModelParams`, `OptimizationParams`, `PipelineParams`) and
roughly forty flags declared directly in `train.py:main()`. Every attribute of a
`ParamGroup` becomes a `--flag`; attributes whose name starts with `_` additionally
get a shorthand `-<first letter>` (e.g. `_source_path` provides `--source_path` and
`-s`). Boolean attributes registered through `ParamGroup` are `store_true` flags.

### Behavior notes (verified in `train.py`)

- `args.iterations` is always appended to `save_iterations`, so the final model is
  always saved as `point_cloud/iteration_<iterations>/point_cloud.ply`.
- The final iteration is appended to `test_iterations` unless `-1` is present in the
  list; passing `--test_iterations -1` therefore disables final-iteration testing.
- `--port -1` (the default) picks a random port in `[5000, 9000)` via
  `np.random.randint(5000, 9000)`.
- The network GUI server is always started on the CLI path
  (`network_gui.init(args.ip, args.port)` and `enable_gui=True` are unconditional);
  there is no flag to disable it.
- `args.use_abs_grad_for_densification` is forced to `True` (RaDe-GS-style
  densification statistics always use absolute gradients).
- When `--rasterizer ours` is selected, `mask_depth_normal` is forced to `True`
  (printed as `[INFO] Using Ours rasterizer. Setting mask_depth_normal to True.`).
- Regularization YAML configs are loaded from
  `gaussian_wrapping/configs/<group>/<name>.yaml` based on the `--*_config` flags;
  see [Regularization configs](regularization-configs.md). Available names:
  `normal_field/{default,default_regular_densification}`,
  `multiview/{default_regular_densification,fast,fast_late}`,
  `mesh_in_the_loop/default_regular_densification`, `depth_order/default`.
- If `--model_path` is empty, the output directory defaults to
  `./output/<first 10 chars of a uuid4>` (or `$OAR_JOB_ID` when set), and a `cfg_args`
  file (a `repr` of the resolved `ModelParams` namespace) is written into it; see
  [Section 11](#11-get_combined_args-and-cfg_args).
- Weights & Biases logging is enabled only when `wandb` is importable **and** both
  `--wandb_project` and `--wandb_entity` are provided; otherwise logging is skipped.
- On completion, total wall-clock training time is written to `<model_path>/time.txt`.

### Model parameters (`ModelParams`)

Registered without sentinel, so the defaults below are the effective CLI defaults.

| Flag | Shorthand | Type | Default | Description |
|------|-----------|------|---------|-------------|
| `--sh_degree` | | int | `3` | Spherical harmonics degree of the Gaussian model. |
| `--source_path` | `-s` | str | `""` | Path to the COLMAP scene directory; converted to an absolute path by `ModelParams.extract`. |
| `--model_path` | `-m` | str | `""` | Output/model directory. Empty means `./output/<uuid>` (see behavior notes). |
| `--images` | `-i` | str | `"images"` | Subdirectory of `source_path` containing the input images. |
| `--resolution` | `-r` | int | `-1` | Image resolution handling: `1/2/4/8` divide width and height by that factor (times the resolution scale); `-1` downscales images wider than 1600 px to 1600 px width (with a one-time warning); any other value `N` divides the width by `N`. |
| `--white_background` | `-w` | bool | `False` | Use a white background instead of black (`store_true`). |
| `--data_device` | | str | `"cuda"` | Device on which per-camera GT images (and masks) are cached; invalid values fall back to `cuda`. |
| `--eval` | | bool | `False` | Split cameras into train/test instead of training on all images (`store_true`). |
| `--save_init_ply` | | bool | `False` | Save the initial point cloud (`store_true`). |
| `--llff` | | int | `8` | With `--eval`, every `llff`-th camera (`idx % llff == 0`) is held out as test (LLFF-style holdout). |
| `--kernel_size` | | float | `0.0` | 2D filter kernel size forwarded to the rasterizers (Mip-Splatting-style anti-aliasing). |
| `--use_unbounded_opacity` | | bool | `False` | Enable the unbounded opacity model (`store_true`). |

### Optimization parameters (`OptimizationParams`)

No shorthands (no attribute names start with `_`).

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--iterations` | int | `30000` | Total number of training iterations. |
| `--position_lr_init` | float | `0.00016` | Initial learning rate for Gaussian positions. |
| `--position_lr_final` | float | `0.0000016` | Final learning rate for Gaussian positions. |
| `--position_lr_delay_mult` | float | `0.01` | Delay multiplier of the position LR scheduler. |
| `--position_lr_max_steps` | int | `30000` | Step count over which the position LR is annealed. |
| `--feature_dc_lr` | float | `0.0025` | Learning rate for the DC spherical-harmonics component. |
| `--feature_rest_lr` | float | `0.000125` | Learning rate for the remaining SH components (`0.0025 / 20`). |
| `--opacity_lr` | float | `0.05` | Learning rate for opacities. |
| `--scaling_lr` | float | `0.005` | Learning rate for scales. |
| `--rotation_lr` | float | `0.001` | Learning rate for rotations. |
| `--percent_dense` | float | `0.01` | Extent threshold (fraction of `cameras_extent`) used to classify Gaussians as "dense" during densification. |
| `--lambda_dssim` | float | `0.2` | Weight of the D-SSIM term in the photometric loss (`(1 - lambda) * L1 + lambda * (1 - SSIM)`). |
| `--densification_interval` | int | `100` | Iterations between densify-and-prune steps (within `[densify_from_iter, densify_until_iter)`). |
| `--opacity_reset_interval` | int | `3000` | Iterations between opacity resets; also gates the `size_threshold = 20` behavior (see `--use_max_size_threshold`). |
| `--densify_from_iter` | int | `500` | First iteration at which densification may run. |
| `--densify_until_iter` | int | `15000` | Last iteration of the densification phase. |
| `--densify_grad_threshold` | float | `0.0002` | Positional-gradient threshold for densification. |
| `--random_background` | bool | `False` | Replace the background with random noise during training (`store_true`). |
| `--appearance_embeddings_lr` | float | `0.001` | Learning rate for per-image appearance embeddings (used with `--decoupled_appearance`). |
| `--appearance_network_lr` | float | `0.001` | Learning rate for the appearance network (used with `--decoupled_appearance`). |
| `--gaussian_features_lr` | float | `0.025` | Learning rate for the learnable Gaussian features (normal field / SDF features; `0.05 / 2`). |
| `--pgsr_appearance_lr` | float | `0.001` | Learning rate for PGSR-style exposure-compensation parameters (used with `--exposure_compensation`). |

### Pipeline parameters (`PipelineParams`)

No shorthands.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--convert_SHs_python` | bool | `False` | Compute SH in Python instead of the CUDA rasterizer (`store_true`). |
| `--compute_cov3D_python` | bool | `False` | Compute 3D covariance in Python instead of the CUDA rasterizer (`store_true`). |
| `--debug` | bool | `False` | Enable rasterizer debug mode (`store_true`); also toggled on from iteration `debug_from + 1` onward by `--debug_from`. |

### Training script flags (declared in `main()`)

#### General

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--ip` | str | `"127.0.0.1"` | IP address the network GUI server binds to. |
| `--port` | int | `-1` | Port of the network GUI server; `-1` selects a random port in `[5000, 9000)`. |
| `--debug_from` | int | `-1` | Iteration from which `pipe.debug` is enabled: at iteration `debug_from + 1` the flag is set and stays on for the rest of training. |
| `--detect_anomaly` | bool | `False` | Enable `torch.autograd.set_detect_anomaly` (`store_true`). |
| `--test_iterations` | int (nargs +) | `[]` | Iterations at which test-set metrics are rendered/logged. The final iteration is appended automatically unless `-1` is in the list. |
| `--save_iterations` | int (nargs +) | `[]` | Iterations at which the Gaussian point cloud is saved. `args.iterations` is always appended. |
| `--quiet` | bool | `False` | Suppress console output (`safe_state` silent mode) (`store_true`). |
| `--checkpoint_iterations` | int (nargs +) | `[]` | Iterations at which full checkpoints (`chkpnt<N>.pth`, Gaussian state + iteration) are saved to `model_path`. |
| `--start_checkpoint` | str | `None` | Path to a `chkpnt<N>.pth` to resume from. Restores Gaussian parameters and the iteration counter, and pushes the normal-field / MiLo / multiview start iterations past the resumed iteration. |

#### Rasterization

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--rasterizer` | str | `"radegs"` | Rasterizer backend; choices `radegs`, `ours`. Selects the `render`/`integrate` functions via `resolve_rasterizer`. `ours` additionally forces `mask_depth_normal = True`. |

#### Normal field

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--no_normal_field` | bool | `False` | Disable the learnable per-Gaussian normal field (`store_true`); `use_normal_field` is set to `not no_normal_field`. When enabled, the number of learnable Gaussian features is 1 (if the config uses the smallest axis) or 4; 0 when disabled. |
| `--normal_field_config` | str | `"default_regular_densification"` | Name of the YAML file loaded from `gaussian_wrapping/configs/normal_field/<name>.yaml`. |

#### Gaussian management

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--dense_gaussians` | bool | `False` | Use dense Gaussians (`store_true`; announced at startup and consumed downstream). |
| `--detach_gaussian_rendering` | bool | `False` | Detach the rendered RGB/depth/normal tensors before the normal-field regularization loss, so that loss does not back-propagate through the rasterization (`store_true`). |

#### Densification and simplification (Mini-Splatting2-inspired)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--N_max_gaussians` | int | `None` | Cap Gaussian count during Normal Field Densification. If the next densification would exceed this, only the highest-error Gaussians are added up to the cap. `None` = no limit. |
| `--warn_until_iter` | int | `3000` | After densification ends, the 3D mip filter is also refreshed at this iteration (in addition to the `--update_mip_filter_every` cadence). |
| `--non_maximal_pruning_iterations` | int (nargs +) | `[]` | Additional iterations at which non-maximal Gaussians are pruned (same routine the normal-field pruning schedule uses). |
| `--use_max_size_threshold` | bool (`--flag`/`--no-flag`) | `False` | When enabled, densification passes a max-size threshold of 20 (applied after `opacity_reset_interval` has elapsed) to `densify_and_prune_radegs`; otherwise no size threshold is used. |

#### Depth-normal consistency regularization (2DGS / GOF / RaDe-GS-inspired)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--regularization_from_iter` | int | `7000` | Iteration from which the depth-normal consistency loss is active. |
| `--lambda_depth_normal` | float | `0.05` | Weight of the depth-normal consistency loss. |
| `--mask_depth_normal` | bool | `False` | Use the validity-masked variant: blend expected and median depth (0.4/0.6) where the median depth is valid, derive normals with a validity mask, and ignore invalid pixels (`store_true`). Forced to `True` when `--rasterizer ours`. |

#### Multiview regularization

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--multiview` | bool (`--flag`/`--no-flag`) | `True` | Enable multiview regularization (enabled by default). |
| `--multiview_config` | str | `"fast"` | Name of the YAML file loaded from `gaussian_wrapping/configs/multiview/<name>.yaml`. |
| `--multiview_rasterizer` | str | `None` | Optional rasterizer override used to render the multiview loss. Only `None` (reuse the main render) or `"ours"` are accepted; anything else raises `ValueError`. |
| `--multiview_factor` | float | `1.0` | Scalar multiplier applied to the multiview loss. |

#### Mesh-in-the-loop (MiLo) regularization

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--milo` | bool | `False` | Enable mesh-in-the-loop regularization: makes occupancy learnable and adds `n_pivots_per_gaussian` pivots per Gaussian (2 when disabled) (`store_true`). |
| `--milo_config` | str | `"default_regular_densification"` | Name of the YAML file loaded from `gaussian_wrapping/configs/mesh_in_the_loop/<name>.yaml`. |

#### Depth-order regularization (learned prior)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--depth_order` | bool | `False` | Enable monocular depth-order supervision. Relies on Depth-Anything-V2 and is **not** used in the MILo paper (`store_true`). |
| `--depth_order_config` | str | `"default"` | Name of the YAML file loaded from `gaussian_wrapping/configs/depth_order/<name>.yaml`. |

#### 3D mip filter (Mip-Splatting-inspired)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--disable_mip_filter` | bool | `False` | Disable the 3D mip filter entirely (`store_true`). |
| `--update_mip_filter_every` | int | `100` | Refresh cadence of the 3D mip filter once densification has ended; updates are skipped within the last `update_mip_filter_every` iterations of training. |

#### Appearance / exposure

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--decoupled_appearance` | bool | `False` | Use a per-image appearance network in the photometric loss (GOF-inspired) (`store_true`). |
| `--exposure_compensation` | bool (`--flag`/`--no-flag`) | `True` | Per-camera exposure compensation (PGSR-inspired); enabled by default. Initializes one exposure per training camera. |

#### PGSR losses

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--use_scale_loss` | bool | `False` | Add the PGSR minimum-scale regularization (mean of the smallest scale axis of visible Gaussians) (`store_true`). |
| `--scale_loss_weight` | float | `100.0` | Weight of the minimum-scale loss. |
| `--weight_by_img_grad` | bool | `False` | Weight the depth-normal loss per pixel by image-gradient magnitude, as in PGSR (`store_true`). |

#### Logging

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--log_interval` | int | `None` | If set, emit the detailed image-rich log (console + WandB) every N iterations, and additionally at the start iteration of the normal field and MiLo regularizers. `None` keeps only the tqdm progress bar (updated every 10 iterations). |
| `--wandb_project` | str | `None` | Weights & Biases project name. WandB is initialized only if this and `--wandb_entity` are both set. |
| `--wandb_entity` | str | `None` | Weights & Biases entity name. |
| `--wandb_name` | str | `None` | Weights & Biases run name. |

---

## 2. gw-render

```bash
python -m gaussian_wrapping.render -m <model_path> [flags]
```

Renders the trained model's train and/or test cameras to PNG images and writes them
under `<model_path>/{train,test}/ours_<iteration>/{renders,gt}/`, alongside the
corresponding ground-truth crops. The rasterizer is selected by `--rasterizer`
(`render_radegs` or `render_ours`); `--iteration -1` loads the latest iteration found
under `point_cloud/` (`searchForMaxIteration`).

> **Known issue:** the `gw-render` console script declared in `pyproject.toml`
> (`gw-render = "gaussian_wrapping.render:main"`) is currently broken:
> `gaussian_wrapping/render.py` defines **no `main()` function** — its argparse
> setup lives inside the `if __name__ == "__main__":` block, so importing
> `gaussian_wrapping.render:main` fails. Use `python -m gaussian_wrapping.render`
> instead, which executes the module as `__main__` and works as documented. This
> matches the note in the downstream experiments README (`GW_experiments/README.md`).

Flags: `ModelParams` is registered with `sentinel=True`, so every `ModelParams` flag
defaults to `None` and its value comes from `<model_path>/cfg_args` unless explicitly
passed on the CLI (see [Section 11](#11-get_combined_args-and-cfg_args)); the flag
names, types and shorthands are identical to the table in [Section 1](#1-gw-train).
`PipelineParams` is registered normally (`False` defaults).

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--iteration` | int | `-1` | Checkpoint iteration to load; `-1` loads the latest iteration under `point_cloud/`. |
| `--skip_train` | bool | `False` | Do not render the training cameras (`store_true`). |
| `--skip_test` | bool | `False` | Do not render the test cameras (`store_true`). |
| `--quiet` | bool | `False` | Suppress console output (`safe_state` silent mode) (`store_true`). |
| `--rasterizer` | str | `"radegs"` | Rasterizer used for rendering; choices `radegs`, `ours`. |

---

## 3. convert

```bash
python -m gaussian_wrapping.convert -s <scene_path> [flags]
```

COLMAP conversion script (based on the MipNerf 360 shell converter). It runs COLMAP
feature extraction, exhaustive matching, and mapping on `<source_path>/input`,
undistorts the images into pinhole intrinsics, and normalizes the sparse model into
`<source_path>/sparse/0`. With `--resize` it additionally produces `images_2`,
`images_4`, and `images_8` downscaled copies using ImageMagick's `mogrify`.

Implementation notes:

- The parser is created and `parse_args()` is called **at module import time** (there
  is no `main()` and no `if __name__ == "__main__"` guard), so the flags below are
  the module's complete interface and importing the module as a library would also
  trigger argument parsing.
- There are no `--chunk` or `--mask` flags in this converter.
- Expects images in `<source_path>/input/`; writes `distorted/`, `sparse/0/`, and
  (with `--resize`) `images_2/4/8` inside `source_path`.

| Flag | Shorthand | Type | Default | Description |
|------|-----------|------|---------|-------------|
| `--no_gpu` | | bool | `False` | Disable GPU usage for COLMAP SIFT extraction/matching (`SiftExtraction.use_gpu` / `SiftMatching.use_gpu` = 0) (`store_true`). |
| `--skip_matching` | | bool | `False` | Skip feature extraction/matching/mapping and only run image undistortion on an existing `distorted/sparse/0` (`store_true`). |
| `--source_path` | `-s` | str | *(required)* | Scene directory containing `input/`; all outputs are written inside it. |
| `--camera` | | str | `"OPENCV"` | COLMAP camera model passed as `--ImageReader.camera_model`. |
| `--colmap_executable` | | str | `""` | Path to the COLMAP binary; empty means `colmap` from `PATH`. |
| `--resize` | | bool | `False` | Additionally create 50% / 25% / 12.5% resized copies (`images_2`, `images_4`, `images_8`) using ImageMagick (`store_true`). |
| `--magick_executable` | | str | `""` | Path to the ImageMagick `magick` binary; empty means `magick` from `PATH`. |

---

## 4. pivot_based_mesh_extraction

```bash
python -m gaussian_wrapping.pivot_based_mesh_extraction \
    -m <model_path> --iteration 30000 [pivot flags]
```

Extracts a mesh from a trained Gaussian model with marching tetrahedra over a
Delaunay triangulation of "pivot" points (adopted from gaussian-opacity-fields'
`extract_mesh.py`). Pivots are placed around Gaussians, an SDF is evaluated at the
pivots (via transmittance/occupancy integration), a Delaunay tetrahedralization is
built (`return_delaunay_tets`, method `"tetranerf"`), and the zero level set is
extracted, optionally refined by binary search, vertex-colored by Gaussian
integration, and post-processed to remove floaters.

Output file: `<model_path>/mesh_<sdf_mode>_<n_pivots>pivots[_iso_<v>|_transmittance_threshold_<t>][_scores][_searched].ply`,
plus a `_post.ply` variant when `--postprocess` is given.

Behavior notes (verified in code):

- `--isosurface_value <= -9999` (the default `-9999.0`) triggers automatic isosurface
  estimation: depth maps are rendered for all training cameras and the isosurface is
  computed from 1,000,000 depth samples with a median reduction
  (`compute_isosurface_value_from_depth`). Any other value is used as-is.
- With `--sdf_mode exact_computation`, the SDF is evaluated exactly from vacancy
  transmittance; the effective transmittance threshold becomes `0.5 + isosurface_value`
  and `isosurface_value` is reset to `0.0`.
- Pivot selection is prioritized as: `--use_tetra_points` (forces `n_pivots = 9` and
  uses tetrahedral corner points of the Gaussians) > `--use_scores` >
  `--random_pivots` > `--use_searched_pivots` > default (intersecting pivots from
  Gaussian normals). When `--use_scores`, `--random_pivots`, or `--use_searched_pivots`
  is set, `use_tetra_points`, `use_intersecting_pivots`, `filter_large_edges`, and
  `collapse_large_edges` are all forced to `False`.
- `--use_valid_mask` marks pivots outside all camera frusta (and outside GT masks, if
  present) as invalid and sets their SDF to `0.5` before tetrahedralization.
- Seeds are fixed (`random`, `numpy`, `torch` = 0) and `cuda:0` is selected.
- `ModelParams` is registered with `sentinel=True` (all `ModelParams` flags default to
  `None`; values come from `cfg_args` unless passed explicitly — see
  [Section 11](#11-get_combined_args-and-cfg_args)); `PipelineParams` is registered
  normally. Flag names/types/shorthands are as in [Section 1](#1-gw-train).

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--iteration` | int | `30000` | Checkpoint iteration to load (`point_cloud/iteration_<N>/point_cloud.ply`). |
| `--quiet` | bool | `False` | Suppress console output (`store_true`). |
| `--rasterizer` | str | `"ours"` | Rasterizer providing the render function used for automatic isosurface estimation; choices `radegs`, `ours`. |
| `--sdf_mode` | str | `"exact_computation"` | SDF evaluation mode; choices `exact_computation` (slower but exact vacancy transmittance) or `ours` (per-view alpha integration). |
| `--n_points_per_sdf_evaluation` | int | `999999999` | Maximum number of points per SDF evaluation batch; larger query sets are split into batches of this size. |
| `--dtype` | str | `"int32"` | Index dtype for marching tetrahedra; choices `int32`, `int64`. |
| `--n_pivots` | int | `2` | Number of pivots sampled per Gaussian (forced to 9 by `--use_tetra_points`). |
| `--use_smallest_axis_as_normal` | bool | `False` | Use the Gaussian's smallest scale axis as its normal when computing intersecting pivots (`store_true`). |
| `--std_factor` | float | `3.0` | Standard-deviation factor controlling how far pivots are placed along the normal axis. |
| `--use_intersecting_pivots` | bool | `False` | Use pivots obtained by intersecting the Gaussian normal axes with the SDF zero crossing (SDF evaluated during pivot search) (`store_true`). |
| `--use_tetra_points` | bool | `False` | Use the Gaussians' tetrahedral corner points as pivots (implies 9 pivots) (`store_true`). |
| `--use_scores` | bool | `False` | Select pivots by per-Gaussian scores instead of normals (`store_true`). |
| `--score_threshold` | float | `0.75` | Score ratio threshold used by `--use_scores`. |
| `--random_pivots` | bool | `False` | Spawn random pivots around each Gaussian (`store_true`). |
| `--random_radius` | float | `2.0` | Sampling radius (in scale units) for `--random_pivots`. |
| `--use_searched_pivots` | bool | `False` | Refine pivots by searching along the normal direction for the SDF zero crossing (`store_true`). |
| `--search_iter` | int | `10` | Number of search iterations for `--use_searched_pivots`. |
| `--search_step_size` | float | `1.0` | Step size of the pivot search. |
| `--mtet_on_cpu` | bool | `False` | Run marching tetrahedra on CPU (`store_true`). |
| `--use_valid_mask` | bool | `False` | Invalidate pivots not visible in any camera frustum (SDF set to 0.5) before tetrahedralization (`store_true`). |
| `--filter_large_edges` | bool | `False` | Filter large edges during mesh extraction (`store_true`). |
| `--collapse_large_edges` | bool | `False` | Collapse large edges during mesh extraction (`store_true`). |
| `--n_binary_steps` | int | `8` | Number of binary-search refinement steps applied to intersection endpoints (0 disables refinement). |
| `--isosurface_value` | float | `-9999.0` | SDF isosurface value. Values `<= -9999` trigger automatic estimation from rendered depth maps (see behavior notes); the code comment notes `0.2` works well for GOF/occupancy SDFs. |
| `--postprocess` | bool | `False` | Remove floaters and small disconnected components (clusters below `max(largest-cluster/1, 50)` triangles) and save a `_post.ply` variant (`store_true`). |

---

## 5. primal_adaptive_meshing_extraction

```bash
python -m gaussian_wrapping.primal_adaptive_meshing_extraction \
    -m <model_path> --input_mesh <mesh.ply> --output_mesh <out.ply> [flags]
```

PoNQ-style primal adaptive meshing: candidate points are sampled from an input mesh,
refined onto the learned occupancy iso-surface by gradient descent on the Gaussian
vector field, filtered by occupancy, tetrahedralized (Delaunay), and classified
inside/outside by the integrated occupancy to produce the final surface.

Behavior notes (verified in code):

- Only the `ours` rasterizer is supported: `--rasterizer radegs` raises
  `ValueError("Always use the Ours rasterizer to apply Primal Adaptive Meshing.")`
  even though `radegs` is an accepted choice value.
- With `ours`, the occupancy threshold actually used is `0.5 - iso_surface_value`
  (printed and written back into `args.iso_surface_value`), so the flag acts as an
  iso-surface shift.
- `--bounding_box_method ground_truth` requires TNT-style crop files in `source_path`
  (`<scene>_COLMAP_SfM.log` and `<scene>.json`) and asserts
  `--bounding_box_scaling 1.0`. `scene` crops to a cube of radius
  `scene_radius * bounding_box_scaling` centered at the origin. `blender` requires
  `--bounding_box_file`, a JSON bounding volume exported by the GW Blender add-on
  (`class_name == "GaussianWrappingBoundingVolume"`), applied as a convex hull.
- By default the script resamples until it collects exactly `--max_points` points
  that survive vacancy filtering (oversampling by `--oversampling_factor` upfront);
  `--no_force_use_all_points` disables this and keeps a single sampling pass.
- `--output_mesh` must end in `.ply` (asserted).
- `--plot_vacancy_histogram` writes `histogram_occupancy_values_<step>.png` files
  into `model_path` after each refinement step.
- **Important:** unlike the other model-directory scripts, `ModelParams` is
  registered **without** `sentinel=True`, so the built-in defaults (including
  `--model_path ""`) always override the values stored in `cfg_args` during the
  `get_combined_args` merge. You must pass `--model_path/-m` (and any other model
  parameter you care about) explicitly on the CLI. See
  [Section 11](#11-get_combined_args-and-cfg_args).
- Seeds are fixed (`numpy`, `torch` = 0) and `cuda:0` is selected.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--iteration` | int | `30000` | Checkpoint iteration to load. |
| `--quiet` | bool | `False` | Suppress console output (`store_true`). |
| `--rasterizer` | str | `"ours"` | Rasterizer used for occupancy integration; choices `ours`, `radegs` (only `ours` actually works). |
| `--disable_mip_filter` | bool | `False` | Disable the 3D mip filter of the loaded Gaussian model (`store_true`). |
| `--input_mesh` | str | *(required)* | Path to the input mesh (`ply`/`obj`) used as the candidate-point source. |
| `--output_mesh` | str | *(required)* | Path to the output mesh; must be a `.ply` file. |
| `--max_points` | int | `1000000` | Maximum number of candidate points to process. |
| `--p_per_tet` | int | `10` | Points per tetrahedron used for the occupancy check (`1` uses tetrahedron barycenters only; `>1` averages random samples per tet). |
| `--n_steps` | int | `10` | Number of gradient-descent refinement steps applied to candidate points. |
| `--vacancy_threshold` | float | `0.1` | Points whose occupancy deviates from the iso-surface value by more than this are filtered out. |
| `--save_candidate_points` | bool | `False` | Save the candidate points to `<output_mesh without .ply>_candidate_points.ply` (`store_true`). |
| `--post_process` | bool | `False` | Post-process the output mesh (remove floaters/disconnected components) before saving (`store_true`). |
| `--bounding_box_method` | str | `"scene"` | Crop-volume source; choices `ground_truth`, `scene`, `blender` (see behavior notes). |
| `--bounding_box_scaling` | float | `1.0` | Scaling applied to the scene-radius-based bounding cube (`scene` method only). |
| `--bounding_box_file` | str | `None` | Path to a bounding-volume JSON exported from the GW Blender add-on (required for `--bounding_box_method=blender`). |
| `--n_neighbors_vector_field` | int | `32` | Number of neighbors used for the gradient-occupancy (vector field) computation. |
| `--mesh_sampling_method` | str | `"proportional_to_camera"` | Surface sampling strategy; choices `surface_even`, `proportional_to_camera`. |
| `--plot_vacancy_histogram` | bool | `False` | Plot a histogram of occupancy values after each refinement step (`store_true`). |
| `--no_force_use_all_points` | bool | `False` | Disable the resampling loop that guarantees `max_points` valid points after vacancy filtering (`store_true`). |
| `--oversampling_factor` | int | `2` | Sample this multiple of `max_points` upfront to reduce the number of occupancy passes; mostly useful when `max_points` is low. |
| `--iso_surface_value` | float | `0.0` | Isosurface value for mesh extraction (interpreted as an occupancy-threshold shift with the `ours` rasterizer; see behavior notes). |
| `--mask_bg_threshold` | float | `0.01` | Mask alpha below this value is treated as definitely-background (occupancy forced to 0). Higher values are more aggressive at removing floaters but may erode fine geometry. |

---

## 6. evaluate_dtu_mesh

```bash
python -m gaussian_wrapping.evaluate_dtu_mesh -m <model_path> --DTU <dtu_eval_data> [flags]
```

DTU mesh evaluation (forked from GOF's `evaluate_dtu_mesh.py`). It loads the trained
scene, aligns the reconstructed mesh `<model_path>/recon_post.ply` to the DTU
coordinate frame by matching training-camera positions against the DTU calibration
poses (`Calibration/cal18/pos_*.txt`, least-squares `best_fit_transform`), exports the
aligned mesh to `<model_path>/recon_aligned.ply`, and shells out to the DTU
evaluation script:

```
python eval/dtu/eval.py --data <aligned_mesh> --scan <scan> --mode mesh \
    --dataset_dir <DTU> --vis_out_dir <model_path>/vis
```

Behavior notes:

- The scan id is **derived** from `source_path` (`int(basename(source_path)[4:])`,
  i.e. a path ending in `scanNN`); the `--scan_id` flag is parsed but never used.
- The subprocess command uses the **relative** path `eval/dtu/eval.py` and the
  interpreter name `python`, so the script must be launched from a working directory
  where that path resolves (repository-layout layouts) with `python` on `PATH`.
- `--skip_train`, `--skip_test`, and `--quiet` are parsed but unused (vestigial flags
  shared with the render-style entry points).
- `ModelParams` is registered with `sentinel=True` (all `ModelParams` flags default to
  `None`; values come from `cfg_args` unless passed explicitly); `PipelineParams` is
  registered normally. See [Section 11](#11-get_combined_args-and-cfg_args) and
  [Section 1](#1-gw-train) for the inherited flags.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--iteration` | int | `30000` | Checkpoint iteration to load for camera recovery. |
| `--skip_train` | bool | `False` | Parsed but unused (`store_true`). |
| `--skip_test` | bool | `False` | Parsed but unused (`store_true`). |
| `--quiet` | bool | `False` | Parsed but unused (`store_true`). |
| `--scan_id` | str | `None` | Parsed but unused; the scan id is derived from `source_path` instead. |
| `--DTU` | str | `None` | Path to the ground-truth DTU evaluation data (passed as `--dataset_dir` to `eval/dtu/eval.py`). |

---

## 7. mesh_decimate

```bash
blender --background --python gaussian_wrapping/mesh_decimate.py -- \
    --in <input.ply> [--out <output.ply>] [--ratio 0.3]
```

Decimates a PLY mesh using Blender's Decimate modifier. The script must be executed
**inside Blender** (`bpy` is imported at module level); arguments after the `--`
separator are forwarded to argparse.

Behavior notes:

- `--in` and `--out` are parsed into `in_path` / `out_path` via `dest`.
- **Known issue:** when `--out` is provided, the local `out_path` variable is never
  assigned from it (only the `args.out_path is None` branch assigns it), so the
  script crashes with `NameError: name 'out_path' is not defined`. In practice only
  the default output path works: the input path with `.ply` replaced by
  `_decimated_with_blender.ply`.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--in` | str | *(required)* | Input PLY path (e.g. `./output/Ignatius/mesh_learnable_sdf.ply`). |
| `--out` | str | `None` | Output PLY path. Currently unused due to the bug above; the default is `<in_path with .ply replaced by _decimated_with_blender.ply>`. |
| `--ratio` | float | `0.3` | Decimation ratio applied by the Blender Decimate modifier. |

---

## 8. texture_mesh

```bash
python -m gaussian_wrapping.texture_mesh -m <model_path> --mesh <mesh.ply> [flags]
```

Refines the vertex colors of an extracted mesh against the trained Gaussian
renderings. The scene is loaded at `--iteration`, all training cameras are rendered
with the selected rasterizer at `--sh_degree_for_texturing`, and the mesh's vertex
colors are optimized with Adam by minimizing an L1 + D-SSIM loss between the
frustum-culled, rasterized mesh and the Gaussian renderings. The refined mesh is
exported to `<model_path>/<mesh_name>_texture_refined_<last_iter>.<ext>`.

Behavior notes:

- `--mesh` has no default; it must be provided or mesh loading fails.
- `--use_scalable_renderer` chooses between `ScalableMeshRenderer` and
  `MeshRenderer` for the mesh rasterization.
- `ModelParams` is registered with `sentinel=True` (all `ModelParams` flags default to
  `None`; values come from `cfg_args` unless passed explicitly); `PipelineParams` is
  registered normally. See [Section 11](#11-get_combined_args-and-cfg_args) and
  [Section 1](#1-gw-train).
- Seeds are fixed (`random`, `numpy`, `torch` = 0) and `cuda:0` is selected.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--iteration` | int | `30000` | Checkpoint iteration whose Gaussians produce the target renderings. |
| `--quiet` | bool | `False` | Suppress console output (`store_true`). |
| `--rasterizer` | str | `"ours"` | Rasterizer used to render the target images; choices `radegs`, `ours`. |
| `--mesh` | str | `None` | Path to the mesh to texture (any trimesh-supported format; the extension is preserved for the output). |
| `--n_iter` | int | `1000` | Number of texture-refinement iterations. |
| `--lambda_dssim` | float | `0.2` | Weight of the D-SSIM term in the texture loss. |
| `--lr` | float | `0.0025` | Learning rate for the vertex-color parameters. |
| `--sh_degree_for_texturing` | int | `0` | SH degree used when rendering the target images (0 = diffuse-only targets). |
| `--use_scalable_renderer` | bool (`--flag`/`--no-flag`) | `True` | Use the scalable mesh renderer (tile-based) instead of the plain mesh renderer. |

---

## 9. metrics

```bash
python -m gaussian_wrapping.metrics -m <model_path> [<model_path> ...]
```

Computes SSIM, PSNR, and LPIPS (VGG) between rendered and ground-truth images for
one or more model directories. For each `<model_path>`, it iterates over the methods
(subdirectories) found in `<model_path>/test/`, reads paired images from
`<method>/renders/` and `<method>/gt/`, and writes `<model_path>/results.json`
(per-scene aggregate means) and `<model_path>/per_view.json` (per-image values).
Errors on a scene directory are caught and reported as
`Unable to compute metrics for model <dir>`.

This is the downstream consumer of the output produced by
[Section 2](#2-gw-render) (`test/<method>/{renders,gt}`). Runs on `cuda:0`.

| Flag | Shorthand | Type | Default | Description |
|------|-----------|------|---------|-------------|
| `--model_paths` | `-m` | str (nargs +) | *(required)* | One or more model directories to evaluate. |

---

## 10. Tanks and Temples toolbox (eval/tnt)

The scripts in `gaussian_wrapping/eval/tnt/` are the Tanks and Temples benchmark
toolbox (from the official website toolbox, adapted). They use flat imports
(`from config import scenes_tau_dict`, `from registration import ...`), so they must
be run from inside the `eval/tnt` directory:

```bash
cd gaussian_wrapping/eval/tnt
python run.py --dataset-dir <data>/<Scene> --traj-path <traj> --ply-path <mesh.ply>
```

Expected dataset layout (per scene `<X>` inside `--dataset-dir`): `<X>.ply`
(ground-truth point cloud), `<X>.json` (crop volume), `<X>_trans.txt` (alignment),
`<X>_COLMAP_SfM.log` (reference trajectory), and optionally
`<X>_mapping_reference.txt`. The per-scene tau threshold comes from
`config.scenes_tau_dict`. Related helpers in the same directory (`help_func.py`,
`trajectory_io.py`, `util.py`, `plot.py`, `config.py`, `evaluate_single_scene.py`)
are library/support modules for the scripts below; see
[Evaluation](evaluation.md) for the workflow.

### 10.1 run.py

```bash
cd gaussian_wrapping/eval/tnt
python run.py --dataset-dir <data>/<Scene> --traj-path <traj.(log|npy|json)> --ply-path <recon.ply> [--out-dir <dir>]
```

Full TNT evaluation of a reconstruction: loads the reconstruction PLY as a point
cloud augmented with face-centroid samples, registers the estimated trajectory to the
COLMAP reference trajectory (RANSAC via `trajectory_alignment`, then three ICP
refinement stages via `registration_vol_ds` / `registration_unif` inside the crop
volume), and computes precision / recall / F-score at the scene's tau
(`EvaluateHisto`), writing distance histograms, precision/recall PLYs, and the
F-score plot into `--out-dir`.

Trajectory formats: `.log` (TNT format, via `read_trajectory`), `.npy` (array of 4x4
poses), or `.json` (Instant-NGP/SDFstudio style). Note: the `.json` branch of
`run.py` references `torch` and `auto_orient_and_center_poses`, whose imports are
commented out in this file, so `.json` trajectories currently raise `NameError` here
(they work in `compute_bbox_for_mesh.py`, which imports them properly).

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--dataset-dir` | str | *(required)* | Path to a dataset/scene directory containing `X.json`, `X.ply`, ... |
| `--traj-path` | str | *(required)* | Path to the estimated trajectory file (`.log`, `.npy`, or `.json`). |
| `--ply-path` | str | *(required)* | Path to the reconstruction PLY file. |
| `--out-dir` | str | `""` | Output directory; empty means an `evaluation/` directory next to the PLY file. |
| `--view-crop` | int | `0` | Whether to view the cropped point cloud after alignment. Parsed but then unconditionally overwritten to `False`, so it has no effect. |

### 10.2 cull_mesh.py

```bash
cd gaussian_wrapping/eval/tnt
python cull_mesh.py --traj-path <traj> --ply-path <mesh.ply>
```

Culls a reconstructed mesh against the camera frusta before TNT evaluation. Builds a
`Mesher` with hard-coded Tanks and Temples camera parameters (`H=1080, W=1920`,
`fx=1163.8678928442187`, `fy=1172.793101201448`, `cx=962.3120628412543`,
`cy=542.0667209577691`, `far=20.0`, `points_batch_size=5e5`), loads the trajectory
(`.npy`, `.json`, or `.log`; 3x4 poses are promoted to 4x4), and renders depth-based
visibility masks to remove unseen geometry (connected-component filtering included).

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--traj-path` | str | *(required)* | Path to the trajectory file (`.log`, `.npy`, or `.json`). |
| `--ply-path` | str | *(required)* | Path to the reconstruction PLY file to cull. |

### 10.3 compute_bbox_for_mesh.py

```bash
cd gaussian_wrapping/eval/tnt
python compute_bbox_for_mesh.py --dataset-dir <data>/<Scene> --traj-path <traj> --ply-path <recon.ply>
```

Utility variant of `run.py` that stops after the trajectory alignment and prints the
axis-aligned bounding box of the ground-truth point cloud expressed in the
reconstruction's coordinate frame (it applies the inverse trajectory transform to the
GT points and prints min/max coordinates). Unlike `run.py`, it does not add
face-centroid samples, run ICP refinement, or compute metrics. Its `.json`
trajectory branch works (it imports `torch` and `auto_orient_and_center_poses`).

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--dataset-dir` | str | *(required)* | Path to a dataset/scene directory containing `X.json`, `X.ply`, ... |
| `--traj-path` | str | *(required)* | Path to the estimated trajectory file. |
| `--ply-path` | str | *(required)* | Path to the reconstruction PLY file (used only to derive the default `--out-dir`). |
| `--out-dir` | str | `""` | Output directory; empty means an `evaluation/` directory next to the PLY file (created but unused by the bbox computation). |
| `--view-crop` | int | `0` | Parsed but unconditionally overwritten to `False`; no effect. |

### 10.4 evaluation.py (library module)

No CLI: this module defines no argparse and no `__main__` block. It provides the
scoring internals used by `run.py`:

- `EvaluateHisto(source, target, trans, crop_volume, voxel_size, threshold,
  filename_mvs, plot_stretch, scene_name, view_crop)` — crops and downsamples both
  point clouds, computes point-to-point distances, and returns
  `[precision, recall, fscore, edges_source, cum_source, edges_target, cum_target]`,
  writing `precision`/`recall` colored PLYs and histogram files.
- `get_f1_score_histo2(threshold, filename_mvs, plot_stretch, distance1, distance2,
  verbose=True)` — computes precision/recall/F-score from the two distance
  histograms and plots the P/R curve.
- `read_alignment_transformation(filename)` — loads a 4x4 alignment from a JSON file.
- `write_color_distances(path, pcd, distances, max_distance)` — color-codes a point
  cloud by distance and writes it.

### 10.5 registration.py (library module)

No CLI: no argparse and no `__main__` block. Provides the trajectory registration
internals used by `run.py` and `compute_bbox_for_mesh.py`:

- `trajectory_alignment(map_file, traj_to_register, gt_traj_col, gt_trans, scene)` —
  rough RANSAC registration of the estimated trajectory point cloud against the
  COLMAP reference trajectory (applying `gt_trans` first); for trajectories longer
  than 1600 poses with a mapping file, subsamples via `read_mapping` /
  `gen_sparse_trajectory`.
- `registration_vol_ds(source, gt_target, init_trans, crop_volume, voxel_size,
  threshold, max_itr, verbose=True)` — crop + voxel downsample + ICP refinement.
- `registration_unif(source, gt_target, init_trans, crop_volume, threshold, max_itr,
  max_size=4 * MAX_POINT_NUMBER (4e6), verbose=True)` — uniform-downsample ICP
  refinement.
- `crop_and_downsample(pcd, crop_volume, down_sample_method="voxel", voxel_size=0.01,
  trans=identity)` — cropping/downsampling helper.
- `read_mapping(filename)` / `gen_sparse_trajectory(mapping, f_trajectory)` —
  mapping-file utilities.
- Re-exports `read_trajectory` and `convert_trajectory_to_pointcloud` from
  `trajectory_io`.

### 10.6 DTU evaluation (eval/dtu/eval.py)

```bash
python gaussian_wrapping/eval/dtu/eval.py --data <mesh_or_pcd> --scan <id> --mode mesh \
    --dataset_dir <DTU_data> --vis_out_dir <dir>
```

Standalone DTU evaluation script (adapted from DTUeval-python), also invoked
internally by `evaluate_dtu_mesh` ([Section 6](#6-evaluate_dtu_mesh)). Samples the
input mesh or point cloud, computes point-to-surface distances against the DTU
ground truth, and reports accuracy / completeness / overall distance and F-score at
the standard DTU thresholds.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--data` | str | `"data_in.ply"` | Path to the reconstruction (mesh in `--mode mesh`, point cloud in `--mode pcd`). |
| `--scan` | int | `1` | DTU scan id. |
| `--mode` | str | `"mesh"` | Input type; choices `mesh`, `pcd`. |
| `--dataset_dir` | str | `"."` | Directory containing the DTU ground-truth data (`Points/stl/stlXXX.mat` etc.). |
| `--vis_out_dir` | str | `"."` | Directory for the visualization point clouds (`visualize_threshold`-filtered). |
| `--downsample_density` | float | `0.2` | Downsampling density (also the accuracy/completeness threshold). |
| `--patch_size` | float | `60` | Patch size used for the outlier-trimming step. |
| `--max_dist` | float | `20` | Maximum distance considered when computing distances. |
| `--visualize_threshold` | float | `10` | Distance threshold for the exported visualization point clouds. |

---

## 11. get_combined_args and cfg_args

Several scripts operate on an already-trained model directory. They build their
argparse parser, then call `get_combined_args(parser)` from
`gaussian_wrapping/arguments/__init__.py` instead of calling `parse_args()` directly.
The helper:

1. Parses the command line with the script's parser.
2. Reads `<args.model_path>/cfg_args` — a file written by `train.py`
   (`prepare_output_and_logger`) containing `str(Namespace(**vars(dataset)))`, i.e. a
   `repr` of the training run's resolved `ModelParams` namespace (including the
   absolute `source_path` and the resolved `model_path`).
3. `eval()`s that string (note: this executes the file content as Python, so only
   load `cfg_args` from directories you trust) and starts the merged namespace from
   the config values.
4. Overwrites every entry with the corresponding CLI value **when that CLI value is
   not `None`**, and returns the merged `Namespace`.

The practical consequence depends on how the script registered its parameter groups:

- Scripts registering `ModelParams(parser, sentinel=True)` — `render.py`,
  `pivot_based_mesh_extraction.py`, `evaluate_dtu_mesh.py`, `texture_mesh.py` — get
  `None` defaults for every `ModelParams` flag, so `cfg_args` values survive unless a
  flag is explicitly passed. `--model_path/-m` must still be passed explicitly,
  because the helper needs it to locate `cfg_args` in the first place (if it is
  `None`, `os.path.join` raises `TypeError`, which is caught and the config is
  silently skipped, leaving all `ModelParams` values `None`).
- Scripts registering `ModelParams(parser)` without the sentinel —
  `primal_adaptive_meshing_extraction.py` — get real defaults (e.g. `--model_path ""`),
  which are non-`None` and therefore **always override** `cfg_args`. For these
  scripts every model parameter you rely on must be passed on the CLI.
- `PipelineParams` and script-specific flags (`--iteration`, `--rasterizer`, ...)
  always use their CLI defaults; `cfg_args` does not contain them.

Scripts using `get_combined_args`: `render.py`, `pivot_based_mesh_extraction.py`,
`primal_adaptive_meshing_extraction.py`, `evaluate_dtu_mesh.py`, `texture_mesh.py`.
`train.py` does not use it — it is the producer of `cfg_args`.

---

## 12. See also

- [API reference](api-reference.md) — the Python API (`gaussian_wrapping.api`),
  including the library entry point that shares `training()` with `gw-train`.
- [Training guide](training-guide.md) — end-to-end training workflows and recipes.
- [Regularization configs](regularization-configs.md) — schema and defaults of the
  YAML files selected by `--normal_field_config`, `--multiview_config`,
  `--milo_config`, and `--depth_order_config`.
- [Mesh extraction](mesh-extraction.md) — mesh extraction methods and pipelines.
- [Evaluation](evaluation.md) — evaluation workflows (DTU, Tanks and Temples).
- [Architecture](architecture.md) — package architecture and module overview.
