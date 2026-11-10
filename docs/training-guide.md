# Training Guide

> Source of truth: `gaussian_wrapping/train.py` (`training()`, `main()`,
> `prepare_output_and_logger`, `resolve_rasterizer`, `load_regularization_configs`),
> `gaussian_wrapping/api.py` (`run_training`), `gaussian_wrapping/utils/log_utils.py`.

This guide explains what a GaussianWrapping training run actually does, in
order, and how every configuration knob affects it. Flag-by-flag reference:
[cli-reference.md](cli-reference.md). Field-by-field API reference:
[api-reference.md](api-reference.md).

## Two entry points, one loop

| Entry point | Invocation | GUI | Notes |
|-------------|------------|-----|-------|
| CLI | `gw-train …` / `python -m gaussian_wrapping.train …` | Always started (`network_gui.init(ip, port)`) | `--port -1` picks a random port in `[5000, 9000)` |
| Python API | `gaussian_wrapping.api.run_training(TrainingConfig(...))` | Only if `enable_gui=True` | Writes the same artifacts; returns `TrainingResult` |

Both paths call the same `training()` loop with explicitly resolved rasterizer
callables (`resolve_rasterizer`) and the same regularization YAML loader
(`load_regularization_configs`). Both write `time.txt` at the end.

## Run lifecycle

1. **Output directory** (`prepare_output_and_logger`): if `model_path` is
   empty, a unique path `./output/<10-char-uuid>` is generated (or
   `./output/<OAR_JOB_ID>` when the `OAR_JOB_ID` environment variable is set).
   The directory is created and `<model_path>/cfg_args` is written — a
   `repr` of the dataset (`ModelParams`) namespace. Downstream scripts
   (`render.py`, extraction, evaluation) read this file via
   `get_combined_args` to recover the training settings.
2. **WandB**: initialized only when the WandB package is importable **and**
   `wandb_project` **and** `wandb_entity` are all set; otherwise the run
   prints `[INFO] WandB not found, skipping logging.`
3. **Model construction** (`GaussianModel`): the constructor flags are derived
   from the config:
   - `sh_degree` from `dataset.sh_degree`,
   - `use_mip_filter = not disable_mip_filter`,
   - `learn_occupancy = milo` (learnable SDF used by MILo),
   - `use_appearance_network = decoupled_appearance`,
   - `n_gaussian_features`: `1` if the normal-field config sets
     `use_smallest_axis`, else `4` (when `use_normal_field`), else `0`,
   - `n_pivots_per_gaussian`: from the MILo config when `milo`, else `2`,
   - `use_radegs_densification=True` (always),
   - `use_unbounded_opacity`, `use_exposure_compensation=exposure_compensation`.
4. **Scene loading** (`Scene(dataset, gaussians)`): reads COLMAP/Blender data,
   builds cameras, initializes Gaussians from the initial point cloud, writes
   `input.ply` and `cameras.json` into `model_path` (the dataset itself is
   never modified; `save_init_ply` additionally persists `points3D.ply`).
5. **Exposure compensation**: when enabled, one exposure parameter per training
   camera is initialized (`gaussians.initialize_exposure_compensation`).
6. **Optimizer setup** (`gaussians.training_setup(opt)`): one param group per
   Gaussian attribute with the learning rates from the optimization config;
   the loop prints every group, its LR and tensor shapes, plus the total
   optimizable parameter count.
7. **Checkpoint resume**: when `start_checkpoint` is given,
   `(model_params, first_iter)` is loaded and regularizer start iterations
   (`normal_field_config["start_iter"]`, `milo_config["start_iter"]`,
   `multiview_config["start_multiview"]`) are bumped to `first_iter + 1` if
   they would otherwise be in the past.
8. **Regularizer state initialization** (each only when enabled):
   `initialize_normal_field`, `initialize_multiview_regularization`,
   `initialize_mesh_in_the_loop_regularization`,
   `initialize_depth_order_supervision` (loads Depth-Anything-V2 priors).
9. **3D mip filter**: computed once over all training cameras before the loop
   when enabled.
10. **Optimization loop** for `iteration` in `[first_iter, iterations]`
    (details below).
11. **Finalization**: prints the final Gaussian count, finishes the WandB run,
    and the entry point writes `time.txt` with the wall-clock duration.

## The optimization loop, iteration by iteration

### Viewpoint selection

Training cameras are consumed in random order without replacement; the stack is
refilled when exhausted (each camera is seen once per "epoch").

### Rendering

A single render call produces everything the losses need:

```python
render_pkg = render_fn(viewpoint_cam, gaussians, pipe, bg,
                       require_coord=False,
                       require_depth=render_depth_in_forward_pass)
```

`render_depth_in_forward_pass` is true when any regularizer is active at this
iteration (depth-normal, normal field, depth order, multiview, or MILo).
`render_pkg` contains `render`, `viewspace_points`, `visibility_filter`,
`radii`, and (when depth is required) `expected_depth`, `median_depth`,
`normal`.

The rasterizer is chosen once at startup:

| `rasterizer` | Render function | Effect |
|--------------|-----------------|--------|
| `radegs` (default) | `gaussian_renderer.radegs.render_radegs` | Returns distinct expected and median depth maps. Best-looking meshes; slightly slower. |
| `ours` | `gaussian_renderer.ours.render_ours` | Faster, better metrics. Force-sets `mask_depth_normal = True` at startup. |

`multiview_rasterizer = "ours"` renders the multiview loss with
`render_ours` instead of reusing the primary render package.

### Ground truth and background

`gt_image` is the camera's stored image; when the camera has a `gt_mask`, the
GT is alpha-composited over the background color
(`gt_image * mask + bg * (1 - mask)`). The background is black or white per
`white_background`, or random per-iteration when `random_background` is set.

### Loss composition

Total loss = rendering loss + all active regularizers.

1. **Rendering loss** — `(1 - lambda_dssim) * L1 + lambda_dssim * (1 - SSIM)`,
   with SSIM computed by `fused_ssim` (valid padding). The L1 term is
   appearance-aware when `decoupled_appearance` or `exposure_compensation` is
   on (`L1_loss_appearance`), otherwise plain `l1_loss`.
2. **Depth-normal consistency** (active from `regularization_from_iter`):
   - With `mask_depth_normal`: blends `(1 - 0.6) * expected_depth + 0.6 * median_depth`
     where the median depth is valid, converts depth to normals
     (`depth_to_normal_with_mask`) and penalizes `1 - ⟨rendered_normal, depth_normal⟩`
     on valid pixels, weighted by `lambda_depth_normal`.
   - Without the mask: `depth_to_normal` over expected/median depth; if it
     returns two normal maps they are blended with the same 0.6 ratio.
   - `weight_by_img_grad` multiplies the map by a squared, clamped image-gradient
     weight (PGSR-style) before the mean.
3. **Minimum-scale loss** (`use_scale_loss`, PGSR-style): mean of the smallest
   scale component of visible Gaussians, weighted by `scale_loss_weight`.
4. **Depth-order loss** (`depth_order`): compares rendered depth (blended by
   `depth_ratio`, or pure median depth when `depth_ratio >= 1`) against
   Depth-Anything-V2 priors; can be switched off after
   `deactivate_depth_order_after` iterations.
5. **Multiview loss** (`multiview`, from `multiview_config["start_multiview"]`):
   NCC/geometric-consistency loss computed by
   `compute_multiview_regularization`, multiplied by `multiview_factor`.
6. **Normal-field loss** (`use_normal_field`, from
   `normal_field_config["start_iter"]`): alignment, front-pivot visibility,
   back-pivot occlusion, Gaussian flattening and SDF-consistency terms computed
   by `compute_normal_field_regularization`. With
   `detach_gaussian_rendering`, the render package is detached first.
7. **MILo loss** (`milo`, from `milo_config["start_iter"]`): mesh depth/normal
   and occupied-center terms from `compute_mesh_in_the_loop_regularization`
   (extracts a mesh in the loop).

All weights and schedules inside the regularizers come from their YAML configs —
see [regularization-configs.md](regularization-configs.md).

### Densification and pruning (before `densify_until_iter`)

- Statistics accumulate every iteration
  (`add_densification_stats_radegs`, on absolute gradients —
  `use_abs_grad_for_densification` is forced `True`).
- Every `densification_interval` iterations between `densify_from_iter` and
  `densify_until_iter`: `densify_and_prune_radegs(densify_grad_threshold, 0.05,
  cameras_extent, size_threshold)` where `size_threshold = 20` only when
  `iteration > opacity_reset_interval` **and** `use_max_size_threshold`;
  otherwise `None`. The 3D mip filter is recomputed afterwards (or reset when
  the mip filter is disabled).
- Every `opacity_reset_interval` iterations (and at `densify_from_iter` when
  `white_background`): `reset_opacity()`.

### Normal-field densification / pruning (own schedule)

Independent of the 3DGS densification above, gated by the normal-field config
(`use_densification`, `start_iter_densification`, `end_iter_densification`,
`densify_every_n_iterations`, and the pruning equivalents). Note the gates
compare `iteration + 1` against the config windows. Densification honours
`N_max_gaussians` (highest-error Gaussians are added first up to the cap) and
can reset normal features afterwards (`reset_normals_after_densification`,
`reset_normal_directions`, `reset_normal_signs`).

### Explicit non-maximal pruning

At every iteration listed in `non_maximal_pruning_iterations`,
`prune_non_maximal_gaussians` removes Gaussians that are never maximal in any
view.

### Mip-filter maintenance (after `densify_until_iter`)

The 3D mip filter is recomputed at `warn_until_iter` and every
`update_mip_filter_every` iterations, skipping updates within the last
`update_mip_filter_every` iterations of the run.

### Optimizer step

- With `decoupled_appearance` or `exposure_compensation`: a plain
  `optimizer.step()`.
- Otherwise: `optimizer.step(visible, radii.shape[0])` where `visible = radii > 0`
  (only visible Gaussians get their LR applied).

### Checkpoints and housekeeping

- At each `checkpoint_iterations` value: `chkpnt<iteration>.pth` (full model +
  optimizer state, loadable via `start_checkpoint`).
- Every 100 iterations: `torch.cuda.empty_cache()` + `gc.collect()`.
- Progress: a `tqdm` bar with an EMA postfix of the active losses; WandB logs
  when configured; `log_interval` controls verbose per-iteration logging
  (including test-set renders at `test_iterations`).

## Output artifacts

A completed run leaves in `model_path`:

| File | Written by | Contents |
|------|------------|----------|
| `cfg_args` | `prepare_output_and_logger` | `repr` of the dataset namespace; consumed by `get_combined_args` in downstream scripts |
| `input.ply` | `Scene` | Initial point cloud (built in memory; the dataset is never written to) |
| `cameras.json` | `Scene` | Camera metadata for viewers |
| `points3D.ply` | `Scene` (only with `save_init_ply`) | Copy of the initial point cloud |
| `point_cloud/iteration_<N>/point_cloud.ply` | training loop at `save_iterations` (+ final iteration) | Trained Gaussians |
| `chkpnt<N>.pth` | training loop at `checkpoint_iterations` | Full optimizer/model state for resuming |
| `time.txt` | CLI `main()` / `run_training` | Wall-clock training duration in seconds |
| `run_manifest.json` | downstream `GW_experiments` harness only | Reproducibility record (not written by this package) |

The `GW_experiments` harness validates the first five artifacts (plus
`time.txt`) after every run — see its `docs/utils-reference.md`.

## Recipes

### Smoke test (minutes)

```bash
gw-train -s <DATASET> -m /tmp/smoke --iterations 200 --densify_until_iter 100 \
    --regularization_from_iter 150 --save_iterations 200
```

### Avoid OOM on large scenes

- Set `--N_max_gaussians` (e.g. `6000000`) to cap the normal-field densification.
- Train at reduced resolution: `-r 1` or `-r 2`.
- Keep images on the host: `--data_device cpu`.

### Resume an interrupted run

```bash
gw-train -s <DATASET> -m <OUTPUT> --start_checkpoint <OUTPUT>/chkpnt7000.pth
```

Regularizer schedules resume past the checkpoint iteration automatically.

### Disable the normal field (plain RaDe-GS-style training)

```bash
gw-train -s <DATASET> -m <OUTPUT> --no_normal_field
```

### Track on WandB

```bash
gw-train -s <DATASET> -m <OUTPUT> \
    --wandb_project <project> --wandb_entity <entity> --wandb_name <run>
```

## See also

- [regularization-configs.md](regularization-configs.md) — every YAML key of every regularizer.
- [mesh-extraction.md](mesh-extraction.md) — what to do with the trained model.
- [architecture.md](architecture.md) — module-level view of the loop.
