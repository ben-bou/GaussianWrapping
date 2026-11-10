# Regularization & Behavior Configs

This page is an exhaustive reference for the YAML configs shipped in `gaussian_wrapping/configs/`. These files control when each regularizer activates, how it behaves, and what weights it uses. They are plain dictionaries loaded with `yaml.safe_load` — there is no schema validation, so every key listed here is read directly by the consumer code (`config["key"]`), and a typo surfaces as a `KeyError` at the point of use.

Two kinds of entries live under `configs/`:

- Four **YAML groups** wired into training via CLI flags: `normal_field/`, `multiview/`, `mesh_in_the_loop/`, `depth_order/` (7 YAML files).
- A legacy **`mesh/` group** (37 YAML files) inherited from the standalone MILo codebase, plus one non-YAML memo file `configs/fast`. Neither is loaded by any code in this package (details in [`mesh group`](#mesh-group-legacy-milo-files) and [`configs/fast`](#configsfast)).

> Source of truth:
>
> - Loader: `gaussian_wrapping/train.py` — `load_regularization_configs()` (the only `yaml.safe_load` call site in the package)
> - Library entry point: `gaussian_wrapping/api.py` — `TrainingConfig`, `run_training()`
> - Consumers: `gaussian_wrapping/regularization/regularizer/{normal_field,multiview,mesh_in_the_loop,depth_order}.py`, `gaussian_wrapping/regularization/{normal_field,multiview,multiview_gggs,mesh_in_the_loop}.py`, `gaussian_wrapping/regularization/depth/{depth_order,depthanythingv2}.py`, `gaussian_wrapping/densification/normal_error.py`, `gaussian_wrapping/scene/gaussian_model.py`, `gaussian_wrapping/train.py` (training loop)
> - Legacy MILo machinery matching the `mesh/` keys: `gaussian_wrapping/regularization/regularizer/{snapping_points,snapping_edges}.py`, `gaussian_wrapping/regularization/sdf/{learnable,depth_fusion,integration}.py`, `gaussian_wrapping/utils/log_utils.py` (`log_training_progress`), `gaussian_wrapping/extraction/pivots.py`

Related: [CLI reference](cli-reference.md), [Training guide](training-guide.md), [API reference](api-reference.md), [Mesh extraction](mesh-extraction.md), [Architecture](architecture.md).

## How configs work

### Loading and selection

`load_regularization_configs(args)` in `gaussian_wrapping/train.py` loads `gaussian_wrapping/configs/<group>/<name>.yaml` for each group whose enabling flag is on, and returns `(depth_order_config, normal_field_config, multiview_config, milo_config)`. The same function is called by `gaussian_wrapping.api.run_training()`, so the CLI and the Python API share one loader.

| Group | Enabling flag | Config-name flag | CLI default config | On by default? |
|---|---|---|---|---|
| `normal_field` | `--use_normal_field` (implicit) / `--no_normal_field` to disable | `--normal_field_config` | `default_regular_densification` | Yes (`use_normal_field = not no_normal_field`) |
| `multiview` | `--multiview` / `--no-multiview` | `--multiview_config` | `fast` | Yes |
| `mesh_in_the_loop` | `--milo` | `--milo_config` | `default_regular_densification` | No |
| `depth_order` | `--depth_order` | `--depth_order_config` | `default` | No |

In `api.py`, `TrainingConfig` exposes the same options as typed fields (`use_normal_field=True`, `normal_field_config`, `multiview=True`, `multiview_config="fast"`, `milo=False`, `milo_config`, `depth_order=False`, `depth_order_config`).

### What happens when a regularizer is disabled

If the enabling flag is off, the loader skips the group entirely and the returned config is `None`. Nothing is read from disk, so edits to the YAML have no effect until the flag is enabled. Related side effects of disabling:

- `use_normal_field=False` builds the model with `n_gaussian_features=0` (no learnable normal features at all) — see `train.py` and `scene/gaussian_model.py`. The regularizer, its densification and its pruning are all skipped.
- `milo=False` builds the model with `learn_occupancy=False` and `n_pivots_per_gaussian=2` (the config value is only consulted when MILo is on).
- `depth_order=False` skips Depth-Anything-V2 prior construction entirely (no model download/inference).

### Scheduling conventions

- Activation gates are evaluated every iteration in the training loop: `normal_field_kick_on = use_normal_field and iteration >= normal_field_config["start_iter"]`, `multiview_kick_on = multiview and iteration >= multiview_config["start_multiview"]`, `milo_kick_on = milo and iteration >= milo_config["start_iter"]`, and `depth_order_kick_on = depth_order` (further gated by its weight schedule and `deactivate_depth_order_after`).
- Normal-field densification and pruning windows use `iteration + 1` in their comparisons (`train.py`, "Normal Field Densification" / "Normal field pruning" blocks). A config value of `start_iter_densification: 22_000` therefore first triggers when the loop variable reads 21_999; console logs report the `iteration + 1` value.
- Resuming from a checkpoint shifts start points forward: if `first_iter` exceeds `start_iter` / `start_multiview`, the loaded config dict is patched to `first_iter + 1` in `training()` — so a resumed run does not re-fire a window that already ended.
- The weight itself is not part of these YAMLs; loss weights live inside the configs, while global knobs like `--lambda_depth_normal`, `--regularization_from_iter` (depth-normal consistency), `--multiview_factor`, `--N_max_gaussians` are CLI-only (see [CLI reference](cli-reference.md)).

### Validation

There is none. Keys are looked up directly; unknown or misspelled keys fail with `KeyError` at the first consumer call, and missing-but-referenced keys behave the same. Extra keys in a YAML are silently ignored.

## `normal_field` group

Files: `default.yaml`, `default_regular_densification.yaml`. Both drive the learnable normal features ("oriented Gaussians") — the core of GaussianWrapping. The config selects the per-Gaussian feature representation (`use_smallest_axis`), the alignment loss, the pivot-based occlusion/visibility/flattening losses, and the normal-error-driven densification and pruning schedules.

Selection: `--normal_field_config <name>` (default `default_regular_densification`), active unless `--no_normal_field`.

| Config | Purpose | Distinctive settings |
|---|---|---|
| `default` | Schedule aligned with the standard 30k-iteration pipeline: normal field starts early (right after the default `densify_until_iter=15_000` window begins), densification window 10_000–14_000, densifies the top 15% normal-error Gaussians. | `start_iter: 8_001`, densification 10_000–14_000, `densification_normal_errors_quantile: 0.15`, pruning 10_500–10_505 |
| `default_regular_densification` (CLI default) | Later-start schedule: the normal field kicks in after standard RaDe-GS densification has finished, densification window 22_000–26_000, densifies only the top 5%. | `start_iter: 20_001`, densification 22_000–26_000, `densification_normal_errors_quantile: 0.05`, pruning 22_500–22_505 |

### Key reference

All 33 keys below exist in **both** files. Where the two configs differ, both values are shown as `default` / `default_regular_densification`.

| Key | Type | Value (default / default_regular_densification) | Description |
|---|---|---|---|
| `start_iter` | int | `8_001` / `20_001` | Iteration at which the normal field activates (`normal_field_kick_on`, train.py), normal features are initialized (`gaussians.reset_normal_features()`), and logging snapshots trigger. Shifted to `first_iter + 1` when resuming past it. |
| `std_factor` | float | `3.0` / `3.0` | Pivot placement: each Gaussian gets two pivots at ±`std_factor` × std(along normal) from its center, via `get_pivots_from_normals` (`regularization/normal_field.py`). Used by the pivot occlusion/visibility losses. |
| `use_normal_field_alignment` | bool | `true` / `true` | Master switch for the alignment loss (rendered Gaussian normal map vs. rendered normals / depth-derived normals). Must be `true` for any of the `align_with_*` keys to matter. |
| `align_with_rendered_normals` | bool | `false` / `false` | Adds `mean(1 − dot(rendered_gaussian_normal, rendered_normal))` to the alignment loss. |
| `align_with_rendered_median_depths` | bool | `true` / `true` | Adds the depth-normal disagreement between the Gaussian normal map and normals derived from the **median** depth map, weighted by `depth_ratio_for_alignment`. Respects the `--mask_depth_normal` validity mask. |
| `align_with_rendered_expected_depths` | bool | `false` / `false` | Same but against normals derived from the **expected** (alpha-composited) depth map, weighted by `1 − depth_ratio_for_alignment`. |
| `depth_ratio_for_alignment` | float | `0.6` / `0.6` | Split of the alignment weight between median-depth and expected-depth terms: median gets `depth_ratio_for_alignment`, expected gets `1 − depth_ratio_for_alignment`. |
| `normal_field_alignment_weight` | float | `0.05` / `0.05` | Final multiplier on the summed alignment loss. |
| `use_smallest_axis` | bool | `false` / `false` | Feature representation: `true` → 1 learnable feature (normal forced to the smallest Gaussian axis), `false` → 4 learnable features (full learnable normal direction + sign). Decided once at model construction (`n_gaussian_features` in train.py). |
| `enforce_back_pivots_to_be_occluded` | bool | `false` / `false` | Loss pushing the back pivot (along −normal) behind the median depth surface (positive signed distance), so back-facing pivots stay occluded. Comment in the file: "Should be True?". |
| `enforce_back_pivots_to_be_occluded_weight` | float | `0.005` / `0.005` | Weight of the back-pivot occlusion hinge (`clamp_min(0).mean()`, normalized by min Gaussian scale). |
| `enforce_front_pivots_to_be_visible` | bool | `false` / `false` | Loss pushing the front pivot (along +normal) in front of the depth surface (negative signed distance), restricted to Gaussians actually rendered. Comment: "Should be True?". |
| `enforce_front_pivots_to_be_visible_weight` | float | `0.005` / `0.005` | Weight of the front-pivot visibility hinge (`clamp_max(0)` summed over rendered Gaussians). |
| `enforce_gaussian_flattening` | bool | `false` / `false` | Penalizes `min(scaling) / spatial_lr_scale` — drives each Gaussian flat along its normal. |
| `enforce_gaussian_flattening_weight` | float | `0.05` / `0.05` | Weight of the flattening loss. |
| `use_sdf` | bool | `false` / `false` | SDF/normal-field consistency term. **Setting `true` raises `NotImplementedError`** in `regularizer/normal_field.py` — the branch is a stub. |
| `enforce_sdf_and_normal_field_consistency` | bool | `false` / `false` | Only meaningful under `use_sdf` (currently unreachable). |
| `enforce_sdf_and_normal_field_consistency_weight` | float | `0.05` / `0.05` | Weight for the (unimplemented) SDF consistency loss. |
| `use_densification` | bool | `true` / `true` | Enables normal-error-driven densification (cloning pairs of Gaussians straddling the surface, as "spokes"). |
| `start_iter_densification` | int | `10_000` / `22_000` | Start of the densification window (compared against `iteration + 1`). |
| `end_iter_densification` | int | `14_000` / `26_000` | End of the densification window (inclusive, `iteration + 1`). |
| `densify_every_n_iterations` | int | `1000` / `1000` | Densification cadence within the window. |
| `densification_normalization_method` | str | `"count"` / `"count"` | How per-Gaussian normal errors are normalized in `compute_normal_error` (`densification/normal_error.py`): `"count"` (per-observation mean), `"area"` (weighted by projected area), or `"none"`. |
| `densification_normal_to_use` | str | `"median_depth"` / `"median_depth"` | Reference normal map for the error: `"rendered"` (rasterized normal output), `"median_depth"` (normals from median depth), or `"expected_depth"`. |
| `densification_normal_errors_quantile` | float | `0.15` / `0.05` | Fraction of highest-error Gaussians densified per pass (`torch.quantile(normal_errors, 1 − q)`). 0.15 densifies more aggressively than 0.05. |
| `maintain_constant_volume` | bool | `true` / `true` | When densifying, rescales each Gaussian along its local basis so the pair volume is halved (constant total volume) instead of blindly cloning. |
| `reset_normals_after_densification` | bool | `false` / `false` | If `true`, calls `reset_normal_features` after each densification pass. |
| `reset_normal_directions` | bool | `true` / `true` | Passed to `reset_normal_features`: re-aligns normal directions to the shortest Gaussian axis (only meaningful with 4 features). |
| `reset_normal_signs` | bool | `true` / `true` | Passed to `reset_normal_features`: zeroes the learnable sign feature. |
| `use_pruning` | bool | `false` / `false` | Enables non-maximal pruning (removes Gaussians never the front-most contributor in any training view, `prune_non_maximal_gaussians`). Disabled in both shipped configs. |
| `start_iter_pruning` | int | `10_500` / `22_500` | Start of the pruning window (`iteration + 1`). |
| `end_iter_pruning` | int | `10_505` / `22_505` | End of the pruning window. With the default cadence this is a single pruning event ("Single pruning" comment). |
| `prune_every_n_iterations` | int | `1000` / `1000` | Pruning cadence within the window. |

Note: the cap `--N_max_gaussians` (CLI-only, no YAML key) limits how many Gaussians each densification pass may add; when the cap is hit only the top-error candidates are kept.

## `multiview` group

Files: `default_regular_densification.yaml`, `fast.yaml`, `fast_late.yaml`. Implements multi-view photometric consistency (NCC) + geometric consistency (reprojection) losses following GGGS-style PatchMatch, with two code paths selected by `use_fast_multiview`:

- **Slow path** (`use_fast_multiview: false`): `PatchMatch` — renders the nearest neighbor camera with the main rasterizer and warps patches using `depth_ratio`-blended depth, `znear`, and the configured `interpolation`.
- **Fast path** (`use_fast_multiview: true`): `PatchMatchFast` (`regularization/multiview_gggs.py`) — reprojects the current median-depth map into the nearest camera and samples depth there with the `ours` median-depth sampling function. `depth_ratio`, `znear_relative` and `interpolation` are **not used** on this path; `multi_view_pixel_noise_th` drives both the validity mask and the geometric consistency loss.

Selection: `--multiview_config <name>` (default `fast`). The total loss is `multi_view_ncc_weight * ncc_loss + multi_view_geo_weight * geo_loss`, multiplied by the CLI `--multiview_factor`.

| Config | `use_fast_multiview` | `start_multiview` | Notes |
|---|---|---|---|
| `default_regular_densification` | `false` | `7000` | Slow, higher-fidelity PatchMatch; starts with the standard `--regularization_from_iter 7000` window. |
| `fast` (CLI default) | `true` | `7000` | Fast reprojection-based NCC/geo losses; prints a warning if the main rasterizer is not `ours` (the fast sampler is the `ours` median-depth sampler). |
| `fast_late` | `true` | `15_000` | Identical to `fast` but starts after standard densification ends (`densify_until_iter=15_000`). |

### Key reference

All 13 keys exist in **all three files**; only the two listed above differ.

| Key | Type | Value | Description |
|---|---|---|---|
| `use_fast_multiview` | bool | `false` / `true` / `true` | Selects `PatchMatchFast` vs `PatchMatch` in `initialize_multiview_regularization` (`regularizer/multiview.py`). |
| `start_multiview` | int | `7000` / `7000` / `15_000` | First iteration with multiview loss (`multiview_kick_on`). Shifted forward on checkpoint resume. |
| `multi_view_max_angle` | float | `30.0` | Max viewing-angle difference (degrees) between a camera and its candidate neighbors in `compute_nearest_cameras`. |
| `multi_view_min_dis_relative` | float | `0.002` | Minimum camera-pair distance as a fraction of scene radius (0.01/5.0 with the standard COLMAP radius). Closer cameras are excluded (insufficient baseline). |
| `multi_view_max_dis_relative` | float | `0.3` | Maximum camera-pair distance as a fraction of scene radius (1.5/5.0). Farther cameras are excluded (too occluded). |
| `multi_view_num` | int | `8` | Number of neighbor candidates retained per camera; one is sampled at random each iteration. |
| `multi_view_patch_size` | int | `3` | NCC patch half-size; total patch is `(2*3+1)² = 49` pixels. |
| `multi_view_pixel_noise_th` | float | `1.0` | Reprojection error threshold (pixels): pixels with larger warp error are masked out of both losses; the same quantity (weighted by `exp(-pixel_noise)`) forms the geometric consistency loss on the fast path. |
| `depth_ratio` | float | `0.6` | Blend of expected vs median depth used by the **slow** path only (`(1-r)*expected + r*median`). Unused by the fast path. |
| `znear_relative` | float | `0.02` | Near-plane clamp as a fraction of scene radius, passed as `znear` to the **slow** path only. Unused by the fast path. |
| `interpolation` | str | `"bilinear"` | Grid-sampling mode for depth interpolation, **slow** path only. |
| `multi_view_ncc_weight` | float | `0.6` | Weight of the normalized-cross-correlation photometric loss. |
| `multi_view_geo_weight` | float | `0.02` | Weight of the geometric consistency (reprojection error) loss. |

## `mesh_in_the_loop` group

File: `default_regular_densification.yaml` (the only one). Enables the mesh-in-the-loop (MILO) regularizer: every iteration past `start_iter`, the code builds a Delaunay tetrahedralization of the Gaussian pivots, derives an occupancy/SDF field from the learnable occupancy, extracts a mesh via marching tetrahedra, renders it, and penalizes mesh-vs-Gaussian depth/normal disagreement.

Selection: `--milo --milo_config <name>` (off by default). When on, the model is built with `learn_occupancy=True` and `n_pivots_per_gaussian` from this config.

### Key reference (25 keys)

| Key | Type | Value | Description |
|---|---|---|---|
| `start_iter` | int | `27_000` | First MILO iteration (`milo_kick_on`); all periodic schedules below are measured as `iteration − start_iter`. Shifted forward on resume. |
| `n_pivots_per_gaussian` | int | `2` | Number of pivots per Gaussian, set on the model at construction (`GaussianModel(n_pivots_per_gaussian=...)`); 2 = one pivot on each side of the surface along the normal. |
| `use_smallest_axis_as_normal` | bool | `false` | If `true`, `get_intersecting_pivots_from_normals` uses the smallest Gaussian axis as the normal instead of the learned normal feature. |
| `use_pivots_colors` | bool | `false` | Would attach per-pivot colors to the extracted mesh. **`true` raises `NotImplementedError`** ("Pivots colors are not implemented"). |
| `delaunay_method` | str | `"tetranerf"` | Triangulation backend passed to `return_delaunay_tets` (`scene/mesh.py`). |
| `reset_delaunay_every` | int | `500` | Re-triangulate every N iterations since `start_iter` (pivots move during training, so the tetrahedralization must be refreshed). |
| `end_reset_delaunay_iter` | int | `29_501` | Last iteration at which periodic re-triangulation happens; after this the last tetrahedralization is kept. |
| `start_reset_base_occupancy_iter` | int | `21_001` | Start of the periodic base-occupancy reset window. **Note:** this window ends before `start_iter` (27_000), so the periodic branch can never fire with the shipped values — occupancy is only (re)computed on the first MILO iteration or after a state reset. |
| `end_reset_base_occupancy_iter` | int | `25_001` | End of the periodic base-occupancy reset window (see note above). |
| `reset_base_occupancy_every` | int | `500` | Cadence of base-occupancy recomputation inside the window, via `compute_occupancy_sof` (visibility/occupancy splatting over all training cameras). |
| `reset_occupancy_ema_ratio` | float | `0.6` | After the first reset, new occupancy = `0.6 * current + 0.4 * recomputed`, clamped to [0.005, 0.995]. |
| `filter_large_edges` | bool | `false` | Drop mesh triangles whose edges exceed a scale threshold during `extract_mesh`. |
| `collapse_large_edges` | bool | `false` | Collapse (rather than only filter) large edges during `extract_mesh`. |
| `use_scalable_renderer` | bool | `false` | Wraps the mesh rasterizer in `ScalableMeshRenderer` instead of `MeshRenderer` (`scene/mesh.py`). |
| `rasterization_backend` | str | `"opengl"` | Mesh rasterizer backend: `"opengl"` (nvdiffrast OpenGL) or `"cuda"`. Consumed as `use_opengl = backend == "opengl"`. |
| `use_rgb_loss` | bool | `false` | RGB loss against the mesh render. **`true` raises `NotImplementedError`** (also in `utils/log_utils.py`). |
| `use_depth_loss` | bool | `true` | Enables the mesh-vs-Gaussian depth loss: `log(1 + |mesh_depth − gaussian_depth| / spatial_lr_scale)` over mesh-visible pixels, weighted by `depth_weight`. |
| `depth_weight` | float | `0.05` | Weight of the mesh depth loss. |
| `depth_ratio` | float | `1.0` | Blend of Gaussian depths used as the target: `1.0` = pure median depth; `(1−r)*expected + r*median` otherwise. |
| `use_normal_loss` | bool | `true` | Enables the mesh-vs-Gaussian normal loss: `1 − |dot(mesh_normal, gaussian_normal)|` over mesh-visible pixels. |
| `normal_weight` | float | `0.05` | Weight of the mesh normal loss. |
| `use_depth_normal` | bool | `true` | Derives the Gaussian reference normals from the `depth_ratio`-blended depth map (masked when `--mask_depth_normal`); if `false`, uses the directly rendered normal map. |
| `enforce_occupied_centers` | bool | `false` | Hinge loss pushing the occupancy of each Gaussian's center pivot up to `occupancy_isosurface` so Gaussian centers stay inside the mesh. |
| `occupied_centers_weight` | float | `0.005` | Weight of the occupied-centers hinge. |
| `occupancy_isosurface` | float | `0.5` | Target occupancy for the hinge above. |

## `depth_order` group

File: `default.yaml` (the only one). Monocular depth-order regularization using Depth-Anything-V2 priors: for each training view a relative-depth prior is precomputed, and the loss encourages the rendered depth to preserve the prior's *ordering* between randomly shifted pixel pairs (scale- and alignment-free). This regularizer is not used in the MILo paper; it is gated behind `--depth_order`.

### Key reference (12 keys)

| Key | Type | Value | Description |
|---|---|---|---|
| `depthanythingv2_checkpoint_dir` | str | `'submodules/Depth-Anything-V2/checkpoints/'` | Where `load_depthanything` looks for the checkpoint (relative to the repo root). |
| `depthanythingv2_encoder` | str | `'vitl'` | DAv2 encoder variant (`vitl` = ViT-Large). |
| `weight_initial_value` | float | `0.0` | Loss weight before the first schedule update — the loss is inactive (`lambda <= 0` short-circuits) until iteration 3000. |
| `weight_update_iters` | list[int] | `[3000, 7000, 15000, 20000, 25000]` | Iterations at which the weight steps to the matching entry of `weight_update_values`; the last value reached at or before the current iteration applies. |
| `weight_update_values` | list[float] | `[1, 0.1, 0.01, 0.001, 0.0001]` | Weight schedule values (1.0 after iter 3000, decaying ×0.1 each step). |
| `depth_ratio` | float | `0.6` | Depth fed to the loss: `0.6`-blended median/expected depth; `1.0` would use pure median depth. |
| `max_pixel_shift_ratio` | float | `0.05` | Random pixel shifts sampled in `[−5%, +5%]` of `max(H, W)` when pairing pixels for the order loss. |
| `normalize_loss` | bool | `true` | Normalizes the prior depth difference by its own magnitude (`prior_diff / |prior_diff|`), making the loss purely order-based. |
| `log_space` | bool | `true` | Compresses the loss via `log(1 + log_scale * loss)`. |
| `log_scale` | float | `20.0` | Scale inside the log compression above. |
| `reduction` | str | `'mean'` | Reduction of the pixel-wise loss (`mean`, `sum`, or `none`). |
| `deactivate_depth_order_after` | int | `15000` | Iteration after which the regularizer is turned off entirely (`-1` disables deactivation). With the shipped schedule this cuts the loss off right at the iter-15000 weight update. |

## `mesh` group (legacy MILo files)

**Status in this package: not loaded by any code.** The only YAML loader in GaussianWrapping is `load_regularization_configs()`, which reads the four groups above. `configs/mesh/*.yaml` are inherited from the standalone MILo repository (the package `.gitignore` still lists paths such as `milo/configs/mesh/ablation*`, `milo/configs/mesh/radegs*`, `milo/configs/mesh/foreground_only.yaml`), where a `--mesh_config <name>` style flag selected the file for the MILo training loop. In this repo the *algorithms* those configs parameterize are present — snapping (`regularizer/snapping_points.py`, `regularizer/snapping_edges.py`), learnable SDF/occupancy resets (`regularization/sdf/learnable.py`, `sdf/depth_fusion.py`, `sdf/integration.py`), pivot extraction (`extraction/pivots.py`), and the legacy logging function `log_training_progress` (`utils/log_utils.py`, which reads `mesh_config["enforce_occupied"]`, `use_occupancy_labels_loss`, `use_qem_loss`, `attract_pivots_along_mesh_edges`, …) — but `log_training_progress` is never called (`train.py` calls `log_normal_field_training_progress`) and no entry point loads these YAMLs. Treat the group as reference documentation of the MILo-v1 design space.

### File inventory (37 files)

| Family | Files | What the family toggles |
|---|---|---|
| Core / resolution | `verylowres`, `lowres`, `default`, `highres`, `veryhighres` | Delaunay point budget (`n_max_points_in_delaunay` 270k → 13.5M), sampling method, scalable rasterizer for the two largest budgets. |
| SH / pivot-shift variants of `default` | `default_sh`, `default_shifts`, `default_sh_shifts` | `default_sh`: harmonic SDF interpolation (SH degree 3). `default_shifts`: linear SDF + learnable pivot shift. `default_sh_shifts`: harmonics + learnable pivot shift. |
| Snapping + single geometry loss | `snapping_angle_defect`, `snapping_area_loss`, `snapping_laplace`, `snapping_laplace_cotcurv`, `snapping_laplace_cotcurv_very_high`, `snapping_normal_consistency` | Snap points (`snapping_method: softmax`, from iter 13001) + enable one geometry loss with weight `1.0` (laplace/very_high) or `0.005`–`0.005` defaults elsewhere. |
| Laplace weight sweep | `snapping_laplace_cotcurv_weight1` … `weight4` | Same as `snapping_laplace_cotcurv` with `laplacian_smoothing_loss_weight` 0.001 / 0.0005 / 0.0001 / 0.00001. |
| Normal-consistency weight sweep | `snapping_normal_consistency_weight1` … `weight4` | `normal_consistency_loss_weight` 0.01 / 0.05 / 0.1 / 1.0 (weight1 also lowers `laplacian_smoothing_loss_weight` to 0.05). |
| Snapping-points variants | `snapping_points`, `snapping_points_enforce_min`, `snapping_points_enforce_non_zero`, `snapping_points_enforce_min_enforce_non_zero` | Softmax snapping from iter 13000; `enforce_occupied_method: "min_sdf_value"` (in the `enforce_min` names) and `enforce_non_zero_sdf: true` (in the `enforce_non_zero` names), combinable. No geometry-loss block. |
| Snapping method variants | `snapping_edges` (+`_iterative`), `snapping_edges_mesh` (+`_iterative`), `snapping_edges_qem` (+`_iterative`) | `snapping_method` = `softmax` / `mesh_edges` / `qem`; `_iterative` variants set `iterative_snapping: true` and `snap_points_scale_factor: 1.1`. `mesh_edges` additionally enables the edge-attraction loss and flips `filter_large_edges: false`, and `snapping_edges_mesh` sets `collapse_large_edges: true`. |
| QEM loss weight sweep | `snapping_qem`, `snapping_qem_weight_1` … `weight_4` | `use_qem_loss: true` from 8001 to 13001 with `qem_loss_weight` 0.0005 / 0.00001 / 0.0001 / 0.001 / 1.0; geometry losses off. |

### Shared baseline

Every key below has the same value in **all 37 files** (unless noted):

| Key | Value | Notes |
|---|---|---|
| `start_iter` | `8_001` | Mesh regularization kick-on (legacy loop). |
| `mesh_update_interval` | `1` | Re-extract/update the mesh every iteration. |
| `stop_iter` | `18000` | Last iteration of mesh regularization. |
| `use_depth_loss` / `depth_weight` / `depth_ratio` / `mesh_depth_loss_type` | `true` / `0.05` / `1.0` / `"log"` | Mesh-render depth loss (log-scale). |
| `use_normal_loss` / `normal_weight` / `use_depth_normal` | `true` / `0.05` / `true` | Mesh-render normal loss. |
| `delaunay_reset_interval` | `500` | Re-triangulation cadence. |
| `sdf_reset_interval` / `sdf_default_isosurface` / `transform_sdf_to_linear_space` / `min_occupancy_value` | `500` / `0.5` / `false` / `0.0000000001` | SDF recomputation cadence and occupancy/SDF conversion floor. |
| `use_ema` / `alpha_ema` | `true` / `0.4` | EMA fusion during SDF reset (Integrate mode). |
| `trunc_margin` | `0.002` | TSDF truncation margin (TSDF mode). |
| `occupancy_mode` | `"occupancy_shift"` | Learnable occupancy parameterization (`"density_shift"` is the alternative). |
| `enforce_occupied` / `occupied_centers_weight` | `true` / `0.005` | Gaussian-center-inside-mesh hinge. |
| `use_occupancy_labels_loss` / `reset_occupancy_labels_every` / `occupancy_labels_loss_weight` | `true` / `200` / `0.005` | Supervises learnable occupancy against freshly fused labels. |
| `fix_set_of_learnable_sdfs` / `learnable_sdf_reset_stop_iter` / `learnable_sdf_reset_alpha_ema` | `true` / `13001` / `0.4` | Learnable-SDF handling. |
| `method_to_reset_sdf` / `n_binary_steps_to_reset_sdf` / `sdf_reset_linearization_n_steps` / `sdf_reset_linearization_enforce_std` / `depth_fusion_reset_tolerance` | `"depth_fusion"` / `0` / `20` / `0.5` / `0.1` | SDF reset via depth fusion (binary search disabled; 20 linearization steps; 0.1 tolerance). The alternative `"integration"` maps to `regularization/sdf/integration.py`. |
| `radius_culling` | `-1.0` | Foreground culling disabled (`-1.0` = no culling). |
| `snap_points_K` / `k_softmax` | `10` / `5000.0` | KNN size and softmax temperature for KNN-based snapping (absent only in the 4 resolution files). |
| `angle_defect_loss_weight` / `area_loss_weight` / `normal_consistency_loss_weight` (where the loss is off) | `0.005` | Inert defaults. |

### Keys that vary between files

| Key | Values across files |
|---|---|
| `n_max_points_in_delaunay` | `270_000` (verylowres), `540_000` (lowres), `5_400_000` (default* and all `snapping_*`), `9_000_000` (highres), `13_500_000` (veryhighres) |
| `delaunay_sampling_method` | `"surface"` everywhere except `highres` / `veryhighres` (`"surface+opacity"`) |
| `delaunay_snapping_reset_interval` | `500` in `default*` and all six `snapping_edges*`; `100` elsewhere |
| `delaunay_triangulation_method` | `"tetranerf"` where present; **absent** in the 4 resolution files |
| `filter_large_edges` / `collapse_large_edges` | `true` / `false` everywhere except `snapping_edges_mesh` (`false` / `true`) and `snapping_edges_mesh_iterative` (`false` / `false`) |
| `use_scalable_renderer` | `false` except `highres`, `veryhighres` (`true`) |
| `sdf_interpolation_method` / `sdf_interpolation_sh_degree` | `"linear"` / `0` (`default`, `default_shifts`), `"harmonics"` / `3` (`default_sh`, `default_sh_shifts`, `snapping_edges_mesh_iterative`); absent elsewhere |
| `learn_pivot_shift` | `false` (`default`, `default_sh`, `snapping_edges_mesh_iterative`), `true` (`default_shifts`, `default_sh_shifts`); absent elsewhere |
| `enforce_occupied_method` | `"center_of_gaussian"` except `snapping_points_enforce_min*` (`"min_sdf_value"`) |
| `enforce_occupied_per_cluster` | `false` in `default*` and `snapping_edges_mesh_iterative`; absent elsewhere |
| `learnable_sdf_reset_mode` | `"ema"` in `default` and the 4 resolution files; `"none"` everywhere else |
| `enforce_non_zero_sdf` / `non_zero_sdf_weight` | `false` / `0.005` except `snapping_points_enforce_non_zero` and `snapping_points_enforce_min_enforce_non_zero` (`true`) |
| `pivot_densification` block (`pivot_densification`, `start_iter_pivot_densification`, `end_iter_pivot_densification`, `densify_pivots_every_n_iterations`, `pivot_densification_depth_error_method`) | `false`, `15002`, `17002`, `500`, `"area"` in `default*` and `snapping_edges_mesh_iterative`; absent elsewhere |
| `snap_points` | `false` in `default*`; `true` in all 31 `snapping_*` files; **absent** in the 4 resolution files |
| `snapping_method` | `"softmax"` everywhere except `snapping_edges_mesh*` (`"mesh_edges"`) and `snapping_edges_qem*` (`"qem"`) |
| `start_iter_snapping` | `10000` (`default*`, `snapping_edges*`), `13001` (`snapping_angle_defect`, `snapping_area_loss`, laplace/normal-consistency families, `snapping_qem*`), `13000` (`snapping_points*`) |
| `iterative_snapping` / `snap_every_n_iterations` / `stop_iterative_snapping` | `false` / `1000` / `15001` in `default*` and `snapping_edges*`; `true` in the three `_iterative` files; absent elsewhere |
| `snap_points_scale_factor` | `1.5` (`default*`, `snapping_edges`, `snapping_edges_qem`), `1.1` (`snapping_edges_iterative`, `snapping_edges_mesh`, `snapping_edges_mesh_iterative`, `snapping_edges_qem_iterative`), `2.0` (angle_defect, area_loss, laplace family, normal-consistency family, points family, qem family) |
| `qem_mode` / `qem_threshold` / `qem_target_ratio` / `qem_normalization_factor` / `qem_average_w_face_area` | `'ratio'` / `0.00001` / `0.25` / `20.0` / `true` in `default*` and `snapping_edges*`; `qem_target_ratio` is `0.5` in `snapping_edges_qem_iterative`; absent in all other families |
| `attract_distance_mode` / `attract_distance_scale_threshold` / `attract_distance_ratio` | `"threshold"` / `1.5` / `0.5` in `default*` and `snapping_edges*` except `snapping_edges_mesh` (`2.` / `0.25`) |
| `attract_pivots_along_mesh_edges` (+ `_weight`, `attract_distance_normalization_factor`, `attract_weight_by_qem`, `attract_qem_normalization_factor`, `attract_qem_average_w_face_area`) | `true` in `snapping_edges_mesh*` (weight `0.0005`, normalization `0.001`, QEM weighting on, factor `20.0`, area-average `true`); `false` in `default*` and the other `snapping_edges*`; absent elsewhere |
| `use_qem_loss` / `qem_loss_weight` / `start_iter_qem_loss` / `end_iter_qem_loss` | `false` / `0.005` / `8001` / `10000` in `default*` and `snapping_edges*`; `true` with `8001`/`13001` and weights `0.0005` (`snapping_qem`), `0.00001` (w1), `0.0001` (w2), `0.001` (w3), `1.0` (w4) |
| `use_geometry_losses` / `start_iter_geometry_losses` | `false` / `8201` in `default*`, `snapping_edges*`, `snapping_qem*`; `true` / `13001` in angle_defect, area_loss, laplace family, normal-consistency family; absent in `snapping_points*` and the resolution files |
| `use_angle_defect_loss` | `true` in `snapping_angle_defect`, `snapping_edges`, `snapping_edges_iterative`, `snapping_edges_qem`, `snapping_edges_qem_iterative`; `false` elsewhere |
| `use_area_loss` | `true` only in `snapping_area_loss` |
| `use_laplacian_smoothing_loss` / `laplacian_smoothing_loss_weight` / `laplacian_smoothing_method` | `true` in `snapping_laplace` (`1.0`, `"cot"`), `snapping_laplace_cotcurv` (`0.005`, `"cotcurv"`), `snapping_laplace_cotcurv_very_high` (`1.0`), weight1–4 (`0.001` / `0.0005` / `0.0001` / `0.00001`, `"cotcurv"`); `false` with inert weights elsewhere (method key present in angle_defect, area_loss, laplace family, normal-consistency family: `"cot"`) |
| `use_normal_consistency_loss` / `normal_consistency_loss_weight` | `true` in `snapping_normal_consistency` (`0.005`), weight1 (`0.01`), weight2 (`0.05`), weight3 (`0.1`), weight4 (`1.0`); `false` elsewhere |

## `configs/fast`

`configs/fast` is **not YAML and not a directory** — it is a one-line plain-text file containing a Python `Namespace` repr:

```
Namespace(iterations=18_000, densify_until_iter=3_000, aggressive_clone_from_iter=500, aggressive_clone_interval=250, warn_until_iter=3_000, depth_reinit_iter=2_000, simp_iteration1=3_000, simp_iteration2=8_000)
```

No code reads it (the string `configs/fast` does not appear anywhere in the package; the `fast` name that *is* referenced is `configs/multiview/fast.yaml` via `--multiview_config fast`). The fields match a Mini-Splatting2-style aggressive-densification schedule (18k iterations, densification stopped at 3k, aggressive cloning every 250 iterations from iteration 500, simplification passes at 3k and 8k), so treat it as a developer memo of an alternative fast-training recipe rather than a loadable config. Do not point `--normal_field_config` or any other config flag at it — `yaml.safe_load` would fail on `Namespace(...)`.

## Which config do I use

| Goal | Choice |
|---|---|
| Reproduce the paper pipeline end-to-end | Defaults: `--normal_field_config default_regular_densification`, `--multiview_config fast` (or run `gaussian_wrapping/scripts/train_and_extract_gw_radegs.py` / `train_and_extract_gw_ours.py`, which forward these). |
| Best mesh quality, training-time mesh supervision | Add `--milo --milo_config default_regular_densification` (only shipped `mesh_in_the_loop` config). |
| Faster multiview regularization | `--multiview_config fast` (default). For photometric accuracy over speed: `default_regular_densification` (slow PatchMatch). |
| 30k-iteration runs with an early normal field | `--normal_field_config default` (densification 10k–14k, quantile 0.15). |
| Late multiview kick-in (e.g. after densification ends at 15k) | `--multiview_config fast_late`. |
| Lower memory / cap Gaussian count | Not a YAML key: use `--N_max_gaussians` (caps normal-field densification) and/or `-r 2` downsampling; see [CLI reference](cli-reference.md). |
| Monocular depth prior stabilization | `--depth_order --depth_order_config default` (needs the Depth-Anything-V2 checkpoint under `submodules/Depth-Anything-V2/checkpoints/`). |
| Mesh resolution knobs for extraction | Resolution variants (`verylowres` … `veryhighres`) in `configs/mesh/` document the point-budget settings used by the MILo lineage; the pipeline actually used here is `pivot_based_mesh_extraction.py` with its own CLI flags — see [Mesh extraction](mesh-extraction.md). |

### Gotchas

- `use_sdf: true` (normal_field) and `use_rgb_loss: true` / `use_pivots_colors: true` (mesh_in_the_loop) raise `NotImplementedError` — keep them `false`.
- The mesh-in-the-loop occupancy-reset window (`start/end_reset_base_occupancy_iter` = 21_001–25_001) lies entirely before `start_iter` = 27_000, so periodic occupancy resets never fire with the shipped values; only the initial reset at MILO start (and post-densification state resets) occur.
- Densification/pruning windows compare `iteration + 1`, so the first effective iteration is one below the configured start.
- The code-only `depth_order_weight_schedule()` helper (`regularization/depth/depth_order.py`, schedules `default`/`strong`/`weak`/`none`) is dead code — the config file's `weight_update_iters`/`weight_update_values` arrays are what the trainer actually consumes.
- Every active group's keys must exist; because loading is unchecked, copying a YAML and deleting a key will crash at runtime rather than fall back to a default.
