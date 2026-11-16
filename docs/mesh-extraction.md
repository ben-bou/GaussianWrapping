# Mesh Extraction

Gaussian Wrapping turns a trained 3D Gaussian model into a mesh with two pipelines. The **pivot-based pipeline** (`pivot_based_mesh_extraction.py`) is the default: it places pivot points around each Gaussian, evaluates an SDF-like occupancy field at them, runs a Delaunay triangulation, and applies marching tetrahedra. The **Primal Adaptive Meshing (PAM) pipeline** (`primal_adaptive_meshing_extraction.py`) is an alternative that samples candidate points from an existing mesh, refines them onto the Gaussian occupancy isosurface by gradient descent, and reconstructs the surface from those points.

> Source of truth: `README.md` (Training & Mesh Extraction, Primal Adaptive Meshing), `gaussian_wrapping/pivot_based_mesh_extraction.py`, `gaussian_wrapping/primal_adaptive_meshing_extraction.py`, `gaussian_wrapping/extraction/pivots.py`, `gaussian_wrapping/extraction/mesh.py`, `gaussian_wrapping/functional/pivots.py`, `gaussian_wrapping/utils/tetmesh.py`, `gaussian_wrapping/utils/tetmesh_gggs.py`, `gaussian_wrapping/utils/primal_adaptive_meshing_utils.py`, `gaussian_wrapping/scene/mesh.py`, `gaussian_wrapping/mesh_decimate.py`, `gaussian_wrapping/texture_mesh.py`, `gaussian_wrapping/blender/bounding_volume_addon.py`, `gaussian_wrapping/scripts/train_and_extract_gw_ours.py`, `gaussian_wrapping/scripts/train_and_extract_gw_radegs.py`.

## Contents

1. [Overview: which pipeline to use](#1-overview-which-pipeline-to-use)
2. [Inputs: the trained model directory](#2-inputs-the-trained-model-directory)
3. [Pivot-based pipeline](#3-pivot-based-pipeline)
4. [Primal Adaptive Meshing pipeline](#4-primal-adaptive-meshing-pipeline)
5. [Post-processing](#5-post-processing)
6. [Canonical command lines](#6-canonical-command-lines)
7. [Troubleshooting notes](#7-troubleshooting-notes)
8. [Related documentation](#8-related-documentation)

## 1. Overview: which pipeline to use

| | Pivot-based | Primal Adaptive Meshing (PAM) |
|---|---|---|
| Script | `gaussian_wrapping/pivot_based_mesh_extraction.py` | `gaussian_wrapping/primal_adaptive_meshing_extraction.py` |
| Input | Trained Gaussian model only | Trained Gaussian model **plus** an input mesh (typically the output of the pivot-based pipeline) |
| Surface definition | Occupancy/transmittance threshold evaluated at pivots spawned from Gaussians | Gaussian occupancy isosurface; candidate points are pulled onto it by gradient descent |
| Control over resolution | Indirect (number of Gaussians, pivot mode) | Direct (`--max_points` controls the number of mesh vertices) |
| Rasterizer | `ours` or `radegs` | `ours` only (the script raises `ValueError` for `radegs`) |
| Typical use | First mesh after training; used by the end-to-end scripts | Object extraction at arbitrary resolution, cleaner surfaces, custom region of interest |

The end-to-end scripts in `gaussian_wrapping/scripts/` chain training, pivot-based extraction, and texture refinement:

- `train_and_extract_gw_radegs.py` — RaDe-GS rasterizer ("Best looking meshes! A bit slower to train, script used for qualitative results on Mipnerf360").
- `train_and_extract_gw_ours.py` — median-depth rasterizer ("faster, better metrics").

Each script runs three steps in sequence: training, mesh extraction, and texture refinement. The refined mesh is saved alongside the extracted mesh with a `_texture_refined` suffix.

## 2. Inputs: the trained model directory

Both pipelines read a trained model directory produced by `gaussian_wrapping/train.py` (see [Training guide](training-guide.md)):

| Path | Role |
|---|---|
| `<model_dir>/point_cloud/iteration_<N>/point_cloud.ply` | Gaussian parameters loaded by `GaussianModel.load_ply` (default `--iteration 30000`) |
| `<model_dir>/cfg_args` | Training-time arguments, merged back into the CLI by `get_combined_args` (see [CLI reference](cli-reference.md)) |
| Dataset at `-s` | COLMAP-formatted dataset providing the training cameras used for SDF/occupancy evaluation |

All extraction scripts use `get_combined_args`, so `-m` is the only mandatory path argument if `cfg_args` is present; `-s` is normally inherited from the training run.

## 3. Pivot-based pipeline

### 3.1 How it works

`pivot_based_mesh_extraction.py` (adapted from Gaussian Opacity Fields) executes the following steps inside `marching_tetrahedra_with_binary_search`:

1. **Load the model.** `Scene(dataset, gaussians, load_iteration=args.iteration)` then `gaussians.load_ply(<model_dir>/point_cloud/iteration_<N>/point_cloud.ply)`. If the model was trained with learnable occupancy, `gaussians.set_occupancy_mode("occupancy_shift")` is applied.
2. **Build the SDF function.** Two modes (`--sdf_mode`):
   - `exact_computation` (default): a transmittance threshold of `0.5 + isosurface_value` is set on the splatting settings and `evaluate_vacancy_sof_fast` (from `gaussian_renderer/sof.py`) returns per-point vacancy; SDF = `0.5 - occupancy`.
   - `ours`: per-view minimum alpha integration via `integrate_ours` (from `gaussian_renderer/ours.py`); SDF = `0.5 - min alpha`.
3. **Isosurface value.** If `--isosurface_value` is left at its default `-9999.`, the value is computed automatically from rendered depth maps via `compute_isosurface_value_from_depth` (median over 1,000,000 depth points). Any explicit value (e.g. `0.0`, `0.2`) is used as-is.
4. **Pivot selection.** One of the following modes (mutually exclusive; see `extraction/pivots.py` and `functional/pivots.py`):
   - `--use_tetra_points`: 9 pivots per Gaussian from `gaussians.get_tetra_points` (tetrahedral corners).
   - `--use_scores`: score-based pivots via `get_pivots_by_scores` (`--score_threshold`, default `0.75`).
   - `--random_pivots`: random pivots sampled around each Gaussian via `sample_random_pivots` (`--n_pivots`, `--random_radius`).
   - `--use_searched_pivots`: pivots searched along the Gaussian normal via `get_searched_pivots` (`--search_iter`, `--search_step_size`, `--std_factor`).
   - Default: intersecting pivots from learned normals via `get_intersecting_pivots_from_normals` (`--n_pivots`, default `2`; `--std_factor`, default `3.0`; `--use_smallest_axis_as_normal` to use the smallest Gaussian axis instead of the learned normal). With `--use_intersecting_pivots`, only pivots whose SDF sign differs from the Gaussian center are kept.
5. **Valid mask** (optional, `--use_valid_mask`): pivots not visible in any camera frustum get their SDF switched to `0.5` (neutral), using the GGGS marching-tetrahedra variant (`utils/tetmesh_gggs.py`) instead of the kaolin-adapted one (`utils/tetmesh.py`).
6. **Delaunay triangulation.** `return_delaunay_tets(pivots, method="tetranerf")` in `scene/mesh.py` calls `cpp.triangulate` from the `tetranerf` extension (installed from `submodules/tetra_triangulation/` by `install.py`).
7. **Marching tetrahedra.** `extract_mesh` in `extraction/mesh.py` walks the tetrahedra, interpolates the zero crossing on each edge, and returns the surface vertices and faces.
8. **Binary search refinement.** With `--n_binary_steps` (default `8`), `refine_intersections_with_binary_search` (from `regularization/sdf/learnable.py`) re-evaluates the SDF along each edge to sharpen vertex placement.
9. **Vertex colors.** `evaluate_mesh_colors_all_vertices` (from `regularization/sdf/depth_fusion.py`) colors the mesh from the Gaussian renders.
10. **Export and post-processing.** The mesh is exported as PLY; with `--postprocess`, `post_process_mesh` keeps only the largest connected component and writes a second file.

### 3.2 Canonical commands

From the README:

```bash
# RaDeGS rasterizer (Best looking meshes! A bit slower to train, script used for qualitative results on Mipnerf360)
python gaussian_wrapping/scripts/train_and_extract_gw_radegs.py \
    -s <PATH_TO_COLMAP_DATASET> \
    -m <OUTPUT_DIR>

# Our median-depth rasterizer (faster, better metrics)
python gaussian_wrapping/scripts/train_and_extract_gw_ours.py \
    -s <PATH_TO_COLMAP_DATASET> \
    -m <OUTPUT_DIR> \
    -r 2 # Flag added to remain comparable to heavier alternatives. This flag was used to obtain the metrics on the paper
```

The equivalent standalone extraction invocation (the exact flags used by `train_and_extract_gw_ours.py`):

```bash
python gaussian_wrapping/pivot_based_mesh_extraction.py \
    -s <PATH_TO_COLMAP_DATASET> \
    -m <OUTPUT_DIR> \
    --rasterizer ours \
    --sdf_mode ours \
    --dtype int32 \
    --isosurface_value 0.0 \
    --n_binary_steps 10 \
    --iteration 30000 \
    --use_valid_mask \
    --postprocess \
    --filter_large_edges \
    --data_device cpu
```

The RaDe-GS variant uses `--sdf_mode exact_computation --rasterizer radegs --std_factor 3.33 --use_searched_pivots --search_iter 5 --search_step_size 1.0` instead. Full flag tables live in the [CLI reference](cli-reference.md).

### 3.3 Outputs

All files are written into the model directory (`-m`). The base name is `mesh_{sdf_mode}_{n_pivots}pivots{iso_suffix}.ply`, where `iso_suffix` is:

- `_transmittance_threshold_{t}` when `--sdf_mode exact_computation` and the threshold `0.5 + isosurface_value` is not `0.5`;
- `_iso_{v}` when `--sdf_mode ours` and `--isosurface_value` is not `0`;
- empty otherwise.

Additional suffixes are appended in this order: `_scores` (`--use_scores`), `_searched` (`--use_searched_pivots`), `_post` (`--postprocess`).

| Example file | Produced by |
|---|---|
| `mesh_ours_2pivots.ply` | `train_and_extract_gw_ours.py` (raw mesh) |
| `mesh_ours_2pivots_post.ply` | same, after `--postprocess` (this is the mesh used for TNT metrics) |
| `mesh_exact_computation_2pivots_searched.ply` | `train_and_extract_gw_radegs.py` (raw mesh) |
| `mesh_exact_computation_2pivots_searched_post.ply` | same, after `--postprocess` |
| `mesh_exact_computation_2pivots_transmittance_threshold_0.7_searched.ply` | `launch_nerf_synthetic.py` (`--isosurface_value 0.2` with RaDe-GS) |
| `mesh_ours_2pivots_post_texture_refined_999.ply` | `texture_mesh.py` (default `--n_iter 1000`) |

The `_post` meshes are the ones evaluated with the benchmark metrics (see [Evaluation](evaluation.md)). The README notes that post-processing can sometimes remove objects (e.g. the chandelier in the Meeting Room scene), so for textured meshes the non-post-processed mesh is recommended.

## 4. Primal Adaptive Meshing pipeline

### 4.1 How it works

`primal_adaptive_meshing_extraction.py` (PoNQ-style extraction) executes the following steps:

1. **Load the model.** `GaussianModel(..., learn_occupancy=True, n_gaussian_features=4)` at `--iteration` (default `30000`).
2. **Build the global occupancy function.** Only `--rasterizer ours` is supported (`radegs` raises `ValueError: Always use the Ours rasterizer to apply Primal Adaptive Meshing.`). For each query point, `compute_global_occupancy` takes the minimum `alpha_integrated` over all training views; points inside a frustum but behind a `gt_mask` alpha below `--mask_bg_threshold` (default `0.01`) are forced to occupancy `0` (background voting), and points outside every frustum are set to `0`.
3. **Generate candidate points from the input mesh** (`load_mesh` + `get_candidate_points`):
   - Crop the input mesh with a bounding volume (see 4.2).
   - Sample `--max_points * --oversampling_factor` points from the cropped mesh (`--mesh_sampling_method`: `proportional_to_camera` (default) or `surface_even`).
   - Refine the points onto the occupancy isosurface with `--n_steps` (default `10`) gradient-descent steps along the Gaussian vector field (`GaussianVectorField` in `utils/primal_adaptive_meshing_utils.py`, `--n_neighbors_vector_field` neighbors, default `32`), targeting `--iso_surface_value` (default `0.0`).
   - Filter points by vacancy: keep points with `|occupancy - iso_surface_value| <= --vacancy_threshold` (default `0.1`). The loop resamples until `--max_points` valid points are collected unless `--no_force_use_all_points` is set.
4. **Reconstruct the surface** (`mesh_from_points_and_occupancy`):
   - Build a Delaunay triangulation of the filtered points with `MeshFromDelaunay` (in `utils/primal_adaptive_meshing_utils.py`).
   - Sample `--p_per_tet` (default `10`) points per tetrahedron and classify each tet as inside/outside by comparing its mean occupancy to `--iso_surface_value`.
   - Extract the surface between inside and outside tets.
5. **Export.** The mesh is written to `--output_mesh` (must end in `.ply`); with `--post_process` it is first passed through `post_process_mesh` (largest connected component).

### 4.2 Bounding volume requirements

The input mesh is cropped before sampling, so a bounding volume is required. `--bounding_box_method` selects the source:

| Method | Volume used | Extra requirements |
|---|---|---|
| `scene` (default) | Axis-aligned box of half-extent `scene_radius * --bounding_box_scaling` centered at the origin, where `scene_radius` comes from `get_cameras_spatial_extent` | None |
| `ground_truth` | Open3D `SelectionPolygonVolume` read from `<scene_dir>/<Scene>.json`, plus the trajectory transform from `<Scene>_COLMAP_SfM.log` (TNT ground-truth crop) | Dataset directory must contain the TNT GT files; `--bounding_box_scaling` must be `1.0` |
| `blender` | Convex hull built from the vertices in a JSON exported by the GW Blender add-on | `--bounding_box_file` is required; the JSON must have `class_name == "GaussianWrappingBoundingVolume"` |

For `blender`, the JSON is loaded by `load_blender_crop_volume`, which builds a `scipy.spatial.ConvexHull` from the `vertices` field and applies an identity transform ("Blender world == model world"), so the volume must be exported in the model's world coordinates.

### 4.3 Using the provided bounding volumes

The README provides ready-to-use bounding volume files for several scenes in `assets/bounding_volumes_examples/` (paths relative to the repository root). These can be used directly with PAM — no Blender setup required:

| File | Scene |
|---|---|
| `Barn_bounding_volume.json` | Tanks and Temples — Barn |
| `bicycle_bounding_volume.json` | MipNeRF 360 — Bicycle |
| `bonsai_bounding_volume.json` | MipNeRF 360 — Bonsai |
| `garden_bounding_volume.json` | MipNeRF 360 — Garden |
| `kitchen_bounding_volume.json` | MipNeRF 360 — Kitchen |

```bash
python gaussian_wrapping/primal_adaptive_meshing_extraction.py \
    -s <PATH_TO_COLMAP_DATASET> \
    -m <OUTPUT_DIR> \
    --input_mesh <PATH_TO_INPUT_MESH> \
    --output_mesh <PATH_TO_OUTPUT_MESH.ply> \
    --bounding_box_method blender \
    --bounding_box_file assets/bounding_volumes_examples/Barn_bounding_volume.json
```

### 4.4 Blender Bounding Volume add-on

The add-on at `gaussian_wrapping/blender/bounding_volume_addon.py` ("GW Bounding Volume Exporter") lets you sculpt an arbitrary region of interest and export it as a convex-hull bounding volume.

#### Installation

1. Open Blender.
2. Go to **Edit → Preferences → Add-ons → Install…**
3. Select `gaussian_wrapping/blender/bounding_volume_addon.py`.
4. Enable the **GW Bounding Volume Exporter** add-on in the list.

The add-on requires Blender 3.0+ (per its `bl_info`) and appears as the **GW Bounds** tab in the 3D Viewport's N-panel.

#### Usage

1. **Import your input mesh** into Blender as a visual reference (e.g. *File → Import → Stanford PLY* and select the mesh you plan to pass to `--input_mesh`).
2. **Create or sculpt a mesh** that encloses the region of interest you want to extract.
3. In the **GW Bounds** panel, select that mesh as the *Bounding Mesh* and set the *Export Path* (default `//bounding_volume.json`, i.e. next to the `.blend` file).
4. Click **Export Bounding Volume**.

What the add-on writes: a JSON file with keys `class_name` (`"GaussianWrappingBoundingVolume"`), `version`, `blender_object_name`, and `vertices` (world-space vertices of the object). If `scipy` is available inside Blender's Python, the vertices are reduced to the convex-hull vertices before export; otherwise the raw mesh vertices are exported and the extraction script computes the hull itself.

Then run the extraction with `--bounding_box_method blender --bounding_box_file <exported .json>` as shown in section 4.3.

### 4.5 Canonical command

From the README:

```bash
python gaussian_wrapping/primal_adaptive_meshing_extraction.py \
    -s <PATH_TO_COLMAP_DATASET> \
    -m <OUTPUT_DIR> \
    --input_mesh <PATH_TO_INPUT_MESH> \
    --output_mesh <PATH_TO_OUTPUT_MESH.ply> \
    --max_points <NUMBER_OF_MESH_VERTICES>
```

### 4.6 Outputs

| File | Condition |
|---|---|
| `<--output_mesh>` (must end in `.ply`) | Always |
| `<--output_mesh>` with `.ply` replaced by `_candidate_points.ply` | `--save_candidate_points` |
| `<model_dir>/histogram_occupancy_values_{i}.png` (one per refinement step) | `--plot_vacancy_histogram` |

## 5. Post-processing

### 5.1 Mesh decimation (`mesh_decimate.py`)

`gaussian_wrapping/mesh_decimate.py` is a Blender script (it imports `bpy`, so it must run inside Blender, not plain Python). It applies Blender's Decimate modifier and exports a PLY:

```bash
blender --background --python gaussian_wrapping/mesh_decimate.py -- \
    --in <PATH_TO_OUTPUT_MESH.ply> \
    --out <PATH_TO_DECIMATED_MESH.ply> \
    --ratio 0.3
```

- `--out` defaults to `<in>` with `.ply` replaced by `_decimated_with_blender.ply`.
- `--ratio` is the decimation ratio (default `0.3`).
- The end-to-end scripts expose the same step via `--apply_decimation --decimate_ratio <ratio>`; note that the decimated mesh is an additional output and is not the mesh used for metric computation.

### 5.2 Texture refinement (`texture_mesh.py`)

`gaussian_wrapping/texture_mesh.py` optimizes the vertex colors of an existing mesh against the Gaussian renders of the training cameras (L1 + D-SSIM loss, `--lambda_dssim`), using the mesh rasterizer from `scene/mesh.py`:

```bash
python gaussian_wrapping/texture_mesh.py \
    -s <PATH_TO_COLMAP_DATASET> \
    -m <OUTPUT_DIR> \
    --rasterizer ours \
    --mesh <PATH_TO_MESH.ply>
```

Key flags: `--n_iter` (default `1000`), `--lr` (default `0.0025`), `--lambda_dssim` (default `0.2`), `--sh_degree_for_texturing` (default `0`), `--use_scalable_renderer` (default on). The output is written into the model directory as `<mesh_name>_texture_refined_<last_iter>.<ext>` — for example `mesh_ours_2pivots_post_texture_refined_999.ply`.

## 6. Canonical command lines

| Task | Command |
|---|---|
| Train + extract + texture (RaDe-GS) | `python gaussian_wrapping/scripts/train_and_extract_gw_radegs.py -s <PATH_TO_COLMAP_DATASET> -m <OUTPUT_DIR>` |
| Train + extract + texture (ours) | `python gaussian_wrapping/scripts/train_and_extract_gw_ours.py -s <PATH_TO_COLMAP_DATASET> -m <OUTPUT_DIR> -r 2` |
| Standalone pivot-based extraction | `python gaussian_wrapping/pivot_based_mesh_extraction.py -s <PATH_TO_COLMAP_DATASET> -m <OUTPUT_DIR> --rasterizer ours --sdf_mode ours --dtype int32 --isosurface_value 0.0 --n_binary_steps 10 --iteration 30000 --use_valid_mask --postprocess --filter_large_edges --data_device cpu` |
| Primal Adaptive Meshing | `python gaussian_wrapping/primal_adaptive_meshing_extraction.py -s <PATH_TO_COLMAP_DATASET> -m <OUTPUT_DIR> --input_mesh <PATH_TO_INPUT_MESH> --output_mesh <PATH_TO_OUTPUT_MESH.ply> --max_points <NUMBER_OF_MESH_VERTICES>` |
| PAM with a provided bounding volume | `python gaussian_wrapping/primal_adaptive_meshing_extraction.py -s <PATH_TO_COLMAP_DATASET> -m <OUTPUT_DIR> --input_mesh <PATH_TO_INPUT_MESH> --output_mesh <PATH_TO_OUTPUT_MESH.ply> --bounding_box_method blender --bounding_box_file assets/bounding_volumes_examples/Barn_bounding_volume.json` |
| Decimate | `blender --background --python gaussian_wrapping/mesh_decimate.py -- --in <PATH_TO_OUTPUT_MESH.ply> --out <PATH_TO_DECIMATED_MESH.ply> --ratio 0.3` |
| Texture refinement | `python gaussian_wrapping/texture_mesh.py -s <PATH_TO_COLMAP_DATASET> -m <OUTPUT_DIR> --rasterizer ours --mesh <PATH_TO_MESH.ply>` |

## 7. Troubleshooting notes

Failure modes that are visible in the code:

- **Missing bounding volume for PAM.** `--bounding_box_method blender` without `--bounding_box_file` raises `AssertionError: --bounding_box_file is required when --bounding_box_method=blender`. An unknown method raises `ValueError: Bounding Box Method: [<method>] does not exist`.
- **Wrong bounding volume JSON.** `load_blender_crop_volume` raises `ValueError: Expected class_name 'GaussianWrappingBoundingVolume'` if the JSON was not exported by the GW add-on. The volume must be exported in the model's world coordinates (the script applies an identity transform).
- **Empty crop.** If the bounding volume does not intersect the input mesh, `load_mesh` prints `[WARNING] Cropped mesh is empty!` and returns zero points; the run then produces an empty mesh. Check that the volume was exported in world space and actually encloses the object.
- **Missing `tetra_triangulation` / `tetranerf` extension.** `scene/mesh.py` imports `tetranerf.utils.extension.cpp` in a `try/except`; if the import fails it prints `[WARNING] Could not import 'tetranerf.utils.extension.cpp'. Mesh regularization requires this.` and `return_delaunay_tets(..., method="tetranerf")` will fail. Build the extension from `submodules/tetra_triangulation/` (see `install.py`: cmake, gmp, `pip install -e .`), or use a `scipy`-based fallback where available.
- **PAM with the wrong rasterizer.** `primal_adaptive_meshing_extraction.py` raises `ValueError: Always use the Ours rasterizer to apply Primal Adaptive Meshing.` for `--rasterizer radegs`.
- **Missing input mesh.** `load_mesh` raises `FileNotFoundError: Input mesh not found: <path>` if `--input_mesh` does not exist.
- **Output mesh extension.** `export_mesh` asserts the output name ends in `.ply` (`AssertionError: Mesh must be saved as PLY file`).
- **`--ground_truth` bounding box scaling.** `load_mesh` asserts `--bounding_box_scaling == 1.0` when `--bounding_box_method ground_truth` (`AssertionError: Can only adjust bounding box for scene BB.`).
- **Post-processing removes objects.** `post_process_mesh` keeps only the largest connected component (clusters smaller than 50 triangles are always dropped). The README warns this can remove objects such as the chandelier in the TNT Meeting Room scene; use the non-post-processed mesh for textured output.
- **CUDA device.** Both extraction scripts hard-code `torch.cuda.set_device(torch.device("cuda:0"))`; set `CUDA_VISIBLE_DEVICES` to select the GPU (the benchmark scripts do this via `--gpu_device`).
- **`--isosurface_value` semantics differ between pipelines.** In the pivot-based pipeline the value shifts the transmittance threshold (`0.5 + value`); in PAM it is compared against the occupancy directly (`--iso_surface_value`). The README suggests `0.2` if the mesh is missing fine details.

## 8. Related documentation

- [CLI reference](cli-reference.md) — full flag tables for every script.
- [Training guide](training-guide.md) — how the model directory is produced.
- [Regularization configs](regularization-configs.md) — depth-order and other regularization configs.
- [Evaluation](evaluation.md) — benchmarking the extracted meshes.
- [Artifacts reference](artifacts.md) — exact filenames and contents of every artifact.
- [API reference](api-reference.md) — module-level documentation.
- [Architecture](architecture.md) — how the components fit together.
