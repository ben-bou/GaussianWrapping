# Python API Reference

> Source of truth: `gaussian_wrapping/api.py`, `gaussian_wrapping/__init__.py`,
> `gaussian_wrapping/train.py` (`resolve_rasterizer`, `load_regularization_configs`).

The package exposes two levels of API:

1. **High-level training API** (`gaussian_wrapping.api`) — a typed, reproducible
   description of a training run (`TrainingConfig`) and a single entry function
   (`run_training`). This is the interface used by the downstream
   `GW_experiments` repository.
2. **Object-level API** (`GaussianWrappingScene`, `Scene`, `GaussianModel`) —
   for loading a dataset, rendering, and saving/loading models programmatically.

## Package exports (`gaussian_wrapping/__init__.py`)

```python
from gaussian_wrapping.scene import Scene, GaussianModel
from gaussian_wrapping.arguments import (
    ModelParams, PipelineParams, OptimizationParams, get_combined_args, read_config,
)
from gaussian_wrapping.gaussian_renderer import (
    render_old, render_imp, render_simp, render_depth, render_full, render_isosurface,
)
from gaussian_wrapping import functional, extraction, api
from gaussian_wrapping.api import TrainingConfig, TrainingResult, run_training

__version__ = "1.0.0"
```

---

## `TrainingConfig`

```python
@dataclass
class TrainingConfig:  # 78 fields
    ...
```

A complete, reproducible description of a training run. Every value the
training loop reads from its `args` namespace is represented here, so a run can
be fully reconstructed from this object. Defaults match the command-line entry
point in `gaussian_wrapping.train`.

### Dataset fields (11)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `source_path` | `str` | `""` | Path to the dataset root (COLMAP layout: `images/`, `sparse/`). **Required.** |
| `model_path` | `str` | `""` | Output directory for the trained model. **Required.** |
| `images` | `str` | `"images"` | Sub-directory containing the training images (e.g. `"delights"` for relit frames). |
| `resolution` | `int` | `-1` | Input image resolution. `-1` = native, `0/1/2` = downsample by 1/2, 1/4, 1/8. |
| `eval` | `bool` | `False` | Hold out every `llff`-th camera as a test split instead of training on all images. |
| `white_background` | `bool` | `False` | Render/composite against a white background. |
| `data_device` | `str` | `"cuda"` | Device the source images are kept on. `"cpu"` saves VRAM for large image sets. |
| `use_unbounded_opacity` | `bool` | `False` | Enable the unbounded-opacity variant of the Gaussian model. |
| `llff` | `int` | `8` | Test-split stride used when `eval=True` (every `llff`-th camera is held out). |
| `kernel_size` | `float` | `0.0` | Extra anti-aliasing kernel size passed to the rasterizer (Mip-Splatting-style). |
| `save_init_ply` | `bool` | `False` | Additionally persist the initial point cloud as `<model_path>/points3D.ply`. The in-memory point cloud is always written as `input.ply`. |

### Optimization fields (23)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `iterations` | `int` | `30_000` | Total optimization steps. **Must be > 0.** The final iteration is always saved and tested. |
| `sh_degree` | `int` | `3` | Maximum spherical-harmonics degree (0–3). |
| `position_lr_init` | `float` | `0.00016` | Initial learning rate for Gaussian positions. |
| `position_lr_final` | `float` | `0.0000016` | Final position learning rate (exponential decay). |
| `position_lr_delay_mult` | `float` | `0.01` | Delay multiplier of the position LR scheduler. |
| `position_lr_max_steps` | `int` | `30_000` | Steps over which the position LR decays. |
| `feature_dc_lr` | `float` | `0.0025` | Learning rate for the DC spherical-harmonics feature. |
| `feature_rest_lr` | `float` | `0.0025 / 20.0` | Learning rate for the remaining SH features. |
| `opacity_lr` | `float` | `0.05` | Learning rate for opacities. |
| `scaling_lr` | `float` | `0.005` | Learning rate for Gaussian scales. |
| `rotation_lr` | `float` | `0.001` | Learning rate for Gaussian rotations. |
| `percent_dense` | `float` | `0.01` | Extent threshold (fraction of `cameras_extent`) used to decide clone vs. split during densification. |
| `lambda_dssim` | `float` | `0.2` | Weight of the D-SSIM term: `loss = (1 - λ)·L1 + λ·(1 - SSIM)`. |
| `densification_interval` | `int` | `100` | Iterations between densify/prune passes. |
| `opacity_reset_interval` | `int` | `3000` | Iterations between opacity resets. |
| `densify_from_iter` | `int` | `500` | First iteration at which densification runs. |
| `densify_until_iter` | `int` | `15_000` | Densification stops after this iteration. |
| `densify_grad_threshold` | `float` | `0.0002` | Position-gradient threshold for densification (on absolute grads — see `use_abs_grad_for_densification`). |
| `random_background` | `bool` | `False` | Randomize the background color each iteration. |
| `appearance_embeddings_lr` | `float` | `0.001` | Learning rate for per-image appearance embeddings (`decoupled_appearance`). |
| `appearance_network_lr` | `float` | `0.001` | Learning rate for the appearance MLP (`decoupled_appearance`). |
| `gaussian_features_lr` | `float` | `0.05 / 2.0` | Learning rate for the learnable Gaussian features used by the normal field. |
| `pgsr_appearance_lr` | `float` | `0.001` | Learning rate for the per-camera exposure-compensation parameters (PGSR-style). |

### Pipeline fields (3)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `convert_SHs_python` | `bool` | `False` | Compute SH features in Python instead of CUDA (debug/slow). |
| `compute_cov3D_python` | `bool` | `False` | Compute 3D covariance in Python instead of CUDA (debug/slow). |
| `debug` | `bool` | `False` | Pass the debug flag through to the rasterizer. |

### Runtime fields (8)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `rasterizer` | `str` | `"radegs"` | Primary rasterizer: `"radegs"` or `"ours"`. **Validated.** `"ours"` force-enables `mask_depth_normal`. |
| `multiview_rasterizer` | `Optional[str]` | `None` | Optional separate rasterizer for the multiview regularizer; `None` or `"ours"`. **Validated.** |
| `enable_gui` | `bool` | `False` | Start the interactive network GUI server (the CLI always starts it; the API defaults to off). |
| `ip` | `str` | `"127.0.0.1"` | GUI server bind address. |
| `port` | `int` | `-1` | GUI server port; `-1` picks a random port in `[5000, 9000)`. **Validated** (`-1` or `1..65535`). |
| `debug_from` | `int` | `-1` | Iteration from which to enable rasterizer debug mode (`-1` = never). |
| `detect_anomaly` | `bool` | `False` | Enable `torch.autograd.set_detect_anomaly` (very slow; for debugging NaNs). |
| `quiet` | `bool` | `False` | Suppress console output (`safe_state(quiet)`). |

### Regularization fields (25)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `use_normal_field` | `bool` | `True` | Enable the learnable normal-field regularizer (the package's core contribution). Disable with `False`. |
| `normal_field_config` | `str` | `"default_regular_densification"` | Name of the YAML under `configs/normal_field/` (see [regularization-configs.md](regularization-configs.md)). |
| `multiview` | `bool` | `True` | Enable multiview (NCC + geometric consistency) regularization. |
| `multiview_config` | `str` | `"fast"` | Name of the YAML under `configs/multiview/`. |
| `multiview_factor` | `float` | `1.0` | Global weight multiplied into the multiview loss. |
| `milo` | `bool` | `False` | Enable mesh-in-the-loop (MILo) regularization. |
| `milo_config` | `str` | `"default_regular_densification"` | Name of the YAML under `configs/mesh_in_the_loop/`. |
| `depth_order` | `bool` | `False` | Enable the learned depth-order prior (requires Depth-Anything-V2; not used in the MILo paper). |
| `depth_order_config` | `str` | `"default"` | Name of the YAML under `configs/depth_order/`. |
| `regularization_from_iter` | `int` | `7_000` | Iteration at which depth-normal consistency regularization kicks in. |
| `lambda_depth_normal` | `float` | `0.05` | Weight of the depth-normal consistency loss. |
| `mask_depth_normal` | `bool` | `False` | Use the median-depth validity mask when blending expected/median depth for the depth-normal loss. Forced `True` when `rasterizer == "ours"`. |
| `dense_gaussians` | `bool` | `False` | Use the "dense Gaussians" management mode. |
| `detach_gaussian_rendering` | `bool` | `False` | Detach the rendered image/depth/normals before the normal-field loss (stops gradients flowing through the rendering). |
| `N_max_gaussians` | `Optional[int]` | `None` | Cap on the Gaussian count during normal-field densification; when the next densification would exceed it, only the highest-error Gaussians are added up to the cap. `None` = unlimited. |
| `warn_until_iter` | `int` | `3000` | Iteration at which an extra 3D mip-filter update is triggered after densification ends. |
| `non_maximal_pruning_iterations` | `List[int]` | `[]` | Explicit iterations at which to prune non-maximal Gaussians. |
| `use_max_size_threshold` | `bool` | `False` | Apply the size threshold (20) when pruning Gaussians after `opacity_reset_interval`. |
| `use_scale_loss` | `bool` | `False` | PGSR-style minimum-scale regularization. |
| `scale_loss_weight` | `float` | `100.0` | Weight of the minimum-scale loss. |
| `weight_by_img_grad` | `bool` | `False` | Weight the depth-normal loss by image gradients (PGSR-style). |
| `disable_mip_filter` | `bool` | `False` | Disable the 3D mip filter (Mip-Splatting-style). |
| `update_mip_filter_every` | `int` | `100` | Period (iterations) of 3D mip-filter updates after densification ends. |
| `decoupled_appearance` | `bool` | `False` | Enable the GOF-style appearance network (per-image embeddings + MLP). |
| `exposure_compensation` | `bool` | `True` | Enable PGSR-style per-camera exposure compensation. |

### Iterations / checkpoints (4)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `save_iterations` | `List[int]` | `[]` | Extra iterations at which to save `point_cloud/iteration_<N>/point_cloud.ply`. The final iteration is always appended. |
| `test_iterations` | `List[int]` | `[]` | Extra iterations at which to run test-set evaluation. The final iteration is always appended (unless `-1` is present). |
| `checkpoint_iterations` | `List[int]` | `[]` | Iterations at which to write full optimizer checkpoints `chkpnt<N>.pth`. |
| `start_checkpoint` | `Optional[str]` | `None` | Path to a `chkpnt<N>.pth` to resume from. Regularizer start iterations are bumped past the checkpoint iteration. |

### Logging fields (4)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `log_interval` | `Optional[int]` | `None` | Interval (iterations) for verbose per-iteration logging; `None` uses the default progress-bar cadence. |
| `wandb_project` | `Optional[str]` | `None` | Weights & Biases project. WandB is only initialized when project **and** entity are set. |
| `wandb_entity` | `Optional[str]` | `None` | Weights & Biases entity/team. |
| `wandb_name` | `Optional[str]` | `None` | Weights & Biases run name. |

### Validation

`TrainingConfig.validate()` (called automatically by `run_training`) enforces:

| Rule |
|------|
| `source_path` must be non-empty |
| `model_path` must be non-empty |
| `rasterizer` ∈ `{"radegs", "ours"}` |
| `multiview_rasterizer` ∈ `{None, "ours"}` |
| `resolution` ∈ `{-1, 0, 1, 2}` |
| `sh_degree` ∈ `{0, 1, 2, 3}` |
| `iterations` > 0 |
| `port` = `-1` or `1..65535` |

Violations raise `ValueError` with the offending field and value.

---

## `run_training`

```python
def run_training(config: TrainingConfig) -> TrainingResult
```

Programmatic equivalent of `python -m gaussian_wrapping.train` with the
corresponding flags. Step by step it:

1. Calls `config.validate()`.
2. Lazily imports the training loop (so importing `gaussian_wrapping.api` does
   not pull in optional dependencies such as `fused_ssim`/`wandb`).
3. Converts the config into the three parameter namespaces the loop expects
   (`ModelParams`-like, `OptimizationParams`-like, `PipelineParams`-like) and a
   flat `args` namespace with two forced values:
   - `args.use_abs_grad_for_densification = True` (always),
   - `args.no_normal_field = not config.use_normal_field`.
4. Appends `config.iterations` to `save_iterations` and `test_iterations`
   (mirroring the CLI).
5. Resolves the rasterizer callables via `resolve_rasterizer()`; when
   `rasterizer == "ours"` it force-sets `args.mask_depth_normal = True`.
6. Loads the four regularization YAMLs via `load_regularization_configs()`
   (disabled regularizers yield `None`).
7. Performs global setup: `safe_state(config.quiet)` (RNG seeding + output
   redirection), optional GUI server (`network_gui.init(ip, port)` with a random
   port when `port == -1`), `torch.autograd.set_detect_anomaly(config.detect_anomaly)`.
8. Runs the training loop, timing it with `torch.cuda.synchronize()` brackets.
9. Writes `<model_path>/time.txt` containing the wall-clock training duration.
10. Returns a `TrainingResult`.

## `TrainingResult`

```python
@dataclass
class TrainingResult:
    scene: Any               # the trained Scene
    gaussians: Any           # the trained GaussianModel
    model_path: str          # echo of config.model_path
    iterations: int          # echo of config.iterations
    duration_seconds: float  # wall-clock training time
```

---

## `GaussianWrappingScene`

High-level wrapper around `Scene` + `GaussianModel` for programmatic use.

```python
GaussianWrappingScene(
    source_path: str,
    model_path: Optional[str] = None,   # default: <source_path>/output
    sh_degree: int = 3,
    images: str = "images",
    white_background: bool = False,
    resolution: int = -1,
    eval: bool = False,
    device: str = "cuda",               # becomes data_device
    save_init_ply: bool = False,
)
```

On construction it builds a `GaussianModel(sh_degree)`, a `Scene` (loading
cameras and initializing Gaussians from the initial point cloud), and exposes
`train_cameras` / `test_cameras`. Note it does **not** construct the
normal-field features, exposure compensation, or any regularizer state — it is
a plain 3DGS-style scene wrapper, not a training entry point (use
`run_training` for that).

### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `render` | `(camera, pipeline=None, background=None, scaling_modifier=1.0, override_color=None, rasterizer="ours") -> Dict[str, Tensor]` | Render one view. `rasterizer` selects `render_ours` (default) or `render_radegs`; anything else raises `ValueError`. Returns the rasterizer's dict (keys include `render`, `depth`, `alpha`, …). |
| `render_all` | `(split="train", pipeline=None, background=None) -> List[Dict[str, Tensor]]` | Render every camera of the chosen split. |
| `save` | `(iteration=0)` | Save the current model to `<model_path>/point_cloud/iteration_<iteration>/`. |
| `load` | `(iteration)` | Rebuild the `Scene` and load `point_cloud/iteration_<iteration>/point_cloud.ply`. |

### `create_scene`

```python
def create_scene(source_path: str, model_path: Optional[str] = None, **kwargs) -> GaussianWrappingScene
```

Convenience factory that forwards everything to `GaussianWrappingScene`.

---

## Shared helpers (in `gaussian_wrapping.train`)

These are reused by both the CLI and `run_training`:

| Function | Signature | Description |
|----------|-----------|-------------|
| `resolve_rasterizer` | `(rasterizer: str, multiview_rasterizer=None) -> (render_fn, integrate_fn, multiview_render_fn)` | Maps `"radegs"` → `gaussian_wrapping.gaussian_renderer.radegs.render_radegs/integrate_radegs`, `"ours"` → `gaussian_wrapping.gaussian_renderer.ours.render_ours/integrate_ours`; raises `ValueError` otherwise. `multiview_rasterizer="ours"` selects `render_ours` for the multiview loss; `None` means "reuse the primary render function". |
| `load_regularization_configs` | `(args) -> (depth_order_config, normal_field_config, multiview_config, milo_config)` | Loads `gaussian_wrapping/configs/<group>/<name>.yaml` for each enabled regularizer (`args.depth_order`, `args.use_normal_field`, `args.multiview`, `args.milo`); disabled regularizers yield `None`. |

## Minimal examples

### Train and keep the model in memory

```python
from gaussian_wrapping.api import TrainingConfig, run_training

result = run_training(TrainingConfig(
    source_path="/data/translab/scene_01",
    model_path="/output/scene_01",
    iterations=30_000,
    rasterizer="radegs",
    N_max_gaussians=6_000_000,
))
print(result.duration_seconds, result.gaussians._xyz.shape)
```

### Load a trained model and render

```python
from gaussian_wrapping.api import GaussianWrappingScene

scene = GaussianWrappingScene(source_path="/data/translab/scene_01",
                              model_path="/output/scene_01")
scene.load(iteration=30_000)
renders = scene.render_all(split="test")
```

### Resume from a checkpoint

```python
run_training(TrainingConfig(
    source_path="/data/scene", model_path="/output/scene",
    start_checkpoint="/output/scene/chkpnt7000.pth",
    iterations=30_000,
))
```

## See also

- [cli-reference.md](cli-reference.md) — the flag equivalent of every field above.
- [training-guide.md](training-guide.md) — what the training loop does with these values.
- [regularization-configs.md](regularization-configs.md) — the YAML configs referenced by `*_config` fields.
