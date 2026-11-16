# AGENTS.md — GaussianWrapping

Guidance for AI agents (and humans) working in this repository. Read this
before making changes; it tells you what the project is, how to validate work,
and which parts are fragile.

## What this repository is

`gaussian_wrapping` is a research Python package for 3D Gaussian Splatting
(3DGS) with mesh extraction ("Gaussian wrapping"). It trains Gaussian scenes
with several pluggable CUDA rasterizers and geometry regularizers (learnable
normal field, multiview NCC, mesh-in-the-loop/MILo, depth-normal consistency,
optional Depth-Anything-V2 depth-order prior), then extracts meshes via
pivot-based Delaunay/marching-tetrahedra or Primal Adaptive Meshing.

It is derived from the 3DGS / RaDe-GS / MILo line of work. Version: `1.0.0`
(`pyproject.toml`).

**Companion repository:** `GW_experiments` (sibling checkout, typically
`~/Projects/Thesis/GW_experiments`) is the downstream consumer. It talks to
this package **only** through `gaussian_wrapping.api`
(`TrainingConfig` + `run_training`) — never by shelling out to the CLI.
Changes to the public API surface must keep that contract intact (see below).

## Documentation map (read before touching code)

Full reference docs live in `docs/` and were generated from the source — keep
them in sync when you change behavior:

| Doc | Covers |
|-----|--------|
| `docs/index.md` | Overview + links |
| `docs/installation.md` | Environment, CUDA submodules, `install.py` |
| `docs/cli-reference.md` | Every CLI flag of every entry point (251 flags) |
| `docs/api-reference.md` | `TrainingConfig` (78 fields), `run_training`, `GaussianWrappingScene` |
| `docs/training-guide.md` | Training loop behavior, losses, outputs, checkpoints |
| `docs/regularization-configs.md` | Every YAML under `gaussian_wrapping/configs/`, key by key |
| `docs/mesh-extraction.md` | Both extraction pipelines, bounding volumes, Blender add-on |
| `docs/evaluation.md` | TNT/DTU/MipNeRF360/NeRF-Synthetic benchmarks, metrics |
| `docs/architecture.md` | Module-by-module internals |

## Repository layout (key files)

```
install.py                       # conda/CUDA bootstrap; builds all submodules
pyproject.toml                   # package metadata; console scripts gw-train / gw-render
gaussian_wrapping/
├── train.py                     # training loop + CLI main() (gw-train)
├── api.py                       # PUBLIC API: TrainingConfig (78 fields), run_training
├── arguments/__init__.py        # ModelParams / OptimizationParams / PipelineParams
├── configs/                     # YAML regularizer configs (4 wired groups + legacy mesh/)
├── scene/                       # Scene, GaussianModel, cameras, COLMAP loaders
├── gaussian_renderer/           # radegs / ours / sof rasterizers, network_gui
├── regularization/              # normal_field, multiview, sdf, regularizer/ adapters
├── scripts/                     # end-to-end benchmark & train+extract wrappers
└── eval/                        # tnt/, dtu/; TNTUniScanEvals git submodule
submodules/                      # CUDA rasterizers, simple-knn, fused-ssim,
                                 # tetra_triangulation, nvdiffrast, Depth-Anything-V2, GGGS
```

## Environment realities

- **Training requires CUDA + compiled submodules** (four
  `diff-gaussian-rasterization` variants, `simple-knn`, `fused-ssim`,
  `tetra_triangulation`, `nvdiffrast`). A machine without them can still
  import `gaussian_wrapping.api` metadata but cannot run training.
- **There is no test suite in this package.** Validation options:
  1. `python -c "from gaussian_wrapping.api import TrainingConfig, run_training"` — import sanity.
  2. The downstream contract check (works without CUDA):
     `cd ../GW_experiments && python scripts/check_api_contract.py --package-path <this repo>`
  3. A GPU smoke run: `gw-train -s <DATASET> -m /tmp/smoke --iterations 200 --densify_until_iter 100`.
- Network access may be sandboxed in agent environments; `pip install` or
  submodule clones may need approval.

## The API contract (do not break casually)

`GW_experiments` flattens its YAML schema into
`gaussian_wrapping.api.TrainingConfig(**kwargs)` and calls `run_training()`.
Consequences:

- `TrainingConfig` currently has **78 fields**; the downstream contract test
  and `scripts/check_api_contract.py` assert that every field the experiments
  schema sends is accepted. Removing/renaming a field breaks the downstream
  repo at runtime (`TrainingError: TrainingConfig does not accept ...`).
- `run_training()` forces two values regardless of input:
  `use_abs_grad_for_densification = True` and
  `mask_depth_normal = True` when `rasterizer == "ours"`. It also appends the
  final iteration to `save_iterations`/`test_iterations`. Preserve these
  behaviors or coordinate the change downstream.
- `validate()` rejects: empty `source_path`/`model_path`, `rasterizer` not in
  `{"radegs", "ours"}`, `multiview_rasterizer` not in `{None, "ours"}`,
  `resolution` not in `{-1,0,1,2}`, `sh_degree` not in `0..3`,
  `iterations <= 0`, `port` not `-1`/`1..65535`.

## Adding a `TrainingConfig` field (checklist)

1. `gaussian_wrapping/api.py`: add the dataclass field with a default matching
   the CLI default; extend `validate()` if it needs range checks; if it maps to
   `ModelParams`/`OptimizationParams`/`PipelineParams`, extend the
   corresponding `_*_params_from_config()` helper.
2. `gaussian_wrapping/train.py`: add the argparse flag in `main()` **and**
   make sure the training loop actually reads it (either via the extracted
   param groups or the flat `args` namespace). A flag nothing reads is dead
   configuration — avoid adding those.
3. Downstream (`../GW_experiments`): add the field to `utils/config.py`
   (dataclass + `to_package_kwargs()`), to `utils/command.py` (flag mapping or
   `NON_CLI_FIELDS` — a meta-test enforces this), plus tests and docs. Their
   contract test will fail until you do.
4. Docs: update `docs/api-reference.md`, `docs/cli-reference.md`, and if it
   touches regularizer behavior `docs/training-guide.md` /
   `docs/regularization-configs.md`.

## Regularizer configs (`gaussian_wrapping/configs/`)

- Only four groups are wired into training: `normal_field/`, `multiview/`,
  `mesh_in_the_loop/`, `depth_order/`, loaded by
  `train.load_regularization_configs()` as
  `configs/<group>/<name>.yaml` selected by the `--*_config` flags.
- **`configs/mesh/` (37 files) and `configs/fast` are legacy/inert** — nothing
  loads them. Do not add keys there expecting effect; the snapping/SDF
  consumers are unwired legacy code.
- YAMLs are plain dicts read with `yaml.safe_load` — **no schema validation**;
  a typo surfaces as `KeyError` deep in a regularizer. Known crash-if-enabled
  keys: `use_sdf` (normal_field), `use_rgb_loss`/`use_pivots_colors` (milo)
  raise `NotImplementedError`.
- Normal-field densification/pruning windows compare `iteration + 1` against
  the config values (off-by-one convention — keep it consistent when editing).

## Known issues / gotchas (verified, not speculation)

- `.gitignore` contains a bare `output*` pattern (line 44) that matches any
  file or directory whose name starts with `output` — including documentation
  files like `docs/outputs.md`. That is why the artifacts reference is named
  `docs/artifacts.md`. Do not create tracked files named `output*` (or add a
  `!`-negation in `.gitignore` if you must).
- `gw-render` console script is **broken**: `pyproject.toml` points at
  `gaussian_wrapping.render:main` but `render.py` defines no `main()`.
  `python -m gaussian_wrapping.render` works. Fixing it is a one-line
  pyproject change plus a `main()` wrapper — coordinate with docs.
- `mesh_decimate.py`: `--out` is parsed but never used; passing it raises
  `NameError`. Only the default `_decimated_with_blender.ply` path works.
- `evaluate_dtu_mesh.py`: `--skip_train`, `--skip_test`, `--quiet`,
  `--scan_id` are parsed but unused; it shells out via relative path
  `python eval/dtu/eval.py`, so CWD matters; it hard-codes input mesh name
  `recon_post.ply`, which no extraction script produces.
- `primal_adaptive_meshing_extraction.py`: registers `ModelParams` without
  `sentinel=True`, so its CLI defaults always override `cfg_args`; accepts
  `--rasterizer radegs` but raises `ValueError` at runtime; with
  `--bounding_box_method ground_truth` it silently depends on the
  TNTUniScanEvals submodule being initialized.
- `eval/tnt/run.py --view-crop` is force-overwritten to `False`; its
  `.json`-trajectory branch references imports that are commented out.
- `render.py` defaults to `--rasterizer radegs` while
  `api.GaussianWrappingScene.render` defaults to `"ours"` — intentional
  difference, don't "unify" blindly.
- `Scene` never writes into the dataset; the initial point cloud is built in
  memory and written as `<model_path>/input.ply` (+ `points3D.ply` only with
  `save_init_ply`).
- The `llff` parameter is only the eval-split stride (`llffhold`) for COLMAP
  scenes; there is no LLFF dataset reader.

## Conventions

- Research codebase: keep diffs minimal and surgical; don't refactor
  `scene/gaussian_model.py` (1,931 lines) or `scene/mesh.py` (1,387 lines)
  opportunistically.
- Python ≥ 3.9 compatibility is required (`pyproject.toml`).
- Console-script names, flag names and YAML key names are load-bearing
  (downstream repo + docs reference them verbatim).
- Docs in `docs/` are part of the deliverable: any behavior change should come
  with the matching doc update. Docs style: GitHub-flavored Markdown, no
  emojis, `> Source of truth:` blockquote listing the files documented,
  tables with exact defaults transcribed from code.
- `datastructure.txt`, `test_file.txt` at the repo root are scratch/untracked
  files — ignore them.

## Git

- `outputs/`, `logs/`, `__pycache__/` are gitignored runtime artifacts.
- Submodules must stay initialized (`git submodule update --init --recursive`)
  for installs and for PAM ground-truth bounding volumes.
- Do not commit unless explicitly asked; keep commits focused on one concern.
