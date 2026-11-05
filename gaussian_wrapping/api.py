"""High-level API for GaussianWrapping.

This module provides a clean, object-oriented interface that wraps the
lower-level scene/model classes for easier programmatic use.
"""

import os
import random
import time
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import Optional, List, Dict, Any
import torch

from gaussian_wrapping.scene import Scene, GaussianModel
from gaussian_wrapping.arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_wrapping.gaussian_renderer import render_imp, render_simp


class GaussianWrappingScene:
    """High-level wrapper around Scene + GaussianModel for easy programmatic use."""

    def __init__(
        self,
        source_path: str,
        model_path: Optional[str] = None,
        sh_degree: int = 3,
        images: str = "images",
        white_background: bool = False,
        resolution: int = -1,
        eval: bool = False,
        device: str = "cuda",
    ):
        """Initialize a scene from a COLMAP or Blender dataset.

        Args:
            source_path: Path to the dataset source directory.
            model_path: Path where the model will be saved/loaded.
            sh_degree: Maximum spherical harmonics degree (default 3).
            images: Subdirectory name containing images (default "images").
            white_background: Whether to use a white background (default False).
            resolution: Resolution scale (-1 for native, 0, 1, 2 for 1/2, 1/4, 1/8).
            eval: Whether to split into train/eval sets (default False).
            device: Device to place data on (default "cuda").
        """
        self.source_path = os.path.abspath(source_path)
        self.model_path = model_path or os.path.join(self.source_path, "output")
        
        # Create a minimal args namespace matching ModelParams expectations
        self._model_args = _SimpleModelParams(
            source_path=self.source_path,
            model_path=os.path.abspath(self.model_path),
            images=images,
            sh_degree=sh_degree,
            white_background=white_background,
            resolution=resolution,
            eval=eval,
            data_device=device,
        )
        
        # Create the Gaussian model
        self.gaussians = GaussianModel(sh_degree)
        
        # Create the scene (loads cameras and initializes Gaussians from point cloud)
        self.scene = Scene(self._model_args, self.gaussians)
        
        self.train_cameras = self.scene.getTrainCameras()
        self.test_cameras = self.scene.getTestCameras()
        
        print(f"[GaussianWrappingScene] Scene initialized with "
              f"{len(self.train_cameras)} train cameras, "
              f"{len(self.test_cameras)} test cameras")

    def render(
        self,
        camera,
        pipeline=None,
        background=None,
        scaling_modifier=1.0,
        override_color=None,
        rasterizer="ours",
    ) -> Dict[str, torch.Tensor]:
        """Render a single view.

        Args:
            camera: A Camera object from the scene.
            pipeline: PipelineParams (optional, uses defaults if None).
            background: Background color tensor (default: white or black).
            scaling_modifier: Scaling modifier for Gaussians (default 1.0).
            override_color: Optional override color for Gaussians.
            rasterizer: Which rasterizer to use ("ours" or "radegs").

        Returns:
            Dict with keys "render", "depth", "alpha", etc.
        """
        if pipeline is None:
            pipeline = _SimplePipelineParams()
        
        if background is None:
            bg_color = [1, 1, 1] if self._model_args.white_background else [0, 0, 0]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        if rasterizer == "ours":
            from gaussian_wrapping.gaussian_renderer.ours import render_ours as render_fn
        elif rasterizer == "radegs":
            from gaussian_wrapping.gaussian_renderer.radegs import render_radegs as render_fn
        else:
            raise ValueError(f"Unknown rasterizer: {rasterizer}")
        
        return render_fn(camera, self.gaussians, pipeline, background, scaling_modifier, override_color)

    def render_all(
        self,
        split: str = "train",
        pipeline=None,
        background=None,
    ) -> List[Dict[str, torch.Tensor]]:
        """Render all cameras in the scene.

        Args:
            split: "train" or "test" (default "train").
            pipeline: PipelineParams (optional).
            background: Background color tensor (optional).

        Returns:
            List of render dicts, one per camera.
        """
        cameras = self.train_cameras if split == "train" else self.test_cameras
        results = []
        for cam in cameras:
            results.append(self.render(cam, pipeline, background))
        return results

    def save(self, iteration: int = 0):
        """Save the current Gaussian model to disk."""
        point_cloud_path = os.path.join(self.model_path, "point_cloud", f"iteration_{iteration}")
        os.makedirs(point_cloud_path, exist_ok=True)
        self.scene.save(iteration)
        print(f"[GaussianWrappingScene] Saved model at iteration {iteration}")

    def load(self, iteration: int):
        """Load a trained Gaussian model from disk."""
        self.scene = Scene(self._model_args, self.gaussians, load_iteration=iteration)
        print(f"[GaussianWrappingScene] Loaded model at iteration {iteration}")


class _SimpleModelParams:
    """Lightweight container matching the ModelParams interface without argparse."""
    
    def __init__(self, source_path, model_path, images, sh_degree, 
                 white_background, resolution, eval, data_device):
        self.source_path = source_path
        self.model_path = model_path
        self.images = images
        self.sh_degree = sh_degree
        self.white_background = white_background
        self.resolution = resolution
        self.eval = eval
        self.data_device = data_device
        self.llff = 8
        self.kernel_size = 0.0
        self.use_unbounded_opacity = False


class _SimplePipelineParams:
    """Lightweight container matching the PipelineParams interface."""
    
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False


def create_scene(
    source_path: str,
    model_path: Optional[str] = None,
    **kwargs,
) -> GaussianWrappingScene:
    """Convenience function to create a GaussianWrappingScene.

    Args:
        source_path: Path to the dataset source directory.
        model_path: Path where the model will be saved/loaded.
        **kwargs: Additional keyword arguments passed to GaussianWrappingScene.

    Returns:
        GaussianWrappingScene: The initialized scene.
    """
    return GaussianWrappingScene(source_path=source_path, model_path=model_path, **kwargs)


# --------------------------------------------------------------------------- #
# Training API
# --------------------------------------------------------------------------- #


@dataclass
class TrainingConfig:
    """Complete, reproducible description of a training run.

    Every value that the training loop reads from its ``args`` namespace is
    represented here, so a run can be fully reconstructed from this object.
    Defaults match the command line entry point in ``gaussian_wrapping.train``.
    """

    # --- Dataset ---------------------------------------------------------
    source_path: str = ""
    model_path: str = ""
    images: str = "images"
    resolution: int = -1
    eval: bool = False
    white_background: bool = False
    data_device: str = "cuda"
    use_unbounded_opacity: bool = False
    llff: int = 8
    kernel_size: float = 0.0

    # --- Optimization ----------------------------------------------------
    iterations: int = 30_000
    sh_degree: int = 3
    position_lr_init: float = 0.00016
    position_lr_final: float = 0.0000016
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 30_000
    feature_dc_lr: float = 0.0025
    feature_rest_lr: float = 0.0025 / 20.0
    opacity_lr: float = 0.05
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001
    percent_dense: float = 0.01
    lambda_dssim: float = 0.2
    densification_interval: int = 100
    opacity_reset_interval: int = 3000
    densify_from_iter: int = 500
    densify_until_iter: int = 15_000
    densify_grad_threshold: float = 0.0002
    random_background: bool = False
    appearance_embeddings_lr: float = 0.001
    appearance_network_lr: float = 0.001
    gaussian_features_lr: float = 0.05 / 2.0
    pgsr_appearance_lr: float = 0.001

    # --- Pipeline --------------------------------------------------------
    convert_SHs_python: bool = False
    compute_cov3D_python: bool = False
    debug: bool = False

    # --- Runtime ---------------------------------------------------------
    rasterizer: str = "radegs"
    multiview_rasterizer: Optional[str] = None
    enable_gui: bool = False
    ip: str = "127.0.0.1"
    port: int = -1
    debug_from: int = -1
    detect_anomaly: bool = False
    quiet: bool = False

    # --- Regularization --------------------------------------------------
    use_normal_field: bool = True
    normal_field_config: str = "default_regular_densification"
    multiview: bool = True
    multiview_config: str = "fast"
    multiview_factor: float = 1.0
    milo: bool = False
    milo_config: str = "default_regular_densification"
    depth_order: bool = False
    depth_order_config: str = "default"
    regularization_from_iter: int = 7_000
    lambda_depth_normal: float = 0.05
    mask_depth_normal: bool = False
    dense_gaussians: bool = False
    detach_gaussian_rendering: bool = False
    N_max_gaussians: Optional[int] = None
    warn_until_iter: int = 3000
    non_maximal_pruning_iterations: List[int] = field(default_factory=list)
    use_max_size_threshold: bool = False
    use_scale_loss: bool = False
    scale_loss_weight: float = 100.0
    weight_by_img_grad: bool = False
    disable_mip_filter: bool = False
    update_mip_filter_every: int = 100
    decoupled_appearance: bool = False
    exposure_compensation: bool = True

    # --- Iterations / checkpoints ---------------------------------------
    save_iterations: List[int] = field(default_factory=list)
    test_iterations: List[int] = field(default_factory=list)
    checkpoint_iterations: List[int] = field(default_factory=list)
    start_checkpoint: Optional[str] = None

    # --- Logging ---------------------------------------------------------
    log_interval: Optional[int] = None
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_name: Optional[str] = None

    def validate(self) -> None:
        """Minimal validation of values that would otherwise fail deep in training."""
        if not self.source_path:
            raise ValueError("TrainingConfig.source_path is required.")
        if not self.model_path:
            raise ValueError("TrainingConfig.model_path is required.")
        if self.rasterizer not in ("radegs", "ours"):
            raise ValueError(
                f"TrainingConfig.rasterizer must be 'radegs' or 'ours' (got {self.rasterizer!r})."
            )
        if self.multiview_rasterizer not in (None, "ours"):
            raise ValueError(
                "TrainingConfig.multiview_rasterizer must be None or 'ours' "
                f"(got {self.multiview_rasterizer!r})."
            )
        if self.resolution not in (-1, 0, 1, 2):
            raise ValueError(
                f"TrainingConfig.resolution must be -1, 0, 1 or 2 (got {self.resolution})."
            )
        if self.sh_degree not in (0, 1, 2, 3):
            raise ValueError(
                f"TrainingConfig.sh_degree must be in 0..3 (got {self.sh_degree})."
            )
        if self.iterations <= 0:
            raise ValueError(
                f"TrainingConfig.iterations must be > 0 (got {self.iterations})."
            )
        if self.port != -1 and not (1 <= self.port <= 65535):
            raise ValueError(
                f"TrainingConfig.port must be -1 or in 1..65535 (got {self.port})."
            )


@dataclass
class TrainingResult:
    """Return value of :func:`run_training`."""

    scene: Any
    gaussians: Any
    model_path: str
    iterations: int
    duration_seconds: float


def _model_params_from_config(config: TrainingConfig) -> SimpleNamespace:
    return SimpleNamespace(
        source_path=config.source_path,
        model_path=config.model_path,
        images=config.images,
        sh_degree=config.sh_degree,
        white_background=config.white_background,
        resolution=config.resolution,
        eval=config.eval,
        data_device=config.data_device,
        llff=config.llff,
        kernel_size=config.kernel_size,
        use_unbounded_opacity=config.use_unbounded_opacity,
    )


def _optimization_params_from_config(config: TrainingConfig) -> SimpleNamespace:
    return SimpleNamespace(
        iterations=config.iterations,
        position_lr_init=config.position_lr_init,
        position_lr_final=config.position_lr_final,
        position_lr_delay_mult=config.position_lr_delay_mult,
        position_lr_max_steps=config.position_lr_max_steps,
        feature_dc_lr=config.feature_dc_lr,
        feature_rest_lr=config.feature_rest_lr,
        opacity_lr=config.opacity_lr,
        scaling_lr=config.scaling_lr,
        rotation_lr=config.rotation_lr,
        percent_dense=config.percent_dense,
        lambda_dssim=config.lambda_dssim,
        densification_interval=config.densification_interval,
        opacity_reset_interval=config.opacity_reset_interval,
        densify_from_iter=config.densify_from_iter,
        densify_until_iter=config.densify_until_iter,
        densify_grad_threshold=config.densify_grad_threshold,
        random_background=config.random_background,
        appearance_embeddings_lr=config.appearance_embeddings_lr,
        appearance_network_lr=config.appearance_network_lr,
        gaussian_features_lr=config.gaussian_features_lr,
        pgsr_appearance_lr=config.pgsr_appearance_lr,
    )


def _pipeline_params_from_config(config: TrainingConfig) -> SimpleNamespace:
    return SimpleNamespace(
        convert_SHs_python=config.convert_SHs_python,
        compute_cov3D_python=config.compute_cov3D_python,
        debug=config.debug,
    )


def run_training(config: TrainingConfig) -> TrainingResult:
    """Train GaussianWrapping in-process from a typed configuration.

    This is the programmatic equivalent of running
    ``python -m gaussian_wrapping.train`` with the corresponding flags.  It
    performs the same global setup (RNG seeding, optional GUI server, anomaly
    detection), calls the training loop with explicit rasterizer callables, and
    returns the trained scene and Gaussian model.

    Parameters
    ----------
    config:
        A validated :class:`TrainingConfig`.

    Returns
    -------
    TrainingResult
        The trained ``scene``/``gaussians`` plus timing information.
    """
    config.validate()

    # Imported lazily so that importing ``gaussian_wrapping.api`` does not pull
    # in the training loop's optional dependencies (fused_ssim, wandb, ...).
    from gaussian_wrapping.train import (
        training,
        resolve_rasterizer,
        load_regularization_configs,
    )
    from gaussian_wrapping.utils.general_utils import safe_state
    from gaussian_wrapping.gaussian_renderer import network_gui

    dataset = _model_params_from_config(config)
    opt = _optimization_params_from_config(config)
    pipe = _pipeline_params_from_config(config)

    # The training loop and regularizers read from a flat ``args`` namespace.
    args = SimpleNamespace(**asdict(config))
    args.use_abs_grad_for_densification = True
    args.no_normal_field = not config.use_normal_field

    # Mirror the CLI: always save and test the final iteration.
    if config.iterations not in args.save_iterations:
        args.save_iterations.append(config.iterations)
    if -1 not in args.test_iterations and config.iterations not in args.test_iterations:
        args.test_iterations.append(config.iterations)

    render_fn, _integrate_fn, multiview_render_fn = resolve_rasterizer(
        config.rasterizer, config.multiview_rasterizer
    )
    if config.rasterizer == "ours":
        args.mask_depth_normal = True

    depth_order_config, normal_field_config, multiview_config, milo_config = \
        load_regularization_configs(args)

    # Global setup, mirroring ``gaussian_wrapping.train.main``.
    safe_state(config.quiet)
    if config.enable_gui:
        port = config.port if config.port != -1 else random.randint(5000, 9000)
        network_gui.init(config.ip, port)
    torch.autograd.set_detect_anomaly(config.detect_anomaly)

    torch.cuda.synchronize()
    start_time = time.time()
    scene, gaussians = training(
        dataset,
        opt,
        pipe,
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        config.start_checkpoint,
        config.debug_from,
        args,
        depth_order_config,
        normal_field_config,
        multiview_config,
        milo_config,
        config.log_interval,
        render_fn=render_fn,
        multiview_render_fn=multiview_render_fn,
        enable_gui=config.enable_gui,
    )
    torch.cuda.synchronize()
    duration = time.time() - start_time

    time_txt_path = os.path.join(config.model_path, "time.txt")
    with open(time_txt_path, "w") as handle:
        handle.write(str(duration))

    return TrainingResult(
        scene=scene,
        gaussians=gaussians,
        model_path=config.model_path,
        iterations=config.iterations,
        duration_seconds=duration,
    )
