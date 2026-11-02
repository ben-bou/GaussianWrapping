"""High-level API for GaussianWrapping.

This module provides a clean, object-oriented interface that wraps the
lower-level scene/model classes for easier programmatic use.
"""

import os
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
