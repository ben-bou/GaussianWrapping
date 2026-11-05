"""GaussianWrapping - A Python package for 3D Gaussian Splatting with mesh wrapping."""

from gaussian_wrapping.scene import Scene, GaussianModel
from gaussian_wrapping.arguments import (
    ModelParams,
    PipelineParams,
    OptimizationParams,
    get_combined_args,
    read_config,
)
from gaussian_wrapping.gaussian_renderer import (
    render_old,
    render_imp,
    render_simp,
    render_depth,
    render_full,
    render_isosurface,
)
from gaussian_wrapping import functional
from gaussian_wrapping import extraction
from gaussian_wrapping import api
from gaussian_wrapping.api import TrainingConfig, TrainingResult, run_training

__version__ = "1.0.0"
