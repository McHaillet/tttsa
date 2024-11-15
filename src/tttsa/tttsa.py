"""Main program for aligning tilt-series."""

import numpy as np
import torch
from cryotypes.projectionmodel import ProjectionModel
from cryotypes.projectionmodel import ProjectionModelDataLabels as PMDL
from rich.console import Console
from rich.progress import track
from torch_fourier_shift import fourier_shift_image_2d
from torch_refine_tilt_axis_angle import refine_tilt_axis_angle
from torch_tiltxcorr import tiltxcorr, tiltxcorr_no_stretch

from tttsa.back_projection import filtered_back_projection_3d
from .optimizers import optimize_tilt_angle_offset
from .projection_matching import projection_matching
from .utils import circle, fit_ice_slab_to_tomo

# update shift
PMDL.SHIFT = [PMDL.SHIFT_Y, PMDL.SHIFT_X]

# import logging
# log = logging.getLogger(__name__)

console = Console()


def tilt_series_alignment(
    tilt_series: torch.Tensor,
    projection_model_prior: ProjectionModel,
    alignment_z_height: int,
    find_tilt_angle_offset: bool = True,
) -> ProjectionModel:
    """Align a tilt-series using AreTomo-style projection matching.

    AreTomo paper:
        Zheng, Shawn, et al. "AreTomo: An integrated software package for automated
        marker-free, motion-corrected cryo-electron tomographic alignment and
        reconstruction." Journal of Structural Biology: X 6 (2022): 100068.
    On GitHub:
        https://github.com/czimaginginstitute/AreTomo2
        https://github.com/czimaginginstitute/AreTomo3
    """
    # set tomogram and tilt-series shape
    device = tilt_series.device
    n_tilts, h, w = tilt_series.shape
    size = min(h, w)
    tomogram_dimensions = (alignment_z_height, size, size)
    tilt_dimensions = (size,) * 2
    reference_tilt = int(projection_model_prior[PMDL.ROTATION_Y].abs().argmin())

    # mask for coarse alignment
    coarse_alignment_mask = circle(  # ttmask -> tt-shapes; maybe add function
        radius=size // 3,
        smoothing_radius=size // 6,
        image_shape=tilt_dimensions,
        device=device,
    )

    console.print("=== Starting teamtomo tilt-series alignment!", style="bold blue")

    # make a copy of the ProjectionModel to store alignments in
    projection_model = projection_model_prior.copy(deep=True)
    projection_model[PMDL.SHIFT] = -tiltxcorr_no_stretch(
        tilt_series=tilt_series,
        tilt_angles=projection_model[PMDL.ROTATION_Y],
        low_pass_cutoff=0.5,
    ).numpy()
    # do an IMOD style coarse tilt-series alignment
    start_taa_grid_points = 1  # taa = tilt-axis angle
    pm_taa_grid_points = 1  # pm = projection matching

    console.print(
        f"=== Optimizing tilt-axis angle with {start_taa_grid_points} grid point."
    )

    for _ in track(range(3)):  # optimize tilt axis angle
        projection_model[PMDL.SHIFT] = -tiltxcorr(
            tilt_series=tilt_series,
            tilt_angles=projection_model[PMDL.ROTATION_Y],
            tilt_axis_angle=torch.mean(
                torch.as_tensor(projection_model[PMDL.ROTATION_Z])
            ),
            low_pass_cutoff=0.5,
        ).numpy()
        aligned_tilt_series = fourier_shift_image_2d(
            tilt_series,
            shifts=-torch.as_tensor(
                projection_model[PMDL.SHIFT].to_numpy(), device=device
            ),
        )
        projection_model[PMDL.ROTATION_Z] = (
            refine_tilt_axis_angle(
                tilt_series=aligned_tilt_series,
                tilt_axis_angle=torch.mean(
                    torch.as_tensor(projection_model[PMDL.ROTATION_Z])
                ),
                grid_points=start_taa_grid_points,
                return_single_angle=False,
            )
            .cpu()
            .numpy()
        )

    console.print(
        f"=== New tilt axis angle: "
        f"{projection_model[PMDL.ROTATION_Z].mean():.2f}° +-"
        f" {projection_model[PMDL.ROTATION_Z].std():.2f}°"
    )

    ry, rx, alignment_z_height, z_offset = fit_ice_slab_to_tomo(
        filtered_back_projection_3d(tilt_series, tomogram_dimensions, projection_model)
        .cpu()
        .numpy()
    )
    print(ry, rx, alignment_z_height, z_offset)
    projection_model[PMDL.ROTATION_Y] += ry
    projection_model[PMDL.ROTATION_X] += rx
    tomogram_dimensions = (alignment_z_height, size, size)

    if find_tilt_angle_offset:
        full_offset = torch.tensor(0.0)
        console.print("=== Optimizing tilt-angle offset.")
        for _ in track(range(3)):
            tilt_angle_offset = optimize_tilt_angle_offset(
                tilt_series,
                coarse_alignment_mask,
                torch.as_tensor(projection_model[PMDL.ROTATION_Y]),
                torch.as_tensor(projection_model[PMDL.ROTATION_Z]),
                torch.as_tensor(projection_model[PMDL.SHIFT].to_numpy()),
            )
            full_offset += tilt_angle_offset.detach()
            projection_model[PMDL.ROTATION_Y] += float(tilt_angle_offset.detach())

            projection_model[PMDL.SHIFT] = -tiltxcorr(
                tilt_series=tilt_series,
                tilt_angles=projection_model[PMDL.ROTATION_Y],
                tilt_axis_angle=torch.mean(
                    torch.as_tensor(projection_model[PMDL.ROTATION_Z])
                ),
                low_pass_cutoff=0.5,
            ).numpy()
        console.print(f"=== Detected tilt-angle offset: {full_offset:.2f}°")

    # some optimizations parameters
    max_iter = 10  # this seems solid
    tolerance = 0.1  # should probably be related to pixel size
    prev_shifts = projection_model[PMDL.SHIFT].to_numpy()
    console.print(
        f"=== Starting projection matching with"
        f" {pm_taa_grid_points} grid points for the tilt-axis angle."
    )
    for i in range(max_iter):
        aligned_tilt_series = fourier_shift_image_2d(
            tilt_series,
            shifts=-torch.as_tensor(
                projection_model[PMDL.SHIFT].to_numpy(), device=device
            ),
        )
        projection_model[PMDL.ROTATION_Z] = (
            refine_tilt_axis_angle(
                tilt_series=aligned_tilt_series,
                tilt_axis_angle=torch.mean(
                    torch.as_tensor(projection_model[PMDL.ROTATION_Z])
                ),
                grid_points=pm_taa_grid_points,
                return_single_angle=False,
            )
            .cpu()
            .numpy()
        )

        projection_model, _ = projection_matching(
            tilt_series,
            projection_model,
            reference_tilt,
            coarse_alignment_mask,
            tomogram_dimensions,
        )

        shifts = projection_model[PMDL.SHIFT].to_numpy()
        abs_diff = np.abs(prev_shifts - shifts)
        console.print(
            f"--> Iteration {i + 1}, "
            f"sum of translation differences ="
            f" {abs_diff.sum():.2f}"
        )

        if np.all(abs_diff < tolerance):
            break

        prev_shifts = shifts

    console.print(
        f"=== Final tilt-axis angle: {projection_model[PMDL.ROTATION_Z].mean():.2f}° +-"
        f" {projection_model[PMDL.ROTATION_Z].std():.2f}°"
    )
    console.print("===== Done!")

    return projection_model
