"""Filtered-back projection in real space."""

from typing import Tuple

import einops
import torch
import torch.nn.functional as F
from cryotypes.projectionmodel import ProjectionModel
from cryotypes.projectionmodel import ProjectionModelDataLabels as PMDL

from tttsa.affine import affine_transform_2d
from tttsa.transformations import (
    projection_model_to_backproject_matrix,
    projection_model_to_tsa_matrix,
)
from tttsa.utils import array_to_grid_sample, prep_grid_cached

# update shift
PMDL.SHIFT = [PMDL.SHIFT_Y, PMDL.SHIFT_X]


def filtered_back_projection_3d(
    tilt_series: torch.Tensor,
    tomogram_dimensions: Tuple[int, int, int],
    projection_model: ProjectionModel,
    weighting: str = "hamming",
    object_diameter: float | None = None,
) -> torch.Tensor:
    """Run weighted back projection incorporating some alignment parameters.

    weighting: str, default "hamming"
        all filters here start at 1/N (instead of 0 for ramp and hamming) which
        improves the low res signal upon forward projection of the reconstruction
        Options:
            - "ramp": increases linearly from 1/N to 1 from the zero frequency to
                nyquist
            - "exact": is based on and improves low-res signal on forward projection:
                Reference : Optik, Exact filters for general geometry three-dimensional
                reconstruction, vol.73,146,1986.
            - "hamming": modified hamming as used in AreTomo, further modified here to
                also start a 1/N
    object_diameter: float | None, default None
        object diameter specified in number of pixels, only needed for the exact filter

    """
    # initializes sizes
    device = tilt_series.device
    n_tilts, h, w = tilt_series.shape  # for simplicity assume square images
    tilt_image_dimensions = (h, w)
    transformed_image_dimensions = tomogram_dimensions[-2:]
    _, filter_size = transformed_image_dimensions

    # generate the 2d alignment affine matrix
    M = projection_model_to_tsa_matrix(
        projection_model, tilt_image_dimensions, transformed_image_dimensions
    ).to(device)

    aligned_ts = affine_transform_2d(
        tilt_series,
        M,
        out_shape=transformed_image_dimensions,
    )

    # generate weighting function and apply to aligned tilt series
    if weighting == "exact":
        if object_diameter is None:
            raise ValueError(
                "Calculation of exact weighting requires an object " "diameter."
            )
        if n_tilts == 1:
            # set explicitly as tensor to ensure correct typing
            filters = torch.tensor(1.0, device=device)
        else:  # slice_width could be provided as a function argument it can be
            # calculated as: (pixel_size * 2 * imdim) / object_diameter
            q = einops.rearrange(
                torch.arange(
                    filter_size // 2 + filter_size % 2 + 1,
                    dtype=torch.float32,
                    device=device,
                )
                / filter_size,
                "q -> 1 1 q",
            )
            tilt_angles = torch.as_tensor(projection_model[PMDL.ROTATION_Y])
            sampling = torch.sin(
                torch.deg2rad(
                    torch.abs(einops.rearrange(tilt_angles, "n -> n 1") - tilt_angles)
                )
            ).to(device)
            sampling = einops.rearrange(sampling, "n m -> n m 1")
            q_overlap_inv = sampling / (2 / object_diameter)
            over_weighting = 1 - torch.clip(q * q_overlap_inv, min=0, max=1)
            filters = 1 / einops.reduce(over_weighting, "n m q -> n q", "sum")
            filters = einops.rearrange(filters, "n w -> n 1 w")
    elif weighting == "ramp":
        filters = torch.arange(
            filter_size // 2 + filter_size % 2 + 1, dtype=torch.float32, device=device
        )
        filters /= filters.max()
        filters = filters * (1 - 1 / n_tilts) + 1 / n_tilts  # start at 1 / N
    elif weighting == "hamming":  # AreTomo3 code uses a modified hamming window
        # 2 * q * (0.55f + 0.45f * cosf(6.2831852f * q))  # with q from 0 to .5 (Ny)
        # https://github.com/czimaginginstitute/AreTomo3/blob/
        #   c39dcdad9525ee21d7308a95622f3d47fe7ab4b9/AreTomo/Recon/GRWeight.cu#L20
        q = (
            torch.arange(
                filter_size // 2 + filter_size % 2 + 1,
                dtype=torch.float32,
                device=device,
            )
            / filter_size
        )
        # regular hamming: q * (.54 + .46 * torch.cos(torch.pi * q))
        filters = 2 * q * (0.54 + 0.46 * torch.cos(2 * torch.pi * q))
        filters /= filters.max()  # 0-1 normalization
        filters = filters * (1 - 1 / n_tilts) + 1 / n_tilts  # start at 1 / N
    else:
        raise ValueError("Invalid weighting option provided for FBP.")

    weighted = torch.fft.irfftn(
        torch.fft.rfftn(aligned_ts, dim=(-2, -1)) * filters, dim=(-2, -1)
    )
    if len(weighted.shape) == 2:  # rfftn gets rid of batch dimension: add it back
        weighted = einops.rearrange(weighted, "h w -> 1 h w")

    # We need to lingalg.inv the matrix as the affine transform is done inside
    # this function. It could be more logical to use affine_transform_3d (and do
    # inversion inside) but it requires recalculation of the grid for every iteration.
    M = einops.rearrange(
        torch.linalg.inv(
            projection_model_to_backproject_matrix(
                projection_model, tomogram_dimensions
            )
        ),
        "... i j -> ... 1 1 i j",
    ).to(device)

    reconstruction = torch.zeros(
        tomogram_dimensions, dtype=torch.float32, device=device
    )
    grid = prep_grid_cached(tomogram_dimensions, device)

    for i in range(n_tilts):  # could do all grids simultaneously
        grid_t = M[i] @ grid
        grid_t = einops.rearrange(grid_t, "... d h w coords 1 -> ... d h w coords")[
            ..., :3
        ].contiguous()
        grid_t = array_to_grid_sample(grid_t, tomogram_dimensions)
        reconstruction += torch.squeeze(
            F.grid_sample(
                einops.rearrange(weighted[i], "h w -> 1 1 1 h w"),
                einops.rearrange(grid_t, "d h w coords -> 1 d h w coords"),
                align_corners=True,
                mode="bilinear",
            )
        )
    return reconstruction
