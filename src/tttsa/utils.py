"""Utility functions for tttsa."""

from functools import lru_cache
from typing import Sequence, Tuple

import einops
import numpy as np
import scipy.ndimage as ndi
import torch
import torch.nn.functional as F
from torch_grid_utils import coordinate_grid


def fit_ice_slab_to_tomo(tomogram: np.ndarray) -> Tuple[float, float, int, int]:
    """Fit an ice slab to a tomogram to find rotations, z-offset and thickness."""
    from slabify.utils import (
        sample_points,
        variance_at_points,
    )

    N = 10000
    dims = tomogram.shape
    boxsize = 16
    z_min = 1
    z_max = None
    seed = 42
    percentile = 95

    Z_rand, Y_rand, X_rand = sample_points(
        mask_size=dims, N=N, boxsize=boxsize, z_min=z_min, z_max=z_max, seed=seed
    )
    # Calculate the variance around each point:
    variances = variance_at_points(
        tomo=tomogram, Z=Z_rand, Y=Y_rand, X=X_rand, N=N, boxsize=boxsize
    )

    variance_threshold = np.percentile(variances, percentile)
    idx = variances[:] > variance_threshold
    idx = idx.squeeze()
    # variances = variances[idx]
    Z_rand, Y_rand, X_rand = Z_rand[idx], Y_rand[idx], X_rand[idx]

    points = np.array((Z_rand, Y_rand, X_rand))
    centroid = np.mean(points, axis=1, keepdims=True)
    points_centered = points - centroid
    U, _, _ = np.linalg.svd(points_centered)
    normal_vector = U[:, -1]
    z_ref = np.array([1, 0, 0])
    n = normal_vector / np.linalg.norm(normal_vector)  # order is zyx
    n = (
        -n
        if np.sqrt(np.sum((z_ref - -n) ** 2)) < np.sqrt(np.sum((z_ref - n) ** 2))
        else n
    )

    # Step 1: Compute the rotation angle around the y-axis (align projections in the
    # yz-plane
    theta_y = np.arctan2(n[2], n[0])  # atan2(n_x, n_z)

    # Step 2: Compute the rotation angle around the x-axis (align z-components)
    theta_x = -np.arctan2(
        n[1], np.sqrt(n[2] ** 2 + n[0] ** 2)
    )  # atan2(n_y, sqrt(n_x^2 + n_z^2))

    # Step 3: Rotation matrix around the y-axis (for aligning projections in the
    # yz-plane
    Ry = np.flip(
        np.array(
            [
                [np.cos(theta_y), 0, np.sin(theta_y)],
                [0, 1, 0],
                [-np.sin(theta_y), 0, np.cos(theta_y)],
            ]
        )
    )

    # Step 4: Rotation matrix around the x-axis (for aligning z-components)
    Rx = np.flip(
        np.array(
            [
                [1, 0, 0],
                [0, np.cos(theta_x), -np.sin(theta_x)],
                [0, np.sin(theta_x), np.cos(theta_x)],
            ]
        )
    )

    # Combined rotation matrix (first around y, then around x)
    R = np.dot(Ry, Rx)

    flattened_points = np.linalg.inv(R) @ points_centered + centroid

    # calculate top bottom and offset
    z_coords = flattened_points[0]
    upper_quartile = np.percentile(z_coords, 75)
    lower_quartile = np.percentile(z_coords, 25)
    iqr = upper_quartile - lower_quartile
    upper_whisker = z_coords[z_coords <= upper_quartile + 1.5 * iqr].max()
    lower_whisker = z_coords[z_coords >= lower_quartile - 1.5 * iqr].min()

    z_height = upper_whisker - lower_whisker
    center = lower_whisker + z_height / 2
    z_offset = center - tomogram.shape[0] // 2
    return (
        float(np.rad2deg(theta_y)),
        float(np.rad2deg(theta_x)),
        int(z_height),
        int(z_offset),
    )


@lru_cache(maxsize=1)
def prep_grid_cached(tomogram_dimensions: tuple[int], device: str) -> torch.Tensor:
    """Create a coordinate grid in the tomogram space for interpolation."""
    grid = coordinate_grid(tomogram_dimensions, device=device)
    grid = homogenise_coordinates(grid)
    grid = einops.rearrange(grid, "d h w coords -> d h w coords 1")
    return grid


def rfft_shape(input_shape: Sequence[int]) -> Sequence[int]:
    """Get the output shape of an rfft on an input with input_shape."""
    rfft_shape = list(input_shape)
    rfft_shape[-1] = int((rfft_shape[-1] / 2) + 1)
    return tuple(rfft_shape)


def dft_center(
    image_shape: Tuple[int, ...],
    rfft: bool,
    fftshifted: bool,
    device: torch.device | None = None,
) -> torch.LongTensor:
    """Return the position of the DFT center for a given input shape."""
    fft_center = torch.zeros(size=(len(image_shape),), device=device)
    image_shape = torch.as_tensor(image_shape, device=device).float()
    if rfft is True:
        image_shape = torch.tensor(rfft_shape(image_shape), device=device)
    if fftshifted is True:
        fft_center = torch.divide(image_shape, 2, rounding_mode="floor")
    if rfft is True:
        fft_center[-1] = 0
    return fft_center.long()


def homogenise_coordinates(coords: torch.Tensor) -> torch.Tensor:
    """3D coordinates to 4D homogeneous coordinates with ones in the last column.

    Parameters
    ----------
    coords: torch.Tensor
        `(..., 3)` array of 3D coordinates

    Returns
    -------
    output: torch.Tensor
        `(..., 4)` array of homogeneous coordinates
    """
    return F.pad(
        torch.as_tensor(coords, device=coords.device),
        pad=(0, 1),
        mode="constant",
        value=1,
    )


def array_to_grid_sample(
    array_coordinates: torch.Tensor, array_shape: Sequence[int]
) -> torch.Tensor:
    """Generate grids for `torch.nn.functional.grid_sample` from array coordinates.

    These coordinates should be used with `align_corners=True` in
    `torch.nn.functional.grid_sample`.


    Parameters
    ----------
    array_coordinates: torch.Tensor
        `(..., d)` array of d-dimensional coordinates.
        Coordinates are in the range `[0, N-1]` for the `N` elements in each dimension.
    array_shape: Sequence[int]
        shape of the array being sampled at `array_coordinates`.
    """
    dtype, device = array_coordinates.dtype, array_coordinates.device
    array_shape_tensor = torch.as_tensor(array_shape, dtype=dtype, device=device)
    grid_sample_coordinates = (array_coordinates / (0.5 * array_shape_tensor - 0.5)) - 1
    grid_sample_coordinates = torch.flip(grid_sample_coordinates, dims=(-1,))
    return grid_sample_coordinates


# CIRCLE MASK UTILS
def _add_soft_edge_single_binary_image(
    image: torch.Tensor, smoothing_radius: float
) -> torch.FloatTensor:
    if smoothing_radius == 0:
        return image.float()
    # move explicitly to cpu for scipy
    distances = ndi.distance_transform_edt(torch.logical_not(image).to("cpu"))
    distances = torch.as_tensor(distances, device=image.device).float()
    idx = torch.logical_and(distances > 0, distances <= smoothing_radius)
    output = torch.clone(image).float()
    output[idx] = torch.cos((torch.pi / 2) * (distances[idx] / smoothing_radius))
    return output


def _add_soft_edge_2d(
    image: torch.Tensor, smoothing_radius: torch.Tensor | float
) -> torch.Tensor:
    image_packed, ps = einops.pack([image], "* h w")
    b = image_packed.shape[0]

    if isinstance(smoothing_radius, float | int):
        smoothing_radius = torch.as_tensor(
            data=[smoothing_radius], device=image.device, dtype=torch.float32
        )
    smoothing_radius = torch.broadcast_to(smoothing_radius, (b,))

    results = [
        _add_soft_edge_single_binary_image(_image, smoothing_radius=_smoothing_radius)
        for _image, _smoothing_radius in zip(image_packed, smoothing_radius)
    ]
    results = torch.stack(results, dim=0)
    [results] = einops.unpack(results, pattern="* h w", packed_shapes=ps)
    return results


def circle(
    radius: float,
    image_shape: tuple[int, int] | int,
    center: tuple[float, float] | None = None,
    smoothing_radius: float = 0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Create a circular mask with optional smooth edge."""
    if isinstance(image_shape, int):
        image_shape = (image_shape, image_shape)
    if center is None:
        center = dft_center(image_shape, rfft=False, fftshifted=True)
    distances = coordinate_grid(
        image_shape=image_shape,
        center=center,
        norm=True,
        device=device,
    )
    mask = torch.zeros_like(distances, dtype=torch.bool)
    mask[distances < radius] = 1
    return _add_soft_edge_2d(mask, smoothing_radius=smoothing_radius)
