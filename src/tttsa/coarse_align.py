"""Coarse tilt-series alignment functions, also with stretching."""

import torch

from .affine import stretch_image
from .alignment import find_image_shift


def coarse_align(
    tilt_series: torch.Tensor,
    reference_tilt_id: int,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Find coarse shifts of images without stretching along tilt axis."""
    shifts = torch.zeros((len(tilt_series), 2), dtype=torch.float32)
    # find coarse alignment for negative tilts
    current_shift = torch.zeros(2)
    for i in range(reference_tilt_id, 0, -1):
        shift = find_image_shift(tilt_series[i] * mask, tilt_series[i - 1] * mask)
        current_shift += shift
        shifts[i - 1] = current_shift

    # find coarse alignment positive tilts
    current_shift = torch.zeros(2)
    for i in range(reference_tilt_id, tilt_series.shape[0] - 1, 1):
        shift = find_image_shift(
            tilt_series[i] * mask,
            tilt_series[i + 1] * mask,
        )
        current_shift += shift
        shifts[i + 1] = current_shift
    return shifts


def stretch_align(
    tilt_series: torch.Tensor,
    reference_tilt_id: int,
    mask: torch.Tensor,
    tilt_angles: torch.Tensor,
    tilt_axis_angles: torch.Tensor,
) -> torch.Tensor:
    """Find coarse shifts of images while stretching each pair along the tilt axis."""
    shifts = torch.zeros((len(tilt_series), 2), dtype=torch.float32)
    # find coarse alignment for negative tilts
    current_shift = torch.zeros(2)
    for i in range(reference_tilt_id, 0, -1):
        scale_factor = torch.cos(torch.deg2rad(tilt_angles[i : i + 1])) / torch.cos(
            torch.deg2rad(tilt_angles[i - 1 : i])
        )
        stretched = (
            stretch_image(
                tilt_series[i - 1],
                scale_factor,
                tilt_axis_angles[i - 1],
            )
            * mask
        )
        stretched = (stretched - stretched.mean()) / stretched.std()
        raw = tilt_series[i] * mask
        raw = (raw - raw.mean()) / raw.std()
        shift = find_image_shift(raw, stretched)
        current_shift += shift
        shifts[i - 1] = current_shift
    # find coarse alignment positive tilts
    current_shift = torch.zeros(2)
    for i in range(reference_tilt_id, tilt_series.shape[0] - 1, 1):
        scale_factor = torch.cos(torch.deg2rad(tilt_angles[i : i + 1])) / torch.cos(
            torch.deg2rad(tilt_angles[i + 1 : i + 2])
        )
        stretched = (
            stretch_image(
                tilt_series[i + 1],
                scale_factor,
                tilt_axis_angles[i + 1],
            )
            * mask
        )
        stretched = (stretched - stretched.mean()) / stretched.std()
        raw = tilt_series[i] * mask
        raw = (raw - raw.mean()) / raw.std()
        shift = find_image_shift(raw, stretched)
        current_shift += shift
        shifts[i + 1] = current_shift
    return shifts
