"""Module for finding shifts between images."""

import torch

from tttsa.correlation import correlate_2d
from tttsa.utils import dft_center


def find_image_shift(
    image_a: torch.Tensor,
    image_b: torch.Tensor,
) -> torch.Tensor:
    """Find the shift between image a and b.

    Applying the shift to b aligns it with
    image a. The region around the maximum in the correlation image is by default
    upsampled with bicubic interpolation to find a more precise shift.

    Parameters
    ----------
    image_a: torch.Tensor
        `(h, w)` image.
    image_b: torch.Tensor
        `(h, w)` image with the same shape as image_a
    mask: torch.Tensor | None, default None
        `(h, w)` mask used for normalization

    Returns
    -------
    shift: torch.Tensor
        `(2, )` shift in y and x, always on CPU
    """
    center = dft_center(image_a.shape, rfft=False, fftshifted=True)

    # calculate initial shift with integer precision
    correlation = correlate_2d(image_a, image_b)
    maximum_idx = torch.unravel_index(correlation.argmax().cpu(), shape=image_a.shape)
    y, x = maximum_idx
    # Ensure that the max index is not on the border
    if (
        y == 0
        or y == correlation.shape[0] - 1
        or x == 0
        or x == correlation.shape[1] - 1
    ):
        return torch.tensor([float(y), float(x)]) - center
    # Parabolic interpolation in the y direction
    f_y0 = correlation[y - 1, x]
    f_y1 = correlation[y, x]
    f_y2 = correlation[y + 1, x]
    subpixel_dy = y + 0.5 * (f_y0 - f_y2) / (f_y0 - 2 * f_y1 + f_y2)

    # Parabolic interpolation in the x direction
    f_x0 = correlation[y, x - 1]
    f_x1 = correlation[y, x]
    f_x2 = correlation[y, x + 1]
    subpixel_dx = x + 0.5 * (f_x0 - f_x2) / (f_x0 - 2 * f_x1 + f_x2)

    shift = torch.tensor([subpixel_dy, subpixel_dx]) - center
    return shift
