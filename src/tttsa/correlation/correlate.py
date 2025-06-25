"""DFT based cross-correlation between images."""

import torch
import torch.nn.functional as F


def correlate_2d(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Calculate the 2D cross correlation between images of the same size.

    The position of the maximum relative to the center of the image gives a shift.
    This is the shift that when applied to `b` best aligns it to `a`.
    """
    # then pad with 2x
    p = int(0.5 * min(a.shape[-2:]))
    a = F.pad(a, [p] * 4, value=0)
    b = F.pad(b, [p] * 4, value=0)

    # do the correlation
    h, w = a.shape[-2:]
    fta = torch.fft.rfftn(a, dim=(-2, -1))
    ftb = torch.fft.rfftn(b, dim=(-2, -1))
    result = fta * torch.conj(ftb)
    # AreTomo using some like this (filtered FFT-based approach):
    # result = result / torch.sqrt(result.abs() + .0001)
    # result = bfactor_dft(result, 300, (result.shape[-2], ) * 2, 1, True)
    result = torch.fft.irfftn(result, dim=(-2, -1), s=a.shape[-2:])
    result = torch.real(torch.fft.ifftshift(result, dim=(-2, -1)))
    result = result / (h * w)  # normalize the value range

    # unpad
    result = F.pad(result, [-p] * 4)

    return result
