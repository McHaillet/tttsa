"""Example of tttsa.tilt_series_alignment() usage."""

from pathlib import Path

import einops
import mrcfile
import numpy as np
import pooch
import torch
from torch_fourier_rescale import fourier_rescale_2d
from torch_subpixel_crop import subpixel_crop_2d

from tttsa import tilt_series_alignment
from tttsa.back_projection import filtered_back_projection_3d
from tttsa.utils import dft_center

# https://github.com/fatiando/pooch
GOODBOY = pooch.create(
    path=pooch.os_cache("tttsa"),
    base_url="doi:10.5281/zenodo.14052854/",
    registry={
        "tomo200528_107.st": "md5:cb07c6962d176150db8e5bcdb7b3f27b",
        "tomo200528_107.rawtlt": "md5:e7a959877356a8241d024124228b0911",
    },
)

IMAGE_FILE = Path(GOODBOY.fetch("tomo200528_107.st", progressbar=True))
with open(Path(GOODBOY.fetch("tomo200528_107.rawtlt"))) as f:
    STAGE_TILT_ANGLE_PRIORS = torch.tensor([float(x) for x in f.readlines()])
IMAGE_PIXEL_SIZE = 1.724
# STAGE_TILT_ANGLE_PRIORS = torch.arange(-51, 54, 3)  # 107: 54, 100: 51
TILT_AXIS_ANGLE_PRIOR = -88.7  # -88.7 according to mdoc, but faulty to test
# this angle is assumed to be a clockwise forward rotation after projecting the sample
ALIGNMENT_PIXEL_SIZE = IMAGE_PIXEL_SIZE * 8
ALIGN_Z = int(1600 / ALIGNMENT_PIXEL_SIZE)  # number is in A
RECON_Z = int(2400 / ALIGNMENT_PIXEL_SIZE)
WEIGHTING = "hamming"  # weighting scheme for filtered back projection
# the object diameter in number of pixels
OBJECT_DIAMETER = 300 / ALIGNMENT_PIXEL_SIZE
OUTPUT_DIR = Path(__file__).parent.resolve().joinpath("data")


tilt_series = torch.as_tensor(mrcfile.read(IMAGE_FILE))

tilt_series, _ = fourier_rescale_2d(  # should normalize beforehand
    image=tilt_series,
    source_spacing=IMAGE_PIXEL_SIZE,
    target_spacing=ALIGNMENT_PIXEL_SIZE,
)

# Temp workaround to ensure square images
tilt_series = tilt_series[:, tilt_series.shape[-2] % 2 :, tilt_series.shape[-1] % 2 :]

# Ensure normalization after fourier rescale
tilt_series -= einops.reduce(tilt_series, "tilt h w -> tilt 1 1", reduction="mean")
tilt_series /= torch.std(tilt_series, dim=(-2, -1), keepdim=True)

n_tilts, h, w = tilt_series.shape
center = dft_center((h, w), rfft=False, fftshifted=True)
center = einops.repeat(center, "yx -> b yx", b=n_tilts)
tilt_series = subpixel_crop_2d(  # torch-subpixel-crop
    image=tilt_series,
    positions=center,
    sidelength=min(h, w),
)
_, h, w = tilt_series.shape
size = min(h, w)

tilt_angles, tilt_axis_angles, shifts = tilt_series_alignment(
    tilt_series,
    STAGE_TILT_ANGLE_PRIORS,
    TILT_AXIS_ANGLE_PRIOR,
    ALIGN_Z,
    find_tilt_angle_offset=False,
)

final, aligned_ts = filtered_back_projection_3d(
    tilt_series,
    (RECON_Z, size, size),
    tilt_angles,
    tilt_axis_angles,
    shifts,
    weighting=WEIGHTING,
    object_diameter=OBJECT_DIAMETER,
)

OUTPUT_DIR.mkdir(exist_ok=True)
mrcfile.write(
    OUTPUT_DIR.joinpath(IMAGE_FILE.with_suffix(".mrc").name),
    final.detach().numpy().astype(np.float32),
    voxel_size=ALIGNMENT_PIXEL_SIZE,
    overwrite=True,
)
