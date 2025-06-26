"""Example of tttsa.tilt_series_alignment() usage."""

from pathlib import Path

import einops
import mrcfile
import numpy as np
import pooch
import torch
from cryotypes.projectionmodel import ProjectionModel
from cryotypes.projectionmodel import ProjectionModelDataLabels as PMDL
from torch_fourier_rescale import fourier_rescale_2d
from torch_subpixel_crop import subpixel_crop_2d
from torch_tomogram import Tomogram

from tttsa import tilt_series_alignment
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
    STAGE_TILT_ANGLE_PRIORS = [float(x) for x in f.readlines()]
IMAGE_PIXEL_SIZE = 1.724
# this angle is assumed to be a clockwise forward rotation after projecting the sample
TILT_AXIS_ANGLE_PRIOR = -88.7
ALIGNMENT_PIXEL_SIZE = IMAGE_PIXEL_SIZE * 8
ALIGN_Z = int(1600 / ALIGNMENT_PIXEL_SIZE)  # number is in A
RECON_Z = int(2400 / ALIGNMENT_PIXEL_SIZE)
WEIGHTING = "hamming"  # weighting scheme for filtered back projection
# the object diameter in number of pixels
OBJECT_DIAMETER = 300 / ALIGNMENT_PIXEL_SIZE
OUTPUT_DIR = Path(__file__).parent.resolve().joinpath("data")

# Set the device for running
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize the projection-model prior
projection_model_prior = ProjectionModel(
    {
        PMDL.ROTATION_Z: TILT_AXIS_ANGLE_PRIOR,
        PMDL.ROTATION_Y: STAGE_TILT_ANGLE_PRIORS,
        PMDL.ROTATION_X: 0.0,
        PMDL.SHIFT_X: 0.0,
        PMDL.SHIFT_Y: 0.0,
        PMDL.EXPERIMENT_ID: IMAGE_FILE.stem,
        PMDL.PIXEL_SPACING: ALIGNMENT_PIXEL_SIZE,
        PMDL.SOURCE: IMAGE_FILE.name,
    }
)

tilt_series = torch.as_tensor(mrcfile.read(IMAGE_FILE))

tilt_series, _ = fourier_rescale_2d(  # should normalize beforehand
    image=tilt_series,
    source_spacing=IMAGE_PIXEL_SIZE,
    target_spacing=ALIGNMENT_PIXEL_SIZE,
)

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

# Move all the input to the device
projection_model_optimized = tilt_series_alignment(
    tilt_series.to(DEVICE),
    projection_model_prior,
    ALIGN_Z,
    find_tilt_angle_offset=True,
)

tomogram = Tomogram(
    tilt_angles=projection_model_optimized[PMDL.ROTATION_Y],
    tilt_axis_angle=projection_model_optimized[PMDL.ROTATION_Z],
    sample_translations=projection_model_optimized[PMDL.SHIFT].to_numpy(),
    images=tilt_series.to("cpu"),
)
# let's make the reconstruction a bit more beautiful ;)
tomogram._pad_factor = 3.0
final = tomogram.reconstruct_tomogram((RECON_Z, size, size), 128)

OUTPUT_DIR.mkdir(exist_ok=True)
mrcfile.write(
    OUTPUT_DIR.joinpath(IMAGE_FILE.with_suffix(".mrc").name),
    final.detach().numpy().astype(np.float32),
    voxel_size=ALIGNMENT_PIXEL_SIZE,
    overwrite=True,
)
