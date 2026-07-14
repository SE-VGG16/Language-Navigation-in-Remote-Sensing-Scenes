from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def metric_sampling_theta(
    altitude_m: Tensor,
    heading_deg: Tensor,
    image_height: int,
    image_width: int,
    field_of_view_m: float,
    focal_length_px: float,
    canonicalize_heading: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build affine-grid matrices for metric crop and heading canonicalization.

    Args:
        altitude_m: Shape [B], altitude above the reference plane in meters.
        heading_deg: Shape [B], clockwise heading from north in degrees.
        image_height: Input image height in pixels.
        image_width: Input image width in pixels.
        field_of_view_m: Requested square ground footprint side in meters.
        focal_length_px: Camera focal length in pixels.
        canonicalize_heading: Whether to rotate the output to north-up.

    Returns:
        theta: [B, 2, 3], output-to-input affine matrices for affine_grid.
        coverage: [B], realized/requested footprint ratio in [0, 1].
        crop_pixels: [B], realized square crop side in input pixels.
    """
    if altitude_m.ndim != 1 or heading_deg.ndim != 1:
        raise ValueError("altitude_m and heading_deg must have shape [B].")
    if torch.any(altitude_m <= 0):
        raise ValueError("All altitude values must be positive.")

    requested_crop = field_of_view_m * focal_length_px / altitude_m
    max_square = float(min(image_height, image_width))
    crop_pixels = requested_crop.clamp(max=max_square)
    coverage = (crop_pixels / requested_crop.clamp_min(1e-6)).clamp(0.0, 1.0)

    sx = crop_pixels / float(image_width)
    sy = crop_pixels / float(image_height)

    angle = torch.deg2rad(heading_deg if canonicalize_heading else torch.zeros_like(heading_deg))
    # theta maps canonical output coordinates to the source image. With heading
    # measured clockwise from north, this convention samples the rotated source
    # to produce a north-up output.
    c = torch.cos(angle)
    s = torch.sin(angle)

    theta = torch.zeros(
        altitude_m.shape[0], 2, 3, dtype=altitude_m.dtype, device=altitude_m.device
    )
    theta[:, 0, 0] = sx * c
    theta[:, 0, 1] = -sx * s
    theta[:, 1, 0] = sy * s
    theta[:, 1, 1] = sy * c
    return theta, coverage, crop_pixels


def warp_perspective_normalized(
    feature_map: Tensor,
    homography: Tensor,
    output_size: tuple[int, int] | None = None,
    align_corners: bool = False,
) -> Tensor:
    """Warp a feature map using a homography in normalized coordinates.

    `homography` maps output normalized coordinates to input normalized
    coordinates. Coordinates use [-1, 1] in x and y.
    """
    if feature_map.ndim != 4:
        raise ValueError("feature_map must have shape [B, C, H, W].")
    if homography.ndim != 3 or homography.shape[-2:] != (3, 3):
        raise ValueError("homography must have shape [B, 3, 3].")

    b, _, h, w = feature_map.shape
    out_h, out_w = output_size or (h, w)
    ys = torch.linspace(-1.0, 1.0, out_h, device=feature_map.device, dtype=feature_map.dtype)
    xs = torch.linspace(-1.0, 1.0, out_w, device=feature_map.device, dtype=feature_map.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    ones = torch.ones_like(xx)
    base = torch.stack([xx, yy, ones], dim=-1).reshape(1, out_h * out_w, 3)
    base = base.expand(b, -1, -1)

    src = torch.bmm(base, homography.transpose(1, 2))
    denom = src[..., 2:3].clamp_min(1e-6)
    src_xy = src[..., :2] / denom
    grid = src_xy.reshape(b, out_h, out_w, 2)
    return F.grid_sample(
        feature_map,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=align_corners,
    )


def metric_cell_coordinates(
    grid_size: int,
    field_of_view_m: float,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Return [G*G, 2] cell centers in x=east, y=north meters."""
    cell = field_of_view_m / grid_size
    idx = torch.arange(grid_size, device=device, dtype=dtype)
    x = (idx + 0.5 - grid_size / 2.0) * cell
    y = (grid_size / 2.0 - idx - 0.5) * cell
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([xx, yy], dim=-1).reshape(-1, 2)


def rotation_rectification_homography(
    pitch_deg: Tensor,
    roll_deg: Tensor,
    focal_normalized: float = 2.0,
) -> Tensor:
    """Approximate pure-rotation rectification homography.

    This utility compensates camera pitch and roll around the optical center.
    It does not reconstruct non-planar terrain or object-specific parallax.
    The returned matrix maps rectified output coordinates to source coordinates.
    """
    if pitch_deg.shape != roll_deg.shape:
        raise ValueError("pitch_deg and roll_deg must have the same shape.")
    p = torch.deg2rad(pitch_deg)
    r = torch.deg2rad(roll_deg)
    b = p.numel()
    device, dtype = p.device, p.dtype

    cp, sp = torch.cos(p), torch.sin(p)
    cr, sr = torch.cos(r), torch.sin(r)

    rx = torch.zeros(b, 3, 3, device=device, dtype=dtype)
    rx[:, 0, 0] = 1
    rx[:, 1, 1] = cr
    rx[:, 1, 2] = -sr
    rx[:, 2, 1] = sr
    rx[:, 2, 2] = cr

    ry = torch.zeros(b, 3, 3, device=device, dtype=dtype)
    ry[:, 0, 0] = cp
    ry[:, 0, 2] = sp
    ry[:, 1, 1] = 1
    ry[:, 2, 0] = -sp
    ry[:, 2, 2] = cp

    rot = torch.bmm(ry, rx)
    k = torch.tensor(
        [[focal_normalized, 0.0, 0.0], [0.0, focal_normalized, 0.0], [0.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    ).expand(b, -1, -1)
    kinv = torch.linalg.inv(k)
    return torch.bmm(torch.bmm(k, rot), kinv)
