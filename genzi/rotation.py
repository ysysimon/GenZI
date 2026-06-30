"""Small torch rotation helpers used by SMPL-X optimization."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle rotations to rotation matrices."""

    if axis_angle.shape[-1] != 3:
        raise ValueError(f"axis_angle must have shape (..., 3), got {axis_angle.shape}")

    angle = torch.linalg.norm(axis_angle, dim=-1, keepdim=True)
    eps = torch.finfo(axis_angle.dtype).eps
    axis = axis_angle / torch.clamp(angle, min=eps)
    x, y, z = axis.unbind(dim=-1)

    angle = angle.squeeze(-1)
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one_minus_cos = 1 - cos

    row0 = torch.stack(
        (
            cos + x * x * one_minus_cos,
            x * y * one_minus_cos - z * sin,
            x * z * one_minus_cos + y * sin,
        ),
        dim=-1,
    )
    row1 = torch.stack(
        (
            y * x * one_minus_cos + z * sin,
            cos + y * y * one_minus_cos,
            y * z * one_minus_cos - x * sin,
        ),
        dim=-1,
    )
    row2 = torch.stack(
        (
            z * x * one_minus_cos - y * sin,
            z * y * one_minus_cos + x * sin,
            cos + z * z * one_minus_cos,
        ),
        dim=-1,
    )
    matrix = torch.stack((row0, row1, row2), dim=-2)

    identity = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device)
    identity = identity.expand(axis_angle.shape[:-1] + (3, 3))
    small = (angle.abs() < 1e-8).view(axis_angle.shape[:-1] + (1, 1))
    return torch.where(small, identity + _skew(axis_angle), matrix)


def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to axis-angle rotations."""

    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"matrix must have shape (..., 3, 3), got {matrix.shape}")

    return _quaternion_to_axis_angle(_matrix_to_quaternion(matrix))


def _skew(vec: torch.Tensor) -> torch.Tensor:
    x, y, z = vec.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    return torch.stack(
        (
            torch.stack((zeros, -z, y), dim=-1),
            torch.stack((z, zeros, -x), dim=-1),
            torch.stack((-y, x, zeros), dim=-1),
        ),
        dim=-2,
    )


def _matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    m00 = matrix[..., 0, 0]
    m01 = matrix[..., 0, 1]
    m02 = matrix[..., 0, 2]
    m10 = matrix[..., 1, 0]
    m11 = matrix[..., 1, 1]
    m12 = matrix[..., 1, 2]
    m20 = matrix[..., 2, 0]
    m21 = matrix[..., 2, 1]
    m22 = matrix[..., 2, 2]

    eps = torch.finfo(matrix.dtype).eps
    q_abs = torch.sqrt(
        torch.clamp(
            torch.stack(
                (
                    1 + m00 + m11 + m22,
                    1 + m00 - m11 - m22,
                    1 - m00 + m11 - m22,
                    1 - m00 - m11 + m22,
                ),
                dim=-1,
            ),
            min=eps,
        )
    )

    quat_by_case = torch.stack(
        (
            torch.stack((q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01), dim=-1),
            torch.stack((m21 - m12, q_abs[..., 1] ** 2, m01 + m10, m02 + m20), dim=-1),
            torch.stack((m02 - m20, m01 + m10, q_abs[..., 2] ** 2, m12 + m21), dim=-1),
            torch.stack((m10 - m01, m02 + m20, m12 + m21, q_abs[..., 3] ** 2), dim=-1),
        ),
        dim=-2,
    )
    quat_candidates = quat_by_case / (2 * q_abs.clamp(min=0.1))[..., None]
    best = q_abs.argmax(dim=-1)
    gather_idx = best[..., None, None].expand(best.shape + (1, 4))
    quat = torch.gather(quat_candidates, dim=-2, index=gather_idx).squeeze(-2)
    quat = F.normalize(quat, dim=-1)
    return torch.where(quat[..., :1] < 0, -quat, quat)


def _quaternion_to_axis_angle(quaternion: torch.Tensor) -> torch.Tensor:
    vector = quaternion[..., 1:]
    norms = torch.linalg.norm(vector, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternion[..., :1])
    angles = 2 * half_angles
    small_angles = angles.abs() < 1e-6
    safe_angles = torch.where(small_angles, torch.ones_like(angles), angles)
    sin_half_over_angle = torch.where(
        small_angles,
        0.5 - (angles * angles) / 48,
        torch.sin(half_angles) / safe_angles,
    )
    return vector / sin_half_over_angle
