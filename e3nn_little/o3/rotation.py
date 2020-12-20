# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring, bare-except
import math

import torch


def rot_z(gamma):
    """
    Rotation around Z axis
    """
    c = gamma.cos()
    s = gamma.sin()
    o = torch.ones_like(gamma)
    z = torch.zeros_like(gamma)
    return torch.stack([
        torch.stack([c, -s, z], dim=-1),
        torch.stack([s, c, z], dim=-1),
        torch.stack([z, z, o], dim=-1)
    ], dim=-2)


def rot_y(beta):
    """
    Rotation around Y axis
    """
    c = beta.cos()
    s = beta.sin()
    o = torch.ones_like(beta)
    z = torch.zeros_like(beta)
    return torch.stack([
        torch.stack([c, z, s], dim=-1),
        torch.stack([z, o, z], dim=-1),
        torch.stack([-s, z, c], dim=-1),
    ], dim=-2)


def rot(alpha, beta, gamma):
    """
    ZYZ Euler angles rotation
    """
    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
    return rot_z(alpha) @ rot_y(beta) @ rot_z(gamma)


def rand_rot():
    """
    random rotation matrix
    """
    return rot(*rand_angles())


def rand_angles():
    """
    random rotation angles
    """
    alpha, gamma = 2 * math.pi * torch.rand(2)
    beta = torch.rand(()).mul(2).sub(1).acos()
    return alpha, beta, gamma


def angles_to_xyz(alpha, beta):
    """
    Convert (alpha, beta) into point (x, y, z) on the sphere
    """
    alpha, beta = torch.broadcast_tensors(alpha, beta)
    x = torch.sin(beta) * torch.cos(alpha)
    y = torch.sin(beta) * torch.sin(alpha)
    z = torch.cos(beta)
    return torch.stack([x, y, z], dim=-1)


def xyz_to_angles(xyz):
    """
    Convert point (x, y, z) on the sphere into (alpha, beta)
    """
    xyz = torch.nn.functional.normalize(xyz, p=2, dim=-1)  # forward 0's instead of nan for zero-radius
    xyz = xyz.clamp(-1, 1)

    beta = torch.acos(xyz[..., 2])
    alpha = torch.atan2(xyz[..., 1], xyz[..., 0])
    return alpha, beta


def rot_to_angles(R):
    """
    Convert rotation matrix into (alpha, beta, gamma)
    """
    x = R @ R.new_tensor([0, 0, 1])
    a, b = xyz_to_angles(x)
    R = rot(a, b, a.new_zeros(a.shape)).transpose(-1, -2) @ R
    c = torch.atan2(R[..., 1, 0], R[..., 0, 0])
    return a, b, c


def compose_angles(a1, b1, c1, a2, b2, c2):
    """
    (a, b, c) = (a1, b1, c1) composed with (a2, b2, c2)
    """
    a1, b1, c1, a2, b2, c2 = torch.broadcast_tensors(a1, b1, c1, a2, b2, c2)
    comp = rot(a1, b1, c1) @ rot(a2, b2, c2)
    xyz = comp @ torch.tensor([0, 0, 1.0])
    a, b = xyz_to_angles(xyz)
    rotz = rot(torch.tensor(0.0), -b, -a) @ comp
    c = torch.atan2(rotz[..., 1, 0], rotz[..., 0, 0])
    return a, b, c


def angles_to_angle(alpha, beta, gamma):
    R = rot(alpha, beta, gamma)
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    return torch.acos(tr.sub(1).div(2).clamp(-1, 1))


def axis_angle_to_quaternion(xyz, angle):
    xyz, angle = torch.broadcast_tensors(xyz, angle[..., None])
    c = torch.cos(angle[..., :1] / 2)
    s = torch.sin(angle / 2)
    return torch.cat([c, xyz * s], dim=-1)


def compose_quaternion(q1, q2):
    """q1 o q2
    """
    q1, q2 = torch.broadcast_tensors(q1, q2)
    return torch.stack([
        q1[..., 0] * q2[..., 0] - q1[..., 1] * q2[..., 1] - q1[..., 2] * q2[..., 2] - q1[..., 3] * q2[..., 3],
        q1[..., 1] * q2[..., 0] + q1[..., 0] * q2[..., 1] + q1[..., 2] * q2[..., 3] - q1[..., 3] * q2[..., 2],
        q1[..., 0] * q2[..., 2] - q1[..., 1] * q2[..., 3] + q1[..., 2] * q2[..., 0] + q1[..., 3] * q2[..., 1],
        q1[..., 0] * q2[..., 3] + q1[..., 1] * q2[..., 2] - q1[..., 2] * q2[..., 1] + q1[..., 3] * q2[..., 0],
    ], dim=-1)


def angles_to_quaternion(alpha, beta, gamma):
    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
    qa = axis_angle_to_quaternion(torch.tensor([0, 0, 1.0]), alpha)
    qb = axis_angle_to_quaternion(torch.tensor([0, 1, 0.0]), beta)
    qc = axis_angle_to_quaternion(torch.tensor([0, 0, 1.0]), gamma)
    return compose_quaternion(qa, compose_quaternion(qb, qc))
