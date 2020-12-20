import torch

from e3nn_little import o3


def test_rot():
    torch.set_default_dtype(torch.float64)

    R = o3.rand_rot()
    assert torch.allclose(R @ R.T, torch.eye(3))

    a, b, c = o3.rot_to_angles(R)
    pos1 = o3.angles_to_xyz(a, b)
    pos2 = R @ torch.tensor([0, 0, 1.0])
    assert torch.allclose(pos1, pos2)


def test_wigner_3j():
    assert torch.allclose(o3.wigner_3j(1, 2, 3), o3.wigner_3j(1, 3, 2).transpose(1, 2))
    assert torch.allclose(o3.wigner_3j(1, 2, 3), o3.wigner_3j(2, 1, 3).transpose(0, 1))
    assert torch.allclose(o3.wigner_3j(1, 2, 3), o3.wigner_3j(3, 2, 1).transpose(0, 2))
    assert torch.allclose(o3.wigner_3j(1, 2, 3), o3.wigner_3j(3, 1, 2).transpose(0, 1).transpose(1, 2))
    assert torch.allclose(o3.wigner_3j(1, 2, 3), o3.wigner_3j(2, 3, 1).transpose(0, 2).transpose(1, 2))


def test_quaternion():
    torch.set_default_dtype(torch.float64)
    abc1 = o3.rand_angles()
    abc2 = o3.rand_angles()
    q1 = o3.angles_to_quaternion(*abc1)
    q2 = o3.angles_to_quaternion(*abc2)

    abc = o3.compose_angles(*abc1, *abc2)
    q = o3.compose_quaternion(q1, q2)

    qq = o3.angles_to_quaternion(*abc)
    assert min((q - qq).abs().max(), (q + qq).abs().max()) < 1e-10
