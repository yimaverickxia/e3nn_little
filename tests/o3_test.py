import torch

from e3nn_little import o3


def test_rot():
    torch.set_default_dtype(torch.float64)

    R = o3.rand_rot()
    assert torch.allclose(R @ R.T, torch.eye(3))

    a, b, c = o3.rot_to_abc(R)
    pos1 = o3.angles_to_xyz(a, b)
    pos2 = R @ torch.tensor([0, 0, 1.0])
    assert torch.allclose(pos1, pos2)


def test_wigner_3j():
    assert torch.allclose(o3.wigner_3j(1, 2, 3), o3.wigner_3j(1, 3, 2).transpose(1, 2))
    assert torch.allclose(o3.wigner_3j(1, 2, 3), o3.wigner_3j(2, 1, 3).transpose(0, 1))
    assert torch.allclose(o3.wigner_3j(1, 2, 3), o3.wigner_3j(3, 2, 1).transpose(0, 2))
    assert torch.allclose(o3.wigner_3j(1, 2, 3), o3.wigner_3j(3, 1, 2).transpose(0, 1).transpose(1, 2))
    assert torch.allclose(o3.wigner_3j(1, 2, 3), o3.wigner_3j(2, 3, 1).transpose(0, 2).transpose(1, 2))
