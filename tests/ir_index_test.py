from e3nn_little import o3


def test_ir():
    o3.IrList([2, (16, 1), (16, 2, -1), (16, (2, -1)), (16, o3.o2)])


def test_slice():
    Rs = o3.IrList([1, 2, 3, 4])
    assert isinstance(Rs[2:], o3.IrList)
