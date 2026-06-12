import torch

from torch_inr import get_coords, get_input_shape, get_target


def test_get_coords_shape_and_range():
    coords = get_coords((8, 8))
    assert coords.shape == (64, 2)
    assert coords.min() >= -1 and coords.max() < 1


def test_get_coords_1d_keeps_column():
    coords = get_coords((10,))
    assert coords.shape == (10, 1)


def test_get_coords_excludes_predict_dims():
    coords = get_coords((8, 8, 3), predict_dims=(2,))
    assert coords.shape == (64, 2)


def test_get_input_shape():
    assert get_input_shape((4, 5, 3), ()) == [4, 5, 3]
    assert get_input_shape((4, 5, 3), (2,)) == [4, 5]
    assert get_input_shape((4, 5, 3), (0, 2)) == [5]


def test_get_target_matches_coord_order():
    X = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    target = get_target(X, ())
    assert target.shape == (12, 1)
    # row-major flattening: target[i * 4 + j] == X[i, j]
    assert torch.equal(target.flatten(), X.flatten())


def test_get_target_predict_dims():
    X = torch.rand(3, 4, 2)
    target = get_target(X, (2,))
    assert target.shape == (12, 2)
    assert torch.equal(target[5], X[1, 1])  # flat index 5 -> (i=1, j=1)


def test_get_target_permutes_leading_predict_dim():
    X = torch.rand(2, 3, 4)
    target = get_target(X, (0,))
    assert target.shape == (12, 2)
    assert torch.equal(target[0], X[:, 0, 0])
