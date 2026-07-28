import torch
from mlp_addition import (
    make_expr, encode_input, encode_target, decode_digits, MLP, dims,
)


def test_make_expr():
    assert make_expr(5, 37, 2) == "05+37"
    assert make_expr(99, 99, 2) == "99+99"


def test_encode_input_shape_and_sparse():
    expr_len, in_dim, _ = dims(2)
    v = encode_input(5, 37, 2)
    assert v.shape == (in_dim,)
    assert v.sum().item() == expr_len  # 每字符恰好一个 1 → 5 个 1


def test_encode_target():
    assert encode_target(85, 2).tolist() == [0, 8, 5]
    assert encode_target(198, 2).tolist() == [1, 9, 8]
    assert encode_target(7, 2).tolist() == [0, 0, 7]


def test_decode_digits():
    assert decode_digits([0, 8, 5]) == 85
    assert decode_digits([1, 9, 8]) == 198


def test_mlp_forward_shape():
    _, _, out_digits = dims(2)
    m = MLP(2)
    x = torch.stack([encode_input(5, 37, 2), encode_input(99, 1, 2)])
    out = m(x)
    assert out.shape == (2, out_digits, 10)
