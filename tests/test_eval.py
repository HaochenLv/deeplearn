from data import Tokenizer, make_target
from eval import extract_answer, greedy_decode


def test_extract_answer_basic():
    text = "37+48=\n7+8+0=15 d5 c1\n3+4+1=8 d8 c0\n=85"
    assert extract_answer(text) == "85"


def test_extract_answer_none_when_missing():
    assert extract_answer("no answer here") is None


def test_greedy_decode_returns_string():
    # 不校验正确性（未训练），只校验接口与可运行
    import torch
    from model import AdditionTransformer
    tok = Tokenizer()
    m = AdditionTransformer(tok.vocab_size, d_model=32, n_heads=4, n_layers=1)
    out = greedy_decode(m, tok, "3+4=", device=torch.device("cpu"), max_new_tokens=20)
    assert isinstance(out, str)


def test_column_accuracy_perfect():
    from eval import column_accuracy
    text = "7+8+0=15 d5 c1\n3+4+1=8 d8 c0\n=85"
    assert column_accuracy(text, 37, 48) == 1.0


def test_column_accuracy_partial():
    from eval import column_accuracy
    # 第二列 digit 写错（d9 应为 d8）→ 2 列只对 1 列
    text = "7+8+0=15 d5 c1\n3+4+1=8 d9 c0\n=85"
    assert column_accuracy(text, 37, 48) == 0.5
