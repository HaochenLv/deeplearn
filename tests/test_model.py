import torch
from data import Tokenizer
from model import AdditionTransformer


def _make(vocab_size):
    return AdditionTransformer(vocab_size, d_model=32, n_heads=4, n_layers=1, max_len=64)


def test_output_shape():
    tok = Tokenizer()
    m = _make(tok.vocab_size)
    idx = torch.randint(0, tok.vocab_size, (4, 10))
    logits = m(idx)
    assert logits.shape == (4, 10, tok.vocab_size)


def test_no_nan():
    tok = Tokenizer()
    m = _make(tok.vocab_size)
    idx = torch.randint(0, tok.vocab_size, (2, 8))
    logits = m(idx)
    assert torch.isfinite(logits).all()


def test_pad_mask_allows_attention():
    tok = Tokenizer()
    m = _make(tok.vocab_size)
    idx = torch.tensor([[tok.bos_id] + tok.encode("3+4=") + [tok.pad_id] * 5])
    pm = (idx == tok.pad_id)
    logits = m(idx, pm)
    assert torch.isfinite(logits).all()


def test_param_count_small():
    tok = Tokenizer()
    m = AdditionTransformer(tok.vocab_size, d_model=128, n_heads=4, n_layers=2, max_len=256)
    n = sum(p.numel() for p in m.parameters())
    assert n < 2_000_000  # 微型：小于 200 万


def test_broadcasts_for_variable_batch():
    import torch
    from data import Tokenizer
    from model import AdditionTransformer
    tok = Tokenizer()
    m = AdditionTransformer(tok.vocab_size, d_model=32, n_heads=4, n_layers=1, max_len=64)
    for B in [1, 3, 8, 64]:
        idx = torch.randint(0, tok.vocab_size, (B, 10))
        logits = m(idx)
        assert logits.shape == (B, 10, tok.vocab_size), f"failed at B={B}"
        assert torch.isfinite(logits).all()
