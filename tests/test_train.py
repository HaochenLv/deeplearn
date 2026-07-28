import torch
from data import Tokenizer, build_length_examples, AdditionDataset, make_collate_fn
from model import AdditionTransformer
from train import compute_loss, teacher_forced_accuracy, exact_match_accuracy, train_stage


def test_compute_loss_finite_and_only_target():
    tok = Tokenizer()
    m = AdditionTransformer(tok.vocab_size, d_model=32, n_heads=4, n_layers=1)
    ds = AdditionDataset(tok, build_length_examples(1, 8, seed=0))
    input_ids, labels, pad_mask = make_collate_fn(tok)([ds[i] for i in range(8)])
    loss, logits = compute_loss(m, input_ids, labels, pad_mask)
    assert torch.isfinite(loss)
    assert logits.shape[:2] == input_ids.shape
    assert loss.item() > 0


def test_teacher_forced_accuracy_in_range():
    tok = Tokenizer()
    m = AdditionTransformer(tok.vocab_size, d_model=32, n_heads=4, n_layers=1)
    ds = AdditionDataset(tok, build_length_examples(1, 16, seed=0))
    acc = teacher_forced_accuracy(m, ds, tok, device=torch.device("cpu"), batch_size=8)
    assert 0.0 <= acc <= 1.0


def test_overfit_tiny_batch():
    """健全性检查：模型能在 ~8 个样本上过拟合（loss 显著下降）。"""
    torch.manual_seed(0)
    tok = Tokenizer()
    m = AdditionTransformer(tok.vocab_size, d_model=64, n_heads=4, n_layers=2)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    ds = AdditionDataset(tok, build_length_examples(1, 8, seed=42))
    collate = make_collate_fn(tok)
    batch = collate([ds[i] for i in range(8)])
    first = None
    for _ in range(300):
        loss, _ = compute_loss(m, *batch)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if first is None:
            first = loss.item()
    assert loss.item() < first * 0.1, (first, loss.item())


def test_exact_match_accuracy_in_range():
    tok = Tokenizer()
    m = AdditionTransformer(tok.vocab_size, d_model=32, n_heads=4, n_layers=1)
    examples = build_length_examples(1, 8, seed=0)
    acc = exact_match_accuracy(m, examples, tok, device=torch.device("cpu"))
    assert 0.0 <= acc <= 1.0


def test_train_stage_graduates_on_low_em_threshold():
    # 毕业判据改为自回归 exact-match；门槛 0.0 → 第一个 epoch 即毕业
    tok = Tokenizer()
    m = AdditionTransformer(tok.vocab_size, d_model=32, n_heads=4, n_layers=1)
    hist = train_stage(m, tok, 1, torch.device("cpu"),
                       n_examples=200, max_epochs=3, grad_threshold_em=0.0,
                       n_val_em=8, probe_lengths=(1,))
    assert isinstance(hist, list) and len(hist) >= 1
    assert len(hist[0]) == 4                      # (epoch, em_acc, probe, mean_loss)
    assert hist[0][1] >= 0.0                      # em_acc 在 [0,1]
