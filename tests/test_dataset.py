import torch
from data import Tokenizer, build_length_examples, build_curriculum_examples, AdditionDataset, make_collate_fn


def test_length_examples_count_and_range():
    exs = build_length_examples(2, 50, seed=1)
    assert len(exs) == 50
    for a, b, d in exs:
        assert d == 2 and 0 <= a <= 99 and 0 <= b <= 99


def test_curriculum_includes_old_lengths():
    exs = build_curriculum_examples(3, 100, seed=2)
    ds = {e[2] for e in exs}
    assert 3 in ds and (1 in ds or 2 in ds)


def test_collate_masks_prompt_and_pad():
    tok = Tokenizer()
    exs = build_length_examples(1, 4, seed=3)
    ds = AdditionDataset(tok, exs)
    collate = make_collate_fn(tok)
    input_ids, labels, pad_mask = collate([ds[i] for i in range(4)])
    assert input_ids.dtype == torch.long and labels.dtype == torch.long
    # 标签里没有出现在"prompt 区"或 pad 上的非 -100 之外的非法情况：
    # 即所有 != -100 的 label 都在 [0, vocab_size)
    valid = labels[labels != -100]
    assert (valid >= 0).all() and (valid < tok.vocab_size).all()
    # 第 0 个样本第一个 label（对应 prompt 第一位）应为 -100
    assert labels[0, 0].item() == -100
    # 每个 sample 都包含 eos（被作为目标预测）
    assert (labels == tok.eos_id).any()


def test_collate_pad_dimensions():
    tok = Tokenizer()
    exs = build_length_examples(2, 8, seed=4)
    ds = AdditionDataset(tok, exs)
    collate = make_collate_fn(tok)
    input_ids, labels, pad_mask = collate([ds[i] for i in range(8)])
    assert input_ids.shape == labels.shape == pad_mask.shape
    assert input_ids.shape[0] == 8
