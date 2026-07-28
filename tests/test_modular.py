import torch
from modular_addition import (add, encode_abc, DigitAdder, IN_DIM,
                               CarryPropagator, generate_carry_data,
                               train_carry_prop, train_adder, neural_add,
                               evaluate_neural_lengths)


class OracleAdder:
    """确定性个位加法器，用于测试进位路由程序本身（不依赖训练）。"""
    def predict(self, a, b, cin):
        s = a + b + cin
        return s % 10, s // 10


def test_encode_abc():
    x = encode_abc(3, 4, 1)
    assert x.shape == (IN_DIM,)
    assert x.sum().item() == 3
    assert x[3].item() == 1 and x[14].item() == 1 and x[21].item() == 1


def test_digit_adder_shape():
    m = DigitAdder()
    x = torch.stack([encode_abc(3, 4, 0), encode_abc(9, 9, 1)])
    d, c = m(x)
    assert d.shape == (2, 10)
    assert c.shape == (2, 2)


def test_add_program_no_carry():
    o = OracleAdder()
    assert add("3", "4", o) == "7"
    assert add("12", "13", o) == "25"          # 逐位无进位


def test_add_program_with_carry():
    o = OracleAdder()
    assert add("37", "48", o) == "85"          # 个位进位
    assert add("99", "1", o) == "100"          # 连续进位
    assert add("999", "1", o) == "1000"        # 全进位滚出


def test_add_program_zero():
    o = OracleAdder()
    assert add("0", "0", o) == "0"


def test_add_program_unequal_length():
    o = OracleAdder()
    assert add("5", "37", o) == "42"           # zfill 对齐


def test_carry_propagator_shape():
    cp = CarryPropagator(hidden=8)
    # cout_seq: 3 个位置，每个 cout one-hot(2)
    cout_seq = torch.tensor([[1., 0.], [0., 1.], [1., 0.]])
    cins = cp(cout_seq)
    assert len(cins) == 3
    for c in cins:
        assert c.shape == (2,)


def test_carry_propagator_cin_from_h():
    cp = CarryPropagator(hidden=8)
    h = torch.zeros(8)
    cin_logits = cp.cin_from_h(h)
    assert cin_logits.shape == (2,)


def test_generate_carry_data_shape():
    cout_seqs, cin_labels, lengths = generate_carry_data(10, max_len=5, seed=42)
    assert cout_seqs.shape == (10, 5, 2)       # N, L, 2
    assert cin_labels.shape == (10, 5)          # N, L
    assert len(lengths) == 10
    for l in lengths:
        assert 1 <= l <= 5


def test_generate_carry_data_values():
    cout_seqs, cin_labels, lengths = generate_carry_data(5, max_len=2, seed=99)
    for i in range(5):
        for j in range(lengths[i]):
            c = cin_labels[i, j].item()
            assert c in (0, 1)
            cout_val = cout_seqs[i, j].argmax().item()
            assert cout_val in (0, 1)


def test_generate_carry_data_cin_first_is_zero():
    """每条序列首位 cin 必须是 0（没有来自更低位的进位）。"""
    cout_seqs, cin_labels, lengths = generate_carry_data(50, max_len=8, seed=7)
    for i in range(50):
        assert cin_labels[i, 0].item() == 0


def test_train_carry_prop_converges():
    """短训练后 CarryPropagator 应在短序列上接近 100%。"""
    cp = train_carry_prop(epochs=50, lr=1e-2, hidden=8, max_len=4, n_per_epoch=500, seed=0)
    # 在 1-4 位上测试
    cout_seqs, cin_labels, lengths = generate_carry_data(200, max_len=4, seed=999)
    cp.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i in range(200):
            L = lengths[i]
            cins = cp(cout_seqs[i, :L])
            for j, cin_logit in enumerate(cins):
                pred = cin_logit.argmax().item()
                target = cin_labels[i, j].item()
                correct += int(pred == target)
                total += 1
    acc = correct / total
    assert acc > 0.95, f"CarryPropagator acc={acc:.3f}, expected > 0.95"


def test_neural_add_returns_string():
    """neural_add 应返回数字字符串。"""
    o = OracleAdder()
    cp = CarryPropagator(hidden=8)
    result = neural_add("3", "4", o, cp)
    assert isinstance(result, str)
    assert result.isdigit()


def test_neural_add_trained_system():
    """训练好的 DigitAdder + CarryPropagator 组合应在简单用例上正确。"""
    adder = train_adder(epochs=200, lr=1e-2, seed=0)
    cp = train_carry_prop(epochs=100, lr=1e-2, hidden=8, max_len=8, n_per_epoch=2000, seed=0)
    # 简单无进位
    assert neural_add("3", "4", adder, cp) == "7"
    # 有进位
    assert neural_add("37", "48", adder, cp) == "85"
    # 连续进位
    assert neural_add("99", "1", adder, cp) == "100"


def test_evaluate_neural_lengths_returns_dict():
    adder = train_adder(epochs=200, lr=1e-2, seed=0)
    cp = train_carry_prop(epochs=100, lr=1e-2, hidden=8, max_len=8, n_per_epoch=2000, seed=0)
    results = evaluate_neural_lengths(adder, cp, n_per_len=50, seed=42, lengths=[1, 2, 4])
    assert isinstance(results, dict)
    assert set(results.keys()) == {1, 2, 4}
    for v in results.values():
        assert 0.0 <= v <= 1.0
