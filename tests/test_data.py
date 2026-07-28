from data import make_columns, make_target, make_full, format_prompt, sample_pair


def test_no_carry():
    # 12 + 13 = 25
    cols = make_columns(12, 13)
    assert cols == [(2, 3, 0, 5, 0), (1, 1, 0, 2, 0)]


def test_carry():
    # 37 + 48 = 85
    cols = make_columns(37, 48)
    assert cols == [(7, 8, 0, 5, 1), (3, 4, 1, 8, 0)]


def test_carry_propagation_longer_result():
    # 85 + 16 = 101（结果比操作数长）
    cols = make_columns(85, 16)
    assert cols == [(5, 6, 0, 1, 1), (8, 1, 1, 0, 1), (0, 0, 1, 1, 0)]


def test_zeros():
    assert make_columns(0, 0) == [(0, 0, 0, 0, 0)]


def test_rollover():
    # 99 + 1 = 100
    cols = make_columns(99, 1)
    assert cols == [(9, 1, 0, 0, 1), (9, 0, 1, 0, 1), (0, 0, 1, 1, 0)]


def test_target_format():
    assert make_target(37, 48) == "\n7+8+0=15 d5 c1\n3+4+1=8 d8 c0\n=85"


def test_target_format_propagation():
    assert make_target(85, 16) == "\n5+6+0=11 d1 c1\n8+1+1=10 d0 c1\n0+0+1=1 d1 c0\n=101"


def test_format_prompt_zero_padded():
    assert format_prompt(5, 37, 2) == "05+37="
    assert format_prompt(37, 48, 2) == "37+48="


def test_make_full():
    assert make_full(37, 48, 2) == "37+48=\n7+8+0=15 d5 c1\n3+4+1=8 d8 c0\n=85"


def test_sample_pair_range():
    import random
    rng = random.Random(0)
    for _ in range(100):
        a, b = sample_pair(2, rng)
        assert 0 <= a <= 99 and 0 <= b <= 99
