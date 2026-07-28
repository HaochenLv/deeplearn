from data import Tokenizer


def test_roundtrip():
    tok = Tokenizer()
    s = "37+48=\n7+8+0=15 d5 c1\n3+4+1=8 d8 c0\n=85"
    assert tok.decode(tok.encode(s)) == s


def test_special_ids():
    tok = Tokenizer()
    assert tok.pad_id == 0
    assert tok.bos_id == 1
    assert tok.eos_id == 2


def test_vocab_size():
    tok = Tokenizer()
    # 3 specials + "0123456789+=dc \n" (16)
    assert tok.vocab_size == 3 + 16
