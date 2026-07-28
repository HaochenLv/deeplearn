import random
import torch
from torch.utils.data import Dataset


def make_columns(a: int, b: int):
    """逐列竖式（从个位到高位）。返回 [(x, y, carry_in, digit_out, carry_out), ...]。"""
    cols = []
    carry = 0
    da, db = a, b
    while da > 0 or db > 0 or carry > 0 or not cols:
        x, y = da % 10, db % 10
        s = x + y + carry
        cols.append((x, y, carry, s % 10, s // 10))
        carry = s // 10
        da, db = da // 10, db // 10
    return cols


def make_target(a: int, b: int) -> str:
    """竖式草稿，以 '\\n' 开头；末行 '=答案'。"""
    cols = make_columns(a, b)
    lines = [f"{x}+{y}+{cin}={x+y+cin} d{dout} c{cout}"
             for (x, y, cin, dout, cout) in cols]
    digits = [dout for (_, _, _, dout, _) in cols]
    answer = "".join(str(d) for d in reversed(digits))
    lines.append(f"={answer}")
    return "\n" + "\n".join(lines)


def format_prompt(a: int, b: int, d: int) -> str:
    return f"{a:0{d}d}+{b:0{d}d}="


def make_full(a: int, b: int, d: int) -> str:
    return format_prompt(a, b, d) + make_target(a, b)


def sample_pair(d: int, rng: random.Random):
    hi = 10 ** d
    return rng.randrange(hi), rng.randrange(hi)


class Tokenizer:
    VOCAB_CHARS = list("0123456789+=dc \n")
    SPECIALS = ["<pad>", "<bos>", "<eos>"]

    def __init__(self):
        self.itos = self.SPECIALS + self.VOCAB_CHARS
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        self.pad_id = self.stoi["<pad>"]
        self.bos_id = self.stoi["<bos>"]
        self.eos_id = self.stoi["<eos>"]

    @property
    def vocab_size(self):
        return len(self.itos)

    def encode(self, s: str):
        return [self.stoi[ch] for ch in s]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)


def build_length_examples(d: int, n: int, seed: int):
    rng = random.Random(seed)
    return [(*sample_pair(d, rng), d) for _ in range(n)]


def build_curriculum_examples(d: int, n: int, seed: int):
    """第 d 阶段：50% 长度 d + 50% 旧长度 1..d-1（防遗忘）。"""
    rng = random.Random(seed)
    out = []
    half = n // 2
    for _ in range(half):
        out.append((*sample_pair(d, rng), d))
    for _ in range(n - half):
        od = rng.randint(1, d - 1) if d > 1 else 1
        out.append((*sample_pair(od, rng), od))
    rng.shuffle(out)
    return out


class AdditionDataset(Dataset):
    def __init__(self, tokenizer, examples):
        self.tok = tokenizer
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        a, b, d = self.examples[i]
        prompt = format_prompt(a, b, d)
        target = make_target(a, b)
        full = ([self.tok.bos_id]
                + self.tok.encode(prompt)
                + self.tok.encode(target)
                + [self.tok.eos_id])
        prompt_tok_count = len(self.tok.encode(prompt))  # labels 中需 mask 的前缀长度
        return full, prompt_tok_count


def make_collate_fn(tokenizer):
    pad = tokenizer.pad_id

    def collate(batch):
        seqs, prompt_lens = zip(*batch)
        max_len = max(len(s) for s in seqs)
        input_ids, labels, pad_mask = [], [], []
        for seq, k in zip(seqs, prompt_lens):
            padded = seq + [pad] * (max_len - len(seq))
            inp = padded[:-1]
            lab = [-100 if (i < k or t == pad) else t for i, t in enumerate(padded[1:])]
            pm = [t == pad for t in inp]
            input_ids.append(inp)
            labels.append(lab)
            pad_mask.append(pm)
        return (torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(labels, dtype=torch.long),
                torch.tensor(pad_mask, dtype=torch.bool))
    return collate
