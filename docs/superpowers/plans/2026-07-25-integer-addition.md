# 整数加法课程学习 + 长度外推 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 训练一个微型 decoder-only Transformer，用竖式 Scratchpad + 纯课程学习（1→4 位）学会整数加法，并对未见过的 5–8 位保持准确率（长度外推）。

**Architecture:** 字符级分词 → 微型 GPT（2 层 / d_model=128 / 4 头 / ALiBi 位置偏置、无绝对位置编码）→ 自回归生成竖式草稿与答案。数据为合成加法对，课程按位逐级训练（保留旧长度防遗忘），评估逐长度精确匹配 + 外推曲线 + 遗忘探针。

**Tech Stack:** Python 3、PyTorch（CPU）、NumPy、Matplotlib、pytest。工程 venv 隔离（`.venv`）。

## Global Constraints

（每个任务的隐含前置；值逐字取自 spec `docs/superpowers/specs/2026-07-25-integer-addition-design.md`）

- **Python 执行约定**：工程内有 `.venv`，所有 `python`/`pytest`/`pip` 命令前缀 `source .venv/bin/activate &&`，裸用 `python3`/`pytest`/`pip`。
- **算力**：单机 CPU；模型保持微型（约几十万参数）。
- **表示**：字符级字符串；词表 ≈17 token（`0-9 + = d c \n` + `<pad> <bos> <eos>`）。
- **Scratchpad 格式**：竖式，每列一行 `<a>+<b>+<进位入>=<和> d<写本位> c<进位出>`，从个位到高位，末行 `=<答案>`。
- **损失**：仅对 target token（`=` 之后）计算，prompt 部分 label 置 `-100`（label masking）。
- **位置编码**：ALiBi（无绝对位置嵌入）。
- **训练长度**：1–4 位；**外推测试**：5–8 位。
- **课程**：第 `d` 阶段数据 = 50% 长度 `d` + 50% 旧长度 1..d-1；**毕业判据**：长度 `d` 验证 token 准确率 ≥ 0.95 或达该阶段最大 epoch。
- **数字采样**：`d` 位数在 `[0, 10^d - 1]` 均匀采样，输入按 `d` 位零填充对齐（如 `05+37=`）。

---

## File Structure

| 文件 | 职责 |
|---|---|
| `data.py` | 加法对采样、竖式 scratchpad 生成、字符级 `Tokenizer`、`AdditionDataset`、collate（含 label masking） |
| `model.py` | `AdditionTransformer`（ALiBi + 因果自注意力 + pad 掩码） |
| `train.py` | `compute_loss`、`teacher_forced_accuracy`、`train_stage`（课程 + 毕业 + 探针日志） |
| `eval.py` | `greedy_decode`、`extract_answer`、`evaluate_lengths` |
| `run.py` | 串联课程 1→4、外推测 1–8、产出外推曲线图 + 遗忘探针图 |
| `tests/test_*.py` | 各单元测试 |
| `requirements.txt` / `pyproject.toml` | 依赖与 pytest 配置 |

**跨任务接口契约**（命名/签名必须一致）：
- `data.py`: `format_prompt(a,b,d)->str`、`make_columns(a,b)->list[(x,y,carry_in,digit_out,carry_out)]`、`make_target(a,b)->str`（以 `\n` 开头）、`sample_pair(d,rng)->(int,int)`、`build_length_examples(d,n,seed)`、`build_curriculum_examples(d,n,seed)`、`class Tokenizer`、`class AdditionDataset(tokenizer, examples)`、`make_collate_fn(tokenizer)`
- `model.py`: `class AdditionTransformer(vocab_size, d_model=128, n_heads=4, n_layers=2, max_len=256)`，`.forward(idx, pad_mask=None)->logits[B,T,V]`
- `train.py`: `compute_loss(model, input_ids, labels, pad_mask)->(loss, logits)`、`teacher_forced_accuracy(model, dataset, tok, device, batch_size=64)->float`、`train_stage(model, tok, stage_d, device, ...)->history`
- `eval.py`: `greedy_decode(model, tok, prompt_str, device, max_new_tokens)->str`、`extract_answer(text)->str|None`、`column_accuracy(text,a,b)->float`、`evaluate_lengths(model, tok, lengths, n, device)->dict[int,dict]`

---

## Task 1: 工程脚手架与环境

**Files:**
- Create: `requirements.txt`
- Create: `pyproject.toml`
- Create: `tests/test_smoke.py`

**Interfaces:** 产出可运行 venv + pytest；后续任务依赖。

- [ ] **Step 1: 写 `requirements.txt`**

```
torch>=2.0
numpy>=1.24
matplotlib>=3.7
pytest>=7.0
```

- [ ] **Step 2: 写 `pyproject.toml`（pytest 配置，让 `tests/` 能 import 根目录模块）**

```toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

- [ ] **Step 3: 写失败测试 `tests/test_smoke.py`**

```python
def test_deps_importable():
    import torch, numpy, matplotlib
    assert torch.__version__
    assert numpy.__version__
    assert matplotlib.__version__

def test_python_version():
    import sys
    assert sys.version_info >= (3, 9)
```

- [ ] **Step 4: 建 venv 并装依赖**

Run:
```bash
python3 -m venv .venv
source .venv/bin/activate && pip install -r requirements.txt
```
Expected: 依赖安装成功（torch 取 CPU 版）。

- [ ] **Step 5: 运行测试验证通过**

Run: `source .venv/bin/activate && pytest tests/test_smoke.py -v`
Expected: 2 passed

- [ ] **Step 6: 提交**

```bash
git add requirements.txt pyproject.toml tests/test_smoke.py
git commit -m "chore: 工程脚手架与依赖（venv/torch/pytest）"
```

---

## Task 2: 竖式 Scratchpad 生成（`data.py` 核心）

**Files:**
- Create: `data.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Produces: `format_prompt`, `make_columns`, `make_target`, `make_full`, `sample_pair`

- [ ] **Step 1: 写失败测试 `tests/test_data.py`**

```python
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
```

- [ ] **Step 2: 运行测试验证失败**

Run: `source .venv/bin/activate && pytest tests/test_data.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'data'`）

- [ ] **Step 3: 写实现 `data.py`（本任务部分；后续任务追加）**

```python
import random


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
    lines = [f"{x}+{y}+{cin}={s} d{dout} c{cout}"
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
```

- [ ] **Step 4: 运行测试验证通过**

Run: `source .venv/bin/activate && pytest tests/test_data.py -v`
Expected: 10 passed

- [ ] **Step 5: 提交**

```bash
git add data.py tests/test_data.py
git commit -m "feat(data): 竖式 scratchpad 生成与采样"
```

---

## Task 3: 字符级 Tokenizer（`data.py` 续）

**Files:**
- Modify: `data.py`（追加 `Tokenizer`）
- Test: `tests/test_tokenizer.py`

**Interfaces:**
- Produces: `class Tokenizer`（`.encode/.decode/.vocab_size/.pad_id/.bos_id/.eos_id`）

- [ ] **Step 1: 写失败测试 `tests/test_tokenizer.py`**

```python
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
    # 3 specials + "0123456789+=dc\n" (14)
    assert tok.vocab_size == 3 + 14
```

- [ ] **Step 2: 运行验证失败**

Run: `source .venv/bin/activate && pytest tests/test_tokenizer.py -v`
Expected: FAIL（`ImportError: cannot import name 'Tokenizer'`）

- [ ] **Step 3: 追加实现到 `data.py`**

```python
class Tokenizer:
    VOCAB_CHARS = list("0123456789+=dc\n")
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
```

- [ ] **Step 4: 运行验证通过**

Run: `source .venv/bin/activate && pytest tests/test_tokenizer.py -v`
Expected: 3 passed

- [ ] **Step 5: 提交**

```bash
git add data.py tests/test_tokenizer.py
git commit -m "feat(data): 字符级 Tokenizer"
```

---

## Task 4: Dataset 与 collate（含 label masking）

**Files:**
- Modify: `data.py`（追加 `build_length_examples`、`build_curriculum_examples`、`AdditionDataset`、`make_collate_fn`）
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: Task 2/3 的 `format_prompt`/`make_target`/`Tokenizer`/`sample_pair`
- Produces: `AdditionDataset(tokenizer, examples)`、`make_collate_fn(tokenizer)`（返回 `(input_ids, labels, pad_mask)`，labels 中 prompt 与 pad 置 `-100`）

- [ ] **Step 1: 写失败测试 `tests/test_dataset.py`**

```python
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
    # 标签里没有出现在“prompt 区”或 pad 上的非 -100 之外的非法情况：
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
```

- [ ] **Step 2: 运行验证失败**

Run: `source .venv/bin/activate && pytest tests/test_dataset.py -v`
Expected: FAIL（`ImportError: cannot import name 'build_length_examples'`）

- [ ] **Step 3: 追加实现到 `data.py`（顶部加 `import torch`）**

```python
import torch
from torch.utils.data import Dataset


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
```

- [ ] **Step 4: 运行验证通过**

Run: `source .venv/bin/activate && pytest tests/test_dataset.py -v`
Expected: 4 passed

- [ ] **Step 5: 提交**

```bash
git add data.py tests/test_dataset.py
git commit -m "feat(data): AdditionDataset + label-masking collate + 课程采样"
```

---

## Task 5: 微型 Transformer（ALiBi）（`model.py`）

**Files:**
- Create: `model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Produces: `AdditionTransformer(vocab_size, d_model=128, n_heads=4, n_layers=2, max_len=256)`，`.forward(idx, pad_mask=None) -> logits[B,T,vocab_size]`

- [ ] **Step 1: 写失败测试 `tests/test_model.py`**

```python
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
```

- [ ] **Step 2: 运行验证失败**

Run: `source .venv/bin/activate && pytest tests/test_model.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'model'`）

- [ ] **Step 3: 写实现 `model.py`**

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def alibi_biases(n_heads: int, seq_len: int, device):
    """返回 [n_heads, 1, seq_len, seq_len] 的 ALiBi 偏置（含因果掩码）。"""
    start = 2.0 ** (-8.0 / n_heads)
    slopes = torch.tensor([start ** (i + 1) for i in range(n_heads)], device=device)
    pos = torch.arange(seq_len, device=device)
    rel = pos[None, :] - pos[:, None]            # rel[q,k] = k - q
    bias = slopes.view(-1, 1, 1, 1) * rel.view(1, 1, seq_len, seq_len)
    causal = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
    bias = bias.masked_fill(~causal, float("-inf"))
    return bias                                   # [H,1,L,L]


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, bias, pad_mask=None):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)               # 各 [B,T,H,dh]
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # [B,H,T,dh]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        att = att + bias                           # [B,H,T,T] + [H,1,T,T]
        if pad_mask is not None:
            att = att.masked_fill(pad_mask[:, None, None, :], float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.out(y)


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x, bias, pad_mask=None):
        x = x + self.attn(self.ln1(x), bias, pad_mask)
        x = x + self.ff(self.ln2(x))
        return x


class AdditionTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2, max_len=256):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.n_heads = n_heads
        self.max_len = max_len

    def forward(self, idx, pad_mask=None):
        B, T = idx.shape
        x = self.tok_emb(idx)
        bias = alibi_biases(self.n_heads, T, idx.device)
        for blk in self.blocks:
            x = blk(x, bias, pad_mask)
        x = self.ln_f(x)
        return self.head(x)
```

- [ ] **Step 4: 运行验证通过**

Run: `source .venv/bin/activate && pytest tests/test_model.py -v`
Expected: 4 passed

- [ ] **Step 5: 提交**

```bash
git add model.py tests/test_model.py
git commit -m "feat(model): 微型 decoder-only Transformer + ALiBi"
```

---

## Task 6: 训练步骤、毕业判据与遗忘探针（`train.py`）

**Files:**
- Create: `train.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: Task 4/5 的 `make_collate_fn`、`AdditionDataset`、`build_*_examples`、`AdditionTransformer`
- Produces: `compute_loss`、`teacher_forced_accuracy`、`train_stage`

- [ ] **Step 1: 写失败测试 `tests/test_train.py`**

```python
import torch
from data import Tokenizer, build_length_examples, AdditionDataset, make_collate_fn
from model import AdditionTransformer
from train import compute_loss, teacher_forced_accuracy


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
```

- [ ] **Step 2: 运行验证失败**

Run: `source .venv/bin/activate && pytest tests/test_train.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'train'`）

- [ ] **Step 3: 写实现 `train.py`**

```python
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import (
    Tokenizer, AdditionDataset, build_curriculum_examples, build_length_examples,
    make_collate_fn,
)


def compute_loss(model, input_ids, labels, pad_mask):
    logits = model(input_ids, pad_mask)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
    )
    return loss, logits


@torch.no_grad()
def teacher_forced_accuracy(model, dataset, tok: Tokenizer, device, batch_size=64):
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=make_collate_fn(tok))
    correct, total = 0, 0
    model.eval()
    for input_ids, labels, pad_mask in loader:
        logits = model(input_ids.to(device), pad_mask.to(device))
        pred = logits.argmax(-1).cpu()
        mask = labels != -100
        correct += (pred[mask] == labels[mask]).sum().item()
        total += int(mask.sum().item())
    model.train()
    return correct / total if total else 0.0


def train_stage(model, tok: Tokenizer, stage_d: int, device,
                n_examples=4000, batch_size=64, max_epochs=30,
                lr=1e-3, grad_clip=1.0, grad_threshold=0.95, seed=0,
                probe_lengths=(1, 2, 3, 4, 5), log_every=1):
    """训练第 stage_d 阶段（课程数据），毕业即返回。history 记录遗忘探针。"""
    collate = make_collate_fn(tok)
    train_ds = AdditionDataset(tok, build_curriculum_examples(stage_d, n_examples, seed=seed))
    val_ds = AdditionDataset(tok, build_length_examples(stage_d, 256, seed=seed + 1000))
    probe_ds = {L: AdditionDataset(tok, build_length_examples(L, 64, seed=9000 + L))
                for L in probe_lengths}
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    history = []  # [(epoch, stage_d_acc, {L: acc}, mean_loss)]，mean_loss 供观察 grokking
    model.train()
    for epoch in range(1, max_epochs + 1):
        ep_loss, nb = 0.0, 0
        for input_ids, labels, pad_mask in loader:
            loss, _ = compute_loss(model, input_ids.to(device), labels.to(device), pad_mask.to(device))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            ep_loss += loss.item(); nb += 1
        mean_loss = ep_loss / max(nb, 1)
        acc_d = teacher_forced_accuracy(model, val_ds, tok, device)
        probe = {L: teacher_forced_accuracy(model, probe_ds[L], tok, device) for L in probe_lengths}
        history.append((epoch, acc_d, probe, mean_loss))
        if epoch % log_every == 0:
            print(f"[stage {stage_d}] epoch {epoch}: loss={mean_loss:.3f} len{stage_d}_acc={acc_d:.3f}")
        if acc_d >= grad_threshold:
            print(f"[stage {stage_d}] 毕业 @ epoch {epoch} (acc={acc_d:.3f})")
            break
    return history
```

- [ ] **Step 4: 运行验证通过**

Run: `source .venv/bin/activate && pytest tests/test_train.py -v`
Expected: 3 passed（`test_overfit_tiny_batch` 可能跑 ~1 分钟）

- [ ] **Step 5: 提交**

```bash
git add train.py tests/test_train.py
git commit -m "feat(train): masked loss + token 准确率 + 课程阶段/毕业/遗忘探针"
```

---

## Task 7: 推理解码与逐长度评估（`eval.py`）

**Files:**
- Create: `eval.py`
- Test: `tests/test_eval.py`

**Interfaces:**
- Consumes: Task 2/3/5 的 `make_target`/`format_prompt`/`Tokenizer`/`AdditionTransformer`
- Produces: `greedy_decode`、`extract_answer`、`column_accuracy`、`evaluate_lengths`

- [ ] **Step 1: 写失败测试 `tests/test_eval.py`**

```python
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
```

- [ ] **Step 2: 运行验证失败**

Run: `source .venv/bin/activate && pytest tests/test_eval.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'eval'`）

- [ ] **Step 3: 写实现 `eval.py`**

```python
import re
import torch

from data import Tokenizer, sample_pair, format_prompt, make_target, make_columns


@torch.no_grad()
def greedy_decode(model, tok: Tokenizer, prompt_str: str, device, max_new_tokens=128):
    """返回模型生成的 target 文本（不含 bos/prompt/eos）。"""
    model.eval()
    prompt_ids = tok.encode(prompt_str)
    ids = [tok.bos_id] + prompt_ids
    generated = []
    for _ in range(max_new_tokens):
        x = torch.tensor([ids], device=device, dtype=torch.long)
        logits = model(x)
        next_id = int(logits[0, -1].argmax().item())
        if next_id == tok.eos_id:
            break
        generated.append(next_id)
        ids.append(next_id)
    return tok.decode(generated)


def column_accuracy(text: str, a: int, b: int):
    """草稿每列 d<digit> c<carry> 与真值的逐列匹配率（进位诊断）。"""
    pairs = re.findall(r"d(\d)\s*c(\d)", text)
    true = [(dout, cout) for (_, _, _, dout, cout) in make_columns(a, b)]
    if not true:
        return 1.0
    correct = sum(1 for i, (d, c) in enumerate(pairs)
                  if i < len(true) and (int(d), int(c)) == true[i])
    return correct / len(true)


def extract_answer(text: str):
    """从生成文本末尾的 '=答案' 提取答案数字串；无则 None。"""
    idx = text.rfind("=")
    if idx < 0:
        return None
    tail = text[idx + 1:]
    ans = "".join(ch for ch in tail if ch.isdigit())
    return ans if ans else None


@torch.no_grad()
def evaluate_lengths(model, tok: Tokenizer, lengths, n: int, device, seed=12345):
    import random
    rng = random.Random(seed)
    results = {}
    model.eval()
    for L in lengths:
        correct, em, carry_tot, total = 0, 0, 0.0, 0
        for _ in range(n):
            a, b = sample_pair(L, rng)
            prompt = format_prompt(a, b, L)
            true_ans = str(a + b)
            out = greedy_decode(model, tok, prompt, device, max_new_tokens=20 * L + 16)
            pred_ans = extract_answer(out)
            total += 1
            if pred_ans == true_ans:
                correct += 1
            if out.strip() == make_target(a, b).strip():
                em += 1
            carry_tot += column_accuracy(out, a, b)
        results[L] = {
            "answer_acc": correct / total,
            "exact_match": em / total,
            "carry_acc": carry_tot / total,
        }
    return results
```

- [ ] **Step 4: 运行验证通过**

Run: `source .venv/bin/activate && pytest tests/test_eval.py -v`
Expected: 3 passed

- [ ] **Step 5: 提交**

```bash
git add eval.py tests/test_eval.py
git commit -m "feat(eval): 贪婪解码 + 答案提取 + 逐长度评估"
```

---

## Task 8: 端到端编排与出图（`run.py`）

**Files:**
- Create: `run.py`
- Test: `tests/test_run.py`

**Interfaces:**
- Consumes: Task 6/7 的 `train_stage`、`evaluate_lengths`
- Produces: `plot_extrapolation`、`plot_forgetting`、`main`

- [ ] **Step 1: 写失败测试 `tests/test_run.py`**

```python
import os
from run import plot_extrapolation, plot_forgetting


def test_plot_extrapolation_writes_file(tmp_path):
    out = tmp_path / "extrap.png"
    plot_extrapolation([1, 2, 3, 4, 5, 6, 7, 8],
                       [1.0, 0.99, 0.98, 0.97, 0.9, 0.7, 0.4, 0.2],
                       str(out))
    assert os.path.getsize(str(out)) > 0


def test_plot_forgetting_writes_file(tmp_path):
    out = tmp_path / "forget.png"
    history = [(1, 0.95, {1: 0.95, 2: 0.1}),
               (2, 0.96, {1: 0.94, 2: 0.96}),
               (3, 0.95, {1: 0.93, 2: 0.95, 3: 0.95})]
    plot_forgetting(history, [1, 2, 3], str(out))
    assert os.path.getsize(str(out)) > 0
```

- [ ] **Step 2: 运行验证失败**

Run: `source .venv/bin/activate && pytest tests/test_run.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'run'`）

- [ ] **Step 3: 写实现 `run.py`**

```python
import os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import Tokenizer
from model import AdditionTransformer
from train import train_stage
from eval import evaluate_lengths

OUT_DIR = "outputs"
CKPT_DIR = "checkpoints"


def plot_extrapolation(lengths, accs, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(lengths, accs, marker="o")
    plt.axvline(4.5, color="r", linestyle="--", label="训练长度上限 (4)")
    plt.xlabel("位数")
    plt.ylabel("答案正确率")
    plt.title("长度外推曲线")
    plt.ylim(-0.05, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_forgetting(history, lengths, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.figure(figsize=(6, 4))
    epochs = [h[0] for h in history]
    for L in lengths:
        accs = [h[2].get(L) for h in history]
        plt.plot(epochs, accs, marker="o", label=f"len {L}")
    plt.xlabel("epoch（末阶段）")
    plt.ylabel("token 准确率")
    plt.title("遗忘探针")
    plt.ylim(-0.05, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = Tokenizer()
    model = AdditionTransformer(tok.vocab_size).to(device)

    all_history = []
    for stage_d in range(1, 5):
        print(f"\n=== 课程阶段 {stage_d} ===")
        hist = train_stage(model, tok, stage_d, device, probe_lengths=(1, 2, 3, 4))
        all_history.extend(hist)
        torch.save(model.state_dict(), os.path.join(CKPT_DIR, f"stage{stage_d}.pt"))

    print("\n=== 评估 1–8 位 ===")
    results = evaluate_lengths(model, tok, range(1, 9), n=200, device=device)
    for L in range(1, 9):
        r = results[L]
        print(f"len {L}: answer_acc={r['answer_acc']:.3f} exact_match={r['exact_match']:.3f}")

    lengths = list(range(1, 9))
    accs = [results[L]["answer_acc"] for L in lengths]
    plot_extrapolation(lengths, accs, os.path.join(OUT_DIR, "extrapolation.png"))
    plot_forgetting(all_history, [1, 2, 3], os.path.join(OUT_DIR, "forgetting.png"))
    print(f"\n图已保存到 {OUT_DIR}/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行验证通过**

Run: `source .venv/bin/activate && pytest tests/test_run.py -v`
Expected: 2 passed

- [ ] **Step 5: 端到端冒烟跑（小规模，验证整条管线）**

Run:
```bash
source .venv/bin/activate && python3 -c "
from data import Tokenizer
from model import AdditionTransformer
from train import train_stage
from eval import evaluate_lengths
import torch
tok=Tokenizer(); m=AdditionTransformer(tok.vocab_size)
train_stage(m, tok, 1, torch.device('cpu'), n_examples=2000, max_epochs=5)
r=evaluate_lengths(m, tok, [1], n=20, device=torch.device('cpu'))
print('len1 answer_acc=', r[1]['answer_acc'])
"
```
Expected: 打印出 len1 answer_acc（>0 即管线通），无异常。

- [ ] **Step 6: 提交**

```bash
git add run.py tests/test_run.py
git commit -m "feat(run): 课程编排 + 外推曲线/遗忘探针出图"
```

---

## 跑完之后（验收对照 spec §10）

- `source .venv/bin/activate && python3 run.py` 完整跑完，产出 `outputs/extrapolation.png` 与 `outputs/forgetting.png`。
- 关注：4 位训练准确率是否达标；5–8 位外推正确率是否显著高于"朴素无 scratchpad"基线（预期后者≈0）。
- 若外推不理想，按 spec §12 风险表排查：scratchpad 正确性（Task 2 测试）→ PE（ALiBi↔加绝对 PE）→ 模型容量/训练步数。

## 可消融旋钮（spec §11，非本计划必做）

1. 位置编码：ALiBi ↔ NoPE（去掉 `alibi_biases` 项）↔ 加可学习绝对位置嵌入。
2. 课程 vs 混合 [1..4]：用 `build_length_examples` 在多长度上混合训练做对照。
3. 训练/测试位数调整。
