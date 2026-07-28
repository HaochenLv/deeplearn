"""MLP 字符加法（MNIST 模式）—— 最朴素的“权重矩阵”入门模型（位数可配置）。

把 N 位加法算式当成【固定向量 → 权重矩阵 → 多分类】来学，和 MNIST 同构：
  MNIST : 784 像素 → MLP → 10 类
  加法  : 算式字符 one-hot 拼接 → MLP → 和的每位数字(多个 10 类头)

用法: python3 mlp_addition.py [max_digits]   默认 max_digits=2

对照意义：transformer 走“算法学习/外推”，MLP 走“模式识别/固定映射”。
位数越多，映射空间指数增长（10^(2N)），MLP 越无法记忆、越依赖泛化——
而 MLP 缺乏“逐位/进位”的归纳偏置，这是它的容量极限所在。
"""
import random
import sys
import torch
import torch.nn as nn

CHARS = "0123456789+"                # 字符集（11 类）
CHAR_TO_IDX = {c: i for i, c in enumerate(CHARS)}
VOCAB = len(CHARS)                   # 11


def dims(max_digits):
    """返回 (expr_len, in_dim, out_digits)。"""
    expr_len = 2 * max_digits + 1
    return expr_len, expr_len * VOCAB, max_digits + 1


def make_expr(a: int, b: int, max_digits: int) -> str:
    return f"{a:0{max_digits}d}+{b:0{max_digits}d}"


def encode_input(a: int, b: int, max_digits: int) -> torch.Tensor:
    """算式 one-hot 拼接 → in_dim 维向量。"""
    expr_len, in_dim, _ = dims(max_digits)
    v = torch.zeros(in_dim)
    for pos, ch in enumerate(make_expr(a, b, max_digits)):
        v[pos * VOCAB + CHAR_TO_IDX[ch]] = 1.0
    return v


def encode_target(s: int, max_digits: int) -> torch.Tensor:
    """和 s → out_digits 位数字标签（高位在前，前导零补齐）。85,2 → [0,8,5]。"""
    _, _, out_digits = dims(max_digits)
    return torch.tensor([int(d) for d in f"{s:0{out_digits}d}"], dtype=torch.long)


def decode_digits(digits) -> int:
    out = 0
    for d in digits:
        out = out * 10 + int(d)
    return out


class MLP(nn.Module):
    def __init__(self, max_digits, hidden=128):
        super().__init__()
        _, in_dim, out_digits = dims(max_digits)
        self.out_digits = out_digits
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head = nn.Linear(hidden, out_digits * 10)

    def forward(self, x):                            # x: [B, in_dim]
        h = self.trunk(x)                            # [B, hidden]
        return self.head(h).view(-1, self.out_digits, 10)  # [B, out_digits, 10]


def sample_batch(n, rng, max_digits):
    hi = 10 ** max_digits
    a = [rng.randrange(hi) for _ in range(n)]
    b = [rng.randrange(hi) for _ in range(n)]
    x = torch.stack([encode_input(ai, bi, max_digits) for ai, bi in zip(a, b)])
    y = torch.stack([encode_target(ai + bi, max_digits) for ai, bi in zip(a, b)])
    return x, y


def train(max_digits, epochs=20, batch=128, lr=1e-3, seed=0, n_per_epoch=20000, hidden=128):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = MLP(max_digits, hidden)
    _, _, out_digits = dims(max_digits)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{max_digits}位] MLP 参数量: {n_params:,}  输入{dims(max_digits)[1]}维, 和最多{out_digits}位, 映射空间 10^{2*max_digits}")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(1, epochs + 1):
        model.train()
        total, steps = 0.0, n_per_epoch // batch
        for _ in range(steps):
            x, y = sample_batch(batch, rng, max_digits)
            logits = model(x)
            loss = loss_fn(logits.reshape(-1, 10), y.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        acc = evaluate(model, max_digits, n=2048, seed=999)
        print(f"epoch {ep:2d}: train_loss={total/steps:.4f}  test_exact_match={acc:.3f}")
    return model


@torch.no_grad()
def evaluate(model, max_digits, n=2048, seed=12345):
    rng = random.Random(seed)
    model.eval()
    hi = 10 ** max_digits
    batch = 256
    nb = n // batch
    correct = 0
    for _ in range(nb):
        a = [rng.randrange(hi) for _ in range(batch)]
        b = [rng.randrange(hi) for _ in range(batch)]
        x = torch.stack([encode_input(ai, bi, max_digits) for ai, bi in zip(a, b)])
        pred = model(x).argmax(-1)
        for ai, bi, p in zip(a, b, pred):
            if decode_digits(p) == ai + bi:
                correct += 1
    return correct / (nb * batch)


def selfcheck():
    assert make_expr(5, 37, 2) == "05+37"
    assert encode_input(5, 37, 2).sum().item() == dims(2)[0]
    assert encode_target(85, 2).tolist() == [0, 8, 5]
    assert encode_target(198, 2).tolist() == [1, 9, 8]
    assert decode_digits([0, 8, 5]) == 85
    print("selfcheck ok")


if __name__ == "__main__":
    selfcheck()
    md = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    model = train(md)
    print(f"\nfinal test exact_match ({md}位) =", evaluate(model, md, n=5000, seed=4321))
