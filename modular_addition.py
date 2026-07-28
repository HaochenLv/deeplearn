"""模块化加法器（神经-符号）：个位加法器(MLP) + 进位路由(程序) = 任意长度加法。

加法分解为两个技能：
  ① 个位加法  a+b+cin → digit + cout     （神经网络学，仅 200 种映射）
  ② 进位路由  cout → 下一位 cin，逐位循环  （确定性程序，不学习）
组合 → 任意 N 位整数加法（突破 MLP 固定长度限制）。

关键：技能②本质是确定性控制流，不需要学习——所以"两个技能"最优实现是
「1 个学习模型 + 1 段程序」，而非两个模型。
"""
import random
import torch
import torch.nn as nn

IN_DIM = 22  # a one-hot(10) + b one-hot(10) + cin one-hot(2)


# ===== 模块①：个位加法器（唯一要学的）=====
class DigitAdder(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(IN_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.digit_head = nn.Linear(hidden, 10)
        self.carry_head = nn.Linear(hidden, 2)

    def forward(self, x):                              # x: [B, 22]
        h = self.net(x)
        return self.digit_head(h), self.carry_head(h)  # [B,10], [B,2]

    @torch.no_grad()
    def predict(self, a, b, cin):
        x = torch.zeros(IN_DIM)
        x[a] = 1.0
        x[10 + b] = 1.0
        x[20 + cin] = 1.0
        d, c = self(x.unsqueeze(0))
        return d.argmax(-1).item(), c.argmax(-1).item()


def encode_abc(a, b, cin):
    x = torch.zeros(IN_DIM)
    x[a] = 1.0
    x[10 + b] = 1.0
    x[20 + cin] = 1.0
    return x


# ===== 模块②：进位传播器（GRU Cell，学习进位传递）=====
class CarryPropagator(nn.Module):
    """GRU Cell 进位传播器：cout 序列 → cin 序列。

    隐状态 h 编码"当前是否有进位需要传递"，
    cin_head 从 h 提取下一位的 cin (0/1)。
    """
    def __init__(self, hidden=8):
        super().__init__()
        self.gru = nn.GRUCell(2, hidden)      # cout one-hot → 隐状态
        self.cin_head = nn.Linear(hidden, 2)   # 隐状态 → cin logits

    def forward(self, cout_seq):
        """前向传播：cout 序列 → cin logits 序列。

        Args:
            cout_seq: [seq_len, 2] 从低位到高位的 cout one-hot 序列
        Returns:
            list of [2] tensors — 每位的 cin logits（首位对应 cin=0）
        """
        h = torch.zeros(self.gru.hidden_size)
        cins = []
        for cout in cout_seq:
            cin_logits = self.cin_head(h)
            cins.append(cin_logits)
            h = self.gru(cout.unsqueeze(0), h.unsqueeze(0)).squeeze(0)
        return cins

    def cin_from_h(self, h):
        """从隐状态提取 cin logits。组合推理时使用。"""
        return self.cin_head(h)

    def step(self, cout, h):
        """单步更新：给定 cout 和当前 h，返回新 h。组合推理时使用。

        Args:
            cout: int (0 or 1)
            h: [hidden] 隐状态
        Returns:
            new_h: [hidden]
        """
        cout_oh = torch.zeros(2)
        cout_oh[cout] = 1.0
        return self.gru(cout_oh.unsqueeze(0), h.unsqueeze(0)).squeeze(0)


def generate_carry_data(n_samples, max_len, seed=0):
    """生成 CarryPropagator 训练数据。

    对每条样本：随机生成 max_len 位以内的 a, b，
    用确定性规则计算每一位的 cout 和下一位的 cin。

    Returns:
        cout_seqs: [n_samples, max_len, 2] — cout one-hot（低位在前，padding 位全零）
        cin_labels: [n_samples, max_len] — cin 标签（0/1，padding 位为 0）
        lengths: list[int] — 每条样本的实际长度
    """
    rng = random.Random(seed)
    cout_seqs = torch.zeros(n_samples, max_len, 2)
    cin_labels = torch.zeros(n_samples, max_len, dtype=torch.long)
    lengths = []
    for i in range(n_samples):
        L = rng.randint(1, max_len)
        lengths.append(L)
        a = rng.randrange(10 ** L)
        b = rng.randrange(10 ** L)
        a_str, b_str = str(a).zfill(L), str(b).zfill(L)
        carry = 0
        for j in range(L):                          # 低位 → 高位
            ai = int(a_str[L - 1 - j])
            bi = int(b_str[L - 1 - j])
            cin_labels[i, j] = carry                # 当前位的 cin
            cout = (ai + bi + carry) // 10          # 当前位的 cout
            cout_seqs[i, j, cout] = 1.0             # cout one-hot
            carry = cout                            # 下一位的 cin = 当前位的 cout
    return cout_seqs, cin_labels, lengths


def train_carry_prop(epochs=100, lr=1e-2, hidden=8, max_len=20, n_per_epoch=5000, seed=0):
    """训练 CarryPropagator：cout 序列 → cin 序列。

    按长度分桶批量训练：同一长度的样本一起 forward + backward，
    避免逐样本逐步的低效方式。
    """
    torch.manual_seed(seed)
    model = CarryPropagator(hidden=hidden)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(1, epochs + 1):
        model.train()
        cout_seqs, cin_labels, lengths = generate_carry_data(
            n_per_epoch, max_len=max_len, seed=seed + ep
        )
        # 按长度分桶
        buckets = {}
        for i, L in enumerate(lengths):
            buckets.setdefault(L, []).append(i)
        total_loss, total_correct, total_tokens = 0.0, 0, 0
        for L, indices in buckets.items():
            batch_cout = cout_seqs[indices, :L]          # [B, L, 2]
            batch_cin = cin_labels[indices, :L]           # [B, L]
            B = len(indices)
            # 逐时间步 forward，收集 logits
            h = torch.zeros(B, model.gru.hidden_size)
            all_logits = []
            for t in range(L):
                cin_logits = model.cin_head(h)            # [B, 2]
                all_logits.append(cin_logits)
                h = model.gru(batch_cout[:, t], h)        # [B, hidden]
            # 一次性计算 loss + backward
            logits_stack = torch.stack(all_logits, dim=1)  # [B, L, 2]
            loss = loss_fn(logits_stack.reshape(-1, 2), batch_cin.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * B * L
            total_correct += (logits_stack.argmax(-1) == batch_cin).sum().item()
            total_tokens += B * L
        if ep == 1 or ep % 25 == 0:
            acc = total_correct / total_tokens
            avg_loss = total_loss / total_tokens
            print(f"carry epoch {ep:3d}: loss={avg_loss:.4f}  cin_acc={acc:.3f}")
    # 最终准确率
    model.eval()
    cout_seqs, cin_labels, lengths = generate_carry_data(1000, max_len=max_len, seed=99999)
    correct, total = 0, 0
    with torch.no_grad():
        for i in range(1000):
            L = lengths[i]
            cins = model(cout_seqs[i, :L])
            for j, cin_logit in enumerate(cins):
                correct += int(cin_logit.argmax().item() == cin_labels[i, j].item())
                total += 1
    print(f"\n进位传播器 参数量={n_params:,}  最终 cin_acc={correct/total:.3f}")
    return model


def train_adder(epochs=200, lr=1e-2, seed=0):
    """在全部 200 种 (a,b,cin) 上训练。"""
    torch.manual_seed(seed)
    model = DigitAdder()
    n_params = sum(p.numel() for p in model.parameters())
    data = [(a, b, cin) for a in range(10) for b in range(10) for cin in range(2)]
    X = torch.stack([encode_abc(a, b, cin) for a, b, cin in data])
    Yd = torch.tensor([(a + b + cin) % 10 for a, b, cin in data])
    Yc = torch.tensor([(a + b + cin) // 10 for a, b, cin in data])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(1, epochs + 1):
        d_logits, c_logits = model(X)
        loss = loss_fn(d_logits, Yd) + loss_fn(c_logits, Yc)
        opt.zero_grad(); loss.backward(); opt.step()
        if ep == 1 or ep % 50 == 0:
            d_acc = (d_logits.argmax(-1) == Yd).float().mean().item()
            c_acc = (c_logits.argmax(-1) == Yc).float().mean().item()
            print(f"epoch {ep:3d}: loss={loss.item():.4f}  digit_acc={d_acc:.3f} carry_acc={c_acc:.3f}")
    # 最终全空间准确率
    d_acc = (model(X)[0].argmax(-1) == Yd).float().mean().item()
    c_acc = (model(X)[1].argmax(-1) == Yc).float().mean().item()
    print(f"\n个位加法器 参数量={n_params:,}  最终 digit_acc={d_acc:.3f} carry_acc={c_acc:.3f}")
    return model


# ===== 模块②：进位路由（确定性程序，不学习）=====
def add(a_str, b_str, adder):
    """任意长度加法：逐位调用 adder.predict，进位由程序传递。"""
    n = max(len(a_str), len(b_str))
    a, b = a_str.zfill(n), b_str.zfill(n)
    out, carry = [], 0
    for i in range(n - 1, -1, -1):                    # 低位 → 高位
        d, carry = adder.predict(int(a[i]), int(b[i]), carry)
        out.append(d)
    if carry:
        out.append(carry)
    return ''.join(str(d) for d in reversed(out))


# ===== 模块②'：纯神经进位推理（替代程序化 add）=====
@torch.no_grad()
def neural_add(a_str, b_str, adder, carry_prop):
    """纯神经加法：DigitAdder + CarryPropagator，零程序化进位传递。"""
    n = max(len(a_str), len(b_str))
    a, b = a_str.zfill(n), b_str.zfill(n)
    h = torch.zeros(carry_prop.gru.hidden_size)
    out = []
    for i in range(n - 1, -1, -1):                    # 低位 → 高位
        cin = carry_prop.cin_from_h(h).argmax(-1).item()
        digit, cout = adder.predict(int(a[i]), int(b[i]), cin)
        out.append(digit)
        h = carry_prop.step(cout, h)
    # 最高位是否还有进位
    cin = carry_prop.cin_from_h(h).argmax(-1).item()
    if cin:
        out.append(1)
    return ''.join(str(d) for d in reversed(out))


# ===== 评估纯神经组合系统在任意长度上 =====
@torch.no_grad()
def evaluate_neural_lengths(adder, carry_prop, n_per_len=500, seed=12345, lengths=None):
    """纯神经组合系统在任意长度上的准确率。"""
    if lengths is None:
        lengths = [1, 2, 3, 4, 5, 8, 12, 16, 20]
    rng = random.Random(seed)
    results = {}
    print("\n=== 纯神经组合系统：任意长度加法 ===")
    for L in lengths:
        ok = 0
        for _ in range(n_per_len):
            a = rng.randrange(10 ** L)
            b = rng.randrange(10 ** L)
            ok += int(neural_add(str(a), str(b), adder, carry_prop)) == a + b
        acc = ok / n_per_len
        results[L] = acc
        print(f"  {L:2d} 位: {acc:.4f}")
    return results


# ===== 评估组合系统在任意长度上 =====
@torch.no_grad()
def evaluate_lengths(adder, n_per_len=500, seed=12345):
    rng = random.Random(seed)
    print("\n=== 组合系统：任意长度加法 ===")
    for L in [1, 2, 3, 4, 5, 8, 12, 16, 20]:
        ok = 0
        for _ in range(n_per_len):
            a = rng.randrange(10 ** L)
            b = rng.randrange(10 ** L)
            ok += int(add(str(a), str(b), adder)) == a + b
        print(f"  {L:2d} 位: {ok / n_per_len:.4f}")


def selfcheck():
    x = encode_abc(3, 4, 0)
    assert x.sum().item() == 3
    assert x[3].item() == 1 and x[14].item() == 1 and x[20].item() == 1
    print("selfcheck ok")


if __name__ == "__main__":
    selfcheck()
    print("\n===== 阶段 1：训练个位加法器 =====")
    adder = train_adder()
    print("\n===== 阶段 2：训练进位传播器 =====")
    carry_prop = train_carry_prop()
    print("\n===== 阶段 3：组合评估 =====")
    evaluate_neural_lengths(adder, carry_prop, n_per_len=500)
    print("\n===== 对照：程序化进位 =====")
    evaluate_lengths(adder, n_per_len=500)
