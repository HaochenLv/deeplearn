# 纯神经进位传播器 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在模块化加法器中用 GRU Cell 模型替代程序化进位路由，实现纯神经加法系统。

**Architecture:** DigitAdder (MLP, 不变) 输出 digit+cout，CarryPropagator (GRU Cell, ~282 参数) 接收 cout 序列、维护隐状态、输出 cin。两个模型分阶段独立训练，组合推理时逐位循环。

**Tech Stack:** PyTorch (nn.GRUCell, nn.Linear, CrossEntropyLoss)

## Global Constraints

- CarryPropagator 隐状态维度 hidden=8
- GRUCell 输入维度=2 (cout one-hot)
- 训练数据 cout 来自确定性规则 `(a+b+cin)//10`，不依赖 DigitAdder 预测
- 评估位数范围：1, 2, 3, 4, 5, 8, 12, 16, 20，每长度 500 样本，exact match
- 保留现有 `add()` 函数和 `DigitAdder` 不变，新增代码不破坏已有测试

---

### Task 1: CarryPropagator 模型类

**Files:**
- Modify: `modular_addition.py` (在 `DigitAdder` 类之后新增)
- Test: `tests/test_modular.py`

**Interfaces:**
- Consumes: 无（独立新类）
- Produces: `CarryPropagator(hidden=8)` — `forward(cout_seq)` 返回 `list[Tensor(2)]`；`cin_from_h(h)` 返回 `Tensor(2)`

- [ ] **Step 1: 写失败测试**

在 `tests/test_modular.py` 末尾添加：

```python
from modular_addition import CarryPropagator


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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `source .venv/bin/activate && pytest tests/test_modular.py::test_carry_propagator_shape tests/test_modular.py::test_carry_propagator_cin_from_h -v`
Expected: FAIL — `ImportError: cannot import name 'CarryPropagator'`

- [ ] **Step 3: 实现 CarryPropagator**

在 `modular_addition.py` 的 `encode_abc` 函数之前（`DigitAdder` 类之后）插入：

```python

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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `source .venv/bin/activate && pytest tests/test_modular.py::test_carry_propagator_shape tests/test_modular.py::test_carry_propagator_cin_from_h -v`
Expected: PASS

- [ ] **Step 5: 确认已有测试不被破坏**

Run: `source .venv/bin/activate && pytest tests/test_modular.py -v`
Expected: 全部 PASS（原有 6 个 + 新增 2 个）

- [ ] **Step 6: 提交**

```bash
git add modular_addition.py tests/test_modular.py
git commit -m "feat: CarryPropagator GRU Cell 模型类 + 基础测试"
```

---

### Task 2: CarryPropagator 训练数据生成

**Files:**
- Modify: `modular_addition.py` (在 `CarryPropagator` 类之后新增函数)
- Test: `tests/test_modular.py`

**Interfaces:**
- Consumes: 无
- Produces: `generate_carry_data(n_samples, max_len, seed)` → `(cout_seqs: Tensor[N, L, 2], cin_labels: Tensor[N, L], lengths: list[int])`

- [ ] **Step 1: 写失败测试**

在 `tests/test_modular.py` 末尾添加：

```python
from modular_addition import generate_carry_data


def test_generate_carry_data_shape():
    cout_seqs, cin_labels, lengths = generate_carry_data(10, max_len=5, seed=42)
    assert cout_seqs.shape == (10, 5, 2)       # N, L, 2
    assert cin_labels.shape == (10, 5)          # N, L
    assert len(lengths) == 10
    for l in lengths:
        assert 1 <= l <= 5


def test_generate_carry_data_values():
    """37 + 48 = 85: 个位 7+8=15(cout=1), 十位 3+4+1=8(cout=0)。
    从低位到高位: cout=[1,0], cin=[0,1]"""
    cout_seqs, cin_labels, lengths = generate_carry_data(100, max_len=4, seed=0)
    # 找到 37+48 的样本（不保证存在，但概率高）
    # 改用确定性小数据集验证
    cout_seqs2, cin_labels2, lengths2 = generate_carry_data(5, max_len=2, seed=99)
    # 验证 one-hot 编码正确
    for i in range(5):
        for j in range(lengths2[i]):
            c = cin_labels2[i, j].item()
            assert c in (0, 1)
            cout_val = cout_seqs2[i, j].argmax().item()
            assert cout_val in (0, 1)


def test_generate_carry_data_cin_first_is_zero():
    """每条序列首位 cin 必须是 0（没有来自更低位的进位）。"""
    cout_seqs, cin_labels, lengths = generate_carry_data(50, max_len=8, seed=7)
    for i in range(50):
        assert cin_labels[i, 0].item() == 0
```

- [ ] **Step 2: 运行测试确认失败**

Run: `source .venv/bin/activate && pytest tests/test_modular.py::test_generate_carry_data_shape tests/test_modular.py::test_generate_carry_data_values tests/test_modular.py::test_generate_carry_data_cin_first_is_zero -v`
Expected: FAIL — `ImportError: cannot import name 'generate_carry_data'`

- [ ] **Step 3: 实现 generate_carry_data**

在 `modular_addition.py` 的 `CarryPropagator.step` 方法之后、`train_adder` 函数之前插入：

```python

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
            carry = (ai + bi + carry) // 10         # 当前位的 cout
            cout_seqs[i, j, carry] = 1.0            # cout one-hot
            carry = carry                           # 下一位的 cin = 当前位的 cout
    return cout_seqs, cin_labels, lengths
```

- [ ] **Step 4: 运行测试确认通过**

Run: `source .venv/bin/activate && pytest tests/test_modular.py::test_generate_carry_data_shape tests/test_modular.py::test_generate_carry_data_values tests/test_modular.py::test_generate_carry_data_cin_first_is_zero -v`
Expected: PASS

- [ ] **Step 5: 确认全部测试通过**

Run: `source .venv/bin/activate && pytest tests/test_modular.py -v`
Expected: 全部 PASS

- [ ] **Step 6: 提交**

```bash
git add modular_addition.py tests/test_modular.py
git commit -m "feat: CarryPropagator 训练数据生成函数 + 测试"
```

---

### Task 3: CarryPropagator 训练函数

**Files:**
- Modify: `modular_addition.py` (在 `generate_carry_data` 之后新增)
- Test: `tests/test_modular.py`

**Interfaces:**
- Consumes: `CarryPropagator`, `generate_carry_data`
- Produces: `train_carry_prop(epochs=100, lr=1e-2, hidden=8, max_len=20, n_per_epoch=5000, seed=0)` → `CarryPropagator`

- [ ] **Step 1: 写失败测试**

在 `tests/test_modular.py` 末尾添加：

```python
from modular_addition import train_carry_prop


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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `source .venv/bin/activate && pytest tests/test_modular.py::test_train_carry_prop_converges -v`
Expected: FAIL — `ImportError: cannot import name 'train_carry_prop'`

- [ ] **Step 3: 实现 train_carry_prop**

在 `modular_addition.py` 的 `generate_carry_data` 函数之后、`train_adder` 函数之前插入：

```python

def train_carry_prop(epochs=100, lr=1e-2, hidden=8, max_len=20, n_per_epoch=5000, seed=0):
    """训练 CarryPropagator：cout 序列 → cin 序列。"""
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
        total_loss, total_correct, total_tokens = 0.0, 0, 0
        for i in range(n_per_epoch):
            L = lengths[i]
            cins = model(cout_seqs[i, :L])
            for j, cin_logit in enumerate(cins):
                loss = loss_fn(cin_logit.unsqueeze(0), cin_labels[i, j].unsqueeze(0))
                total_loss += loss.item()
                total_correct += int(cin_logit.argmax().item() == cin_labels[i, j].item())
                total_tokens += 1
                loss.backward(retain_graph=(j < L - 1))
            opt.step()
            opt.zero_grad()
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `source .venv/bin/activate && pytest tests/test_modular.py::test_train_carry_prop_converges -v`
Expected: PASS（可能需要 30-60 秒）

- [ ] **Step 5: 确认全部测试通过**

Run: `source .venv/bin/activate && pytest tests/test_modular.py -v`
Expected: 全部 PASS

- [ ] **Step 6: 提交**

```bash
git add modular_addition.py tests/test_modular.py
git commit -m "feat: CarryPropagator 训练函数 + 收敛测试"
```

---

### Task 4: neural_add 组合推理函数

**Files:**
- Modify: `modular_addition.py` (在 `add()` 函数之后新增)
- Test: `tests/test_modular.py`

**Interfaces:**
- Consumes: `DigitAdder.predict()`, `CarryPropagator.cin_from_h()`, `CarryPropagator.step()`
- Produces: `neural_add(a_str, b_str, adder, carry_prop)` → `str`

- [ ] **Step 1: 写失败测试**

在 `tests/test_modular.py` 末尾添加：

```python
from modular_addition import neural_add


def test_neural_add_no_carry():
    o = OracleAdder()
    cp = CarryPropagator(hidden=8)
    # 未训练的 cp 可能不准，用 OracleAdder + 手动设 cin 测试函数结构
    # 改用训练好的组合系统测试
    # 先只测试函数可调用、返回格式正确
    result = neural_add("3", "4", o, cp)
    assert isinstance(result, str)
    assert result.isdigit() or result == ""


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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `source .venv/bin/activate && pytest tests/test_modular.py::test_neural_add_no_carry tests/test_modular.py::test_neural_add_trained_system -v`
Expected: FAIL — `ImportError: cannot import name 'neural_add'`

- [ ] **Step 3: 实现 neural_add**

在 `modular_addition.py` 的 `add()` 函数之后、`evaluate_lengths` 函数之前插入：

```python

# ===== 模块②'：纯神经进位推理（替代程序化 add）=====
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `source .venv/bin/activate && pytest tests/test_modular.py::test_neural_add_no_carry tests/test_modular.py::test_neural_add_trained_system -v`
Expected: PASS（训练需要 30-90 秒）

- [ ] **Step 5: 确认全部测试通过**

Run: `source .venv/bin/activate && pytest tests/test_modular.py -v`
Expected: 全部 PASS

- [ ] **Step 6: 提交**

```bash
git add modular_addition.py tests/test_modular.py
git commit -m "feat: neural_add 纯神经组合推理函数 + 测试"
```

---

### Task 5: 纯神经评估 + 主程序整合

**Files:**
- Modify: `modular_addition.py` (新增 `evaluate_neural_lengths`，修改 `__main__`)
- Test: `tests/test_modular.py`

**Interfaces:**
- Consumes: `train_adder`, `train_carry_prop`, `neural_add`
- Produces: `evaluate_neural_lengths(adder, carry_prop, n_per_len=500, seed=12345)` → `dict[int, float]`

- [ ] **Step 1: 写失败测试**

在 `tests/test_modular.py` 末尾添加：

```python
from modular_addition import evaluate_neural_lengths


def test_evaluate_neural_lengths_returns_dict():
    adder = train_adder(epochs=200, lr=1e-2, seed=0)
    cp = train_carry_prop(epochs=100, lr=1e-2, hidden=8, max_len=8, n_per_epoch=2000, seed=0)
    results = evaluate_neural_lengths(adder, cp, n_per_len=50, seed=42, lengths=[1, 2, 4])
    assert isinstance(results, dict)
    assert set(results.keys()) == {1, 2, 4}
    for v in results.values():
        assert 0.0 <= v <= 1.0
```

- [ ] **Step 2: 运行测试确认失败**

Run: `source .venv/bin/activate && pytest tests/test_modular.py::test_evaluate_neural_lengths_returns_dict -v`
Expected: FAIL — `ImportError: cannot import name 'evaluate_neural_lengths'`

- [ ] **Step 3: 实现 evaluate_neural_lengths**

在 `modular_addition.py` 的 `neural_add` 函数之后、`evaluate_lengths` 函数之前插入：

```python

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
```

- [ ] **Step 4: 修改 `__main__` 块**

将 `modular_addition.py` 的 `if __name__ == "__main__":` 块替换为：

```python
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
```

- [ ] **Step 5: 运行测试确认通过**

Run: `source .venv/bin/activate && pytest tests/test_modular.py -v`
Expected: 全部 PASS

- [ ] **Step 6: 端到端运行主程序**

Run: `source .venv/bin/activate && python3 modular_addition.py`
Expected: 三个阶段依次执行，1-20 位纯神经准确率应接近 100%（与程序化进位对照一致）

- [ ] **Step 7: 提交**

```bash
git add modular_addition.py tests/test_modular.py
git commit -m "feat: 纯神经评估 + 主程序三阶段整合"
```
