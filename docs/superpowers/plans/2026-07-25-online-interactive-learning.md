# 在线交互式学习 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现幼儿式在线交互学习系统——空模型从零开始，外部出题、教师逐位纠正、学生边推理边学习，课程从1位到多位逐步升级。

**Architecture:** StudentAgent 封装空初始化的 DigitAdder + CarryPropagator + 统一 optimizer；Teacher 是确定性数学规则；CurriculumEnv 按课程出题并检查毕业。solve_and_learn 逐位推理→纠正→即时更新权重。

**Tech Stack:** PyTorch (nn.Module, Adam, CrossEntropyLoss), matplotlib (学习曲线可视化)

## Global Constraints

- StudentAgent 内 student_adder = DigitAdder(hidden=32), student_carry = CarryPropagator(hidden=8)
- Teacher 是确定性规则 `(a+b+cin)%10` / `(a+b+cin)//10`，不是神经网络
- 逐步即时学习：每做一步 backward + optimizer.step()
- 学生用自己预测的 cin 做加法，loss 用教师正确标签
- 隐状态用学生的 cout 更新，carry_true 由教师维护
- 课程从 1 位开始，毕业条件 exact_match ≥ 0.95
- 复用 modular_addition.py 中的 DigitAdder, CarryPropagator, encode_abc, IN_DIM
- 不修改 modular_addition.py

---

### Task 1: StudentAgent 类

**Files:**
- Create: `online_addition.py`
- Create: `tests/test_online.py`

**Interfaces:**
- Consumes: `modular_addition.DigitAdder`, `modular_addition.CarryPropagator`, `modular_addition.encode_abc`, `modular_addition.IN_DIM`
- Produces: `StudentAgent(lr=1e-2)` — `solve_and_learn(a_str: str, b_str: str) → (answer: str, correct: bool, total_loss: float)`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_online.py`：

```python
import torch
from online_addition import StudentAgent


def test_student_agent_init():
    agent = StudentAgent(lr=1e-2)
    assert hasattr(agent, 'student_adder')
    assert hasattr(agent, 'student_carry')
    assert hasattr(agent, 'optimizer')
    # 参数量应与离线训练一致
    n = sum(p.numel() for p in agent.parameters())
    assert n == 2188 + 306  # DigitAdder(32) + CarryPropagator(8)


def test_solve_and_learn_returns():
    agent = StudentAgent(lr=1e-2)
    answer, correct, loss = agent.solve_and_learn("3", "4")
    assert isinstance(answer, str)
    assert isinstance(correct, bool)
    assert isinstance(loss, float)


def test_solve_and_learn_1digit():
    """1 位加法经过少量学习后应能正确。"""
    agent = StudentAgent(lr=1e-2)
    # 反复练习 3+4
    for _ in range(50):
        agent.solve_and_learn("3", "4")
    answer, correct, _ = agent.solve_and_learn("3", "4")
    assert correct, f"3+4={answer}, expected 7"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `source .venv/bin/activate && pytest tests/test_online.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'online_addition'`

- [ ] **Step 3: 实现 StudentAgent**

创建 `online_addition.py`：

```python
"""在线交互式学习：幼儿式加法学习系统。

空模型从零开始，外部出题、教师逐位纠正、学生边推理边学习。
课程从1位到多位逐步升级，模拟幼儿学习过程。
"""
import random
import torch
import torch.nn as nn
from torch.optim import Adam
from modular_addition import DigitAdder, CarryPropagator, encode_abc, IN_DIM


class StudentAgent(nn.Module):
    """学生加法器：空模型，边推理边学习。

    封装 DigitAdder + CarryPropagator，统一 optimizer。
    solve_and_learn 逐位推理→教师纠正→即时更新权重。
    """
    def __init__(self, lr=1e-2):
        super().__init__()
        self.student_adder = DigitAdder(hidden=32)
        self.student_carry = CarryPropagator(hidden=8)
        self.optimizer = Adam(
            list(self.student_adder.parameters()) +
            list(self.student_carry.parameters()),
            lr=lr
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def solve_and_learn(self, a_str, b_str):
        """边推理边学习：逐位推理，教师即时纠正，权重即时更新。

        Args:
            a_str: 第一个数（字符串）
            b_str: 第二个数（字符串）
        Returns:
            (answer, correct, total_loss)
            - answer: 学生的答案字符串
            - correct: 是否完全正确
            - total_loss: 本题总 loss
        """
        n = max(len(a_str), len(b_str))
        a, b = a_str.zfill(n), b_str.zfill(n)
        h = torch.zeros(self.student_carry.gru.hidden_size)
        out = []
        total_loss = 0.0
        carry_true = 0  # 教师维护的正确进位

        for i in range(n - 1, -1, -1):              # 低位 → 高位
            ai, bi = int(a[i]), int(b[i])

            # ---- 学生推理 ----
            cin_logits = self.student_carry.cin_from_h(h)       # [2]
            cin_pred = cin_logits.argmax(-1).item()

            x = encode_abc(ai, bi, cin_pred)                    # [22]
            d_logits, c_logits = self.student_adder(x.unsqueeze(0))  # [1,10], [1,2]
            digit_pred = d_logits.argmax(-1).item()
            cout_pred = c_logits.argmax(-1).item()

            # ---- 教师纠正 ----
            cin_true = carry_true
            digit_true = (ai + bi + carry_true) % 10
            cout_true = (ai + bi + carry_true) // 10

            # ---- 即时学习 ----
            loss = (self.loss_fn(cin_logits.unsqueeze(0), torch.tensor([cin_true])) +
                    self.loss_fn(d_logits, torch.tensor([digit_true])) +
                    self.loss_fn(c_logits, torch.tensor([cout_true])))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            # ---- 记录输出，更新隐状态 ----
            out.append(digit_pred)
            h = self.student_carry.step(cout_pred, h)           # 用学生的 cout
            carry_true = cout_true                               # 教师进位给下一位

        # 最高位进位
        cin_logits = self.student_carry.cin_from_h(h)
        cin_pred = cin_logits.argmax(-1).item()
        if cin_pred:
            out.append(1)

        answer = ''.join(str(d) for d in reversed(out))
        correct = (answer == str(int(a) + int(b)))
        return answer, correct, total_loss
```

- [ ] **Step 4: 运行测试确认通过**

Run: `source .venv/bin/activate && pytest tests/test_online.py -v`
Expected: PASS

- [ ] **Step 5: 确认不破坏已有测试**

Run: `source .venv/bin/activate && pytest tests/test_modular.py -v`
Expected: 全部 PASS（未修改 modular_addition.py）

- [ ] **Step 6: 提交**

```bash
git add online_addition.py tests/test_online.py
git commit -m "feat: StudentAgent 在线交互式学习类 + 基础测试"
```

---

### Task 2: CurriculumEnv 类

**Files:**
- Modify: `online_addition.py`
- Modify: `tests/test_online.py`

**Interfaces:**
- Consumes: `StudentAgent.solve_and_learn()`
- Produces: `CurriculumEnv(max_digits, graduate_acc, samples_per_test, seed)` — `next_problem() → (a_str, b_str)`, `check_graduation(agent) → (graduated: bool, acc: float)`, `current_level: int`

- [ ] **Step 1: 写失败测试**

在 `tests/test_online.py` 末尾添加：

```python
from online_addition import CurriculumEnv


def test_curriculum_env_init():
    env = CurriculumEnv(max_digits=5, seed=0)
    assert env.current_level == 1


def test_curriculum_env_next_problem():
    env = CurriculumEnv(max_digits=3, seed=42)
    a, b = env.next_problem()
    assert len(a) <= 1 and len(b) <= 1  # level=1 → 1位数
    assert a.isdigit() and b.isdigit()


def test_curriculum_env_level_up():
    env = CurriculumEnv(max_digits=3, graduate_acc=0.95, samples_per_test=20, seed=0)
    # 用 OracleAgent 模拟完美学生
    class OracleAgent:
        def solve_and_learn(self, a_str, b_str):
            answer = str(int(a_str) + int(b_str))
            return answer, True, 0.0
    oracle = OracleAgent()
    graduated, acc = env.check_graduation(oracle)
    assert graduated is True
    assert acc == 1.0
    assert env.current_level == 2  # 升级了


def test_curriculum_env_no_level_up():
    env = CurriculumEnv(max_digits=3, graduate_acc=0.95, samples_per_test=20, seed=0)
    # 用 RandomAgent 模拟差学生
    class RandomAgent:
        def solve_and_learn(self, a_str, b_str):
            return "0", False, 1.0
    bad = RandomAgent()
    graduated, acc = env.check_graduation(bad)
    assert graduated is False
    assert env.current_level == 1  # 没升级
```

- [ ] **Step 2: 运行测试确认失败**

Run: `source .venv/bin/activate && pytest tests/test_online.py::test_curriculum_env_init -v`
Expected: FAIL — `ImportError: cannot import name 'CurriculumEnv'`

- [ ] **Step 3: 实现 CurriculumEnv**

在 `online_addition.py` 的 `StudentAgent` 类之后添加：

```python


class CurriculumEnv:
    """课程环境：从简单到复杂出题，学生掌握后升级。

    从 1 位加法开始，准确率 ≥ graduate_acc 后升级到更多位。
    check_graduation 中 agent 也在学习（solve_and_learn 有 side effect），
    这是设计意图——没有"纯测试"模式，学习永不停止。
    """
    def __init__(self, max_digits=5, graduate_acc=0.95,
                 samples_per_test=100, seed=0):
        self.max_digits = max_digits
        self.graduate_acc = graduate_acc
        self.samples_per_test = samples_per_test
        self.rng = random.Random(seed)
        self.current_level = 1

    def next_problem(self):
        """出题：当前难度级别的随机加法。"""
        L = self.current_level
        a = self.rng.randrange(10 ** L)
        b = self.rng.randrange(10 ** L)
        return str(a), str(b)

    def check_graduation(self, agent):
        """检查学生是否毕业，可以升级。

        Returns:
            (graduated, acc) — 是否毕业，当前准确率
        """
        correct = 0
        for _ in range(self.samples_per_test):
            a_str, b_str = self.next_problem()
            _, is_correct, _ = agent.solve_and_learn(a_str, b_str)
            correct += is_correct
        acc = correct / self.samples_per_test
        if acc >= self.graduate_acc:
            self.current_level = min(self.current_level + 1, self.max_digits)
            return True, acc
        return False, acc
```

- [ ] **Step 4: 运行测试确认通过**

Run: `source .venv/bin/activate && pytest tests/test_online.py -v`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add online_addition.py tests/test_online.py
git commit -m "feat: CurriculumEnv 课程环境类 + 测试"
```

---

### Task 3: train_student 训练主循环

**Files:**
- Modify: `online_addition.py`
- Modify: `tests/test_online.py`

**Interfaces:**
- Consumes: `StudentAgent`, `CurriculumEnv`
- Produces: `train_student(max_digits, problems_per_level, check_interval, lr, seed) → (agent, history)` where history is `list[dict]` with keys `level, step, correct, loss, answer, expected`

- [ ] **Step 1: 写失败测试**

在 `tests/test_online.py` 末尾添加：

```python
from online_addition import train_student


def test_train_student_1digit():
    """1 位加法应能在少量练习后学会。"""
    agent, history = train_student(
        max_digits=1, problems_per_level=200,
        check_interval=50, lr=1e-2, seed=0
    )
    # 检查 history 格式
    assert len(history) > 0
    assert 'level' in history[0]
    assert 'correct' in history[0]
    assert 'loss' in history[0]
    # 1 位加法最终应接近 100%
    last_20 = [h['correct'] for h in history[-20:]]
    acc = sum(last_20) / len(last_20)
    assert acc > 0.8, f"1-digit acc={acc:.3f}, expected > 0.8"


def test_train_student_curriculum_progress():
    """课程应从 level 1 逐步升级。"""
    agent, history = train_student(
        max_digits=2, problems_per_level=300,
        check_interval=50, lr=1e-2, seed=0
    )
    levels_seen = set(h['level'] for h in history)
    assert 1 in levels_seen  # 至少从 level 1 开始
```

- [ ] **Step 2: 运行测试确认失败**

Run: `source .venv/bin/activate && pytest tests/test_online.py::test_train_student_1digit -v`
Expected: FAIL — `ImportError: cannot import name 'train_student'`

- [ ] **Step 3: 实现 train_student**

在 `online_addition.py` 的 `CurriculumEnv` 类之后添加：

```python


def train_student(max_digits=5, problems_per_level=500,
                  check_interval=50, lr=1e-2, seed=0):
    """训练学生：课程出题，边推理边学习。

    Args:
        max_digits: 最大位数
        problems_per_level: 每个级别最多做多少题
        check_interval: 每隔多少题检查一次毕业
        lr: 学习率
        seed: 随机种子
    Returns:
        (agent, history) — 训练好的学生，学习历史
    """
    torch.manual_seed(seed)
    agent = StudentAgent(lr=lr)
    env = CurriculumEnv(max_digits=max_digits, seed=seed)
    history = []

    while env.current_level <= max_digits:
        level_start = len(history)
        for step in range(problems_per_level):
            a_str, b_str = env.next_problem()
            answer, correct, loss = agent.solve_and_learn(a_str, b_str)
            expected = str(int(a_str) + int(b_str))
            history.append({
                'level': env.current_level,
                'step': step,
                'correct': correct,
                'loss': loss,
                'answer': answer,
                'expected': expected,
            })

            if (step + 1) % check_interval == 0:
                graduated, acc = env.check_graduation(agent)
                n_done = len(history) - level_start
                print(f"Level {env.current_level}, {n_done}题: acc={acc:.3f}"
                      + (" → 毕业升级!" if graduated else ""))
                if graduated:
                    break
        else:
            # 用完了 problems_per_level 还没毕业
            print(f"Level {env.current_level}: 未毕业，继续下一轮")

        if env.current_level > max_digits:
            break

    return agent, history
```

- [ ] **Step 4: 运行测试确认通过**

Run: `source .venv/bin/activate && pytest tests/test_online.py::test_train_student_1digit tests/test_online.py::test_train_student_curriculum_progress -v`
Expected: PASS（可能需要 10-30 秒）

- [ ] **Step 5: 确认全部测试通过**

Run: `source .venv/bin/activate && pytest tests/test_online.py -v`
Expected: 全部 PASS

- [ ] **Step 6: 提交**

```bash
git add online_addition.py tests/test_online.py
git commit -m "feat: train_student 课程训练主循环 + 测试"
```

---

### Task 4: 评估 + 学习曲线可视化 + __main__

**Files:**
- Modify: `online_addition.py`
- Modify: `tests/test_online.py`

**Interfaces:**
- Consumes: `StudentAgent`, `train_student`
- Produces: `evaluate_agent(agent, max_digits, n_per_len, seed) → dict[int, float]`, `plot_learning_curve(history, save_path)`, `__main__` 块

- [ ] **Step 1: 写失败测试**

在 `tests/test_online.py` 末尾添加：

```python
from online_addition import evaluate_agent


def test_evaluate_agent_returns_dict():
    agent, _ = train_student(max_digits=1, problems_per_level=100,
                             check_interval=50, lr=1e-2, seed=0)
    results = evaluate_agent(agent, max_digits=1, n_per_len=20, seed=42)
    assert isinstance(results, dict)
    assert 1 in results
    assert 0.0 <= results[1] <= 1.0
```

- [ ] **Step 2: 运行测试确认失败**

Run: `source .venv/bin/activate && pytest tests/test_online.py::test_evaluate_agent_returns_dict -v`
Expected: FAIL — `ImportError: cannot import name 'evaluate_agent'`

- [ ] **Step 3: 实现 evaluate_agent 和 plot_learning_curve**

在 `online_addition.py` 的 `train_student` 函数之后添加：

```python


@torch.no_grad()
def evaluate_agent(agent, max_digits=5, n_per_len=200, seed=12345):
    """评估学生模型在各位数上的准确率（纯推理，不学习）。"""
    rng = random.Random(seed)
    results = {}
    agent.eval()
    for L in [1, 2, 3, 4, 5, 8, 12, 16, 20]:
        if L > max_digits:
            break
        ok = 0
        for _ in range(n_per_len):
            a = rng.randrange(10 ** L)
            b = rng.randrange(10 ** L)
            # 纯推理模式：用 predict 而非 solve_and_learn
            n = max(len(str(a)), len(str(b)))
            a_s, b_s = str(a).zfill(n), str(b).zfill(n)
            h = torch.zeros(agent.student_carry.gru.hidden_size)
            out = []
            for i in range(n - 1, -1, -1):
                cin = agent.student_carry.cin_from_h(h).argmax(-1).item()
                digit, cout = agent.student_adder.predict(int(a_s[i]), int(b_s[i]), cin)
                out.append(digit)
                h = agent.student_carry.step(cout, h)
            cin = agent.student_carry.cin_from_h(h).argmax(-1).item()
            if cin:
                out.append(1)
            answer = ''.join(str(d) for d in reversed(out))
            ok += int(answer == str(a + b))
        results[L] = ok / n_per_len
    agent.train()
    return results


def plot_learning_curve(history, save_path=None):
    """绘制学习曲线：loss 和 accuracy，按 level 分色。"""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    levels = sorted(set(h['level'] for h in history))
    colors = plt.cm.tab10(range(len(levels)))

    for level, color in zip(levels, colors):
        data = [h for h in history if h['level'] == level]
        steps = range(len(data))
        # 平滑 loss
        losses = [h['loss'] for h in data]
        window = min(20, len(losses))
        if window > 1:
            smooth = []
            for i in range(len(losses)):
                start = max(0, i - window + 1)
                smooth.append(sum(losses[start:i+1]) / (i - start + 1))
            losses = smooth
        ax1.plot(steps, losses, color=color, label=f'{level}位', alpha=0.8)

        # 准确率（累积）
        corrects = [h['correct'] for h in data]
        cum_acc = []
        for i in range(len(corrects)):
            cum_acc.append(sum(corrects[:i+1]) / (i+1))
        ax2.plot(steps, cum_acc, color=color, label=f'{level}位', alpha=0.8)

    ax1.set_ylabel('Loss (smoothed)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.set_ylabel('Cumulative Accuracy')
    ax2.set_xlabel('Problems within level')
    ax2.axhline(y=0.95, color='red', linestyle='--', alpha=0.5, label='毕业线')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig.suptitle('幼儿式在线学习曲线')
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"学习曲线已保存到 {save_path}")
    return fig
```

- [ ] **Step 4: 添加 __main__ 块**

在 `online_addition.py` 末尾添加：

```python

if __name__ == "__main__":
    print("===== 幼儿式在线学习：从零开始学加法 =====\n")
    agent, history = train_student(
        max_digits=5, problems_per_level=500,
        check_interval=50, lr=1e-2, seed=0
    )
    print("\n===== 最终评估 =====")
    results = evaluate_agent(agent, max_digits=5, n_per_len=200, seed=999)
    for L, acc in results.items():
        print(f"  {L:2d} 位: {acc:.4f}")

    print("\n===== 学习曲线 =====")
    plot_learning_curve(history, save_path="outputs/online_learning_curve.png")

    print("\n===== 与离线训练对比 =====")
    from modular_addition import train_adder, train_carry_prop, evaluate_neural_lengths
    adder = train_adder(epochs=200, lr=1e-2, seed=0)
    cp = train_carry_prop(epochs=100, lr=1e-2, hidden=8, max_len=20, n_per_epoch=5000, seed=0)
    offline_results = evaluate_neural_lengths(adder, cp, n_per_len=200, seed=999, lengths=list(results.keys()))
    print("\n在线 vs 离线:")
    for L in results:
        print(f"  {L:2d} 位: 在线={results[L]:.4f}  离线={offline_results.get(L, 0):.4f}")
```

- [ ] **Step 5: 运行测试确认通过**

Run: `source .venv/bin/activate && pytest tests/test_online.py -v`
Expected: 全部 PASS

- [ ] **Step 6: 端到端运行主程序**

Run: `source .venv/bin/activate && python3 -u online_addition.py`
Expected: 课程从1位逐步升级到5位，最终评估各位准确率，输出学习曲线图

- [ ] **Step 7: 提交**

```bash
git add online_addition.py tests/test_online.py
git commit -m "feat: 评估 + 学习曲线可视化 + __main__ 端到端流程"
```
