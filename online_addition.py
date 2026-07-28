"""在线交互式学习：幼儿式加法学习系统。

空模型从零开始，外部出题、教师逐位纠正、学生边推理边学习。
课程从1位到多位逐步升级，模拟幼儿学习过程。

核心设计：3 个模型配合 ——
  ① DigitAdder (推理)        : a+b+cin → digit+cout（前馈 MLP）
  ② CarryPropagator (推理)   : cout → cin 单步进位传递（前馈 MLP）
  ③ LearningMechanism (学习) : 独立参数集，通过 FA + delta 规则修改 ①② 的权重

进位传递说明：进位规则 cin_{t} = cout_{t-1} 本质是单步映射（恒等），
不依赖序列历史。因此用前馈模型（而非 GRU）实现：FA 能完整训练它，
且单步映射天然与序列长度无关 → N 位泛化。
"""
import random
import torch
import torch.nn as nn
from modular_addition import DigitAdder, encode_abc, IN_DIM
from learning_mechanism import HebbianLearner, RewardLearner, CorrectionLearner


LEARNER_CLASSES = {
    'hebbian': HebbianLearner,
    'reward': RewardLearner,
    'correction': CorrectionLearner,
}


class FeedForwardCarry(nn.Module):
    """前馈进位传播器：cout → cin 单步映射。

    进位传递规则 cin_{t} = cout_{t-1} 是单步恒等映射，不依赖历史，
    用前馈 MLP 即可。FA 能完整训练（无 BPTT 也能收敛），
    且单步映射天然与序列长度无关。

    结构与 DigitAdder 对齐：[Linear, ReLU, Linear, ReLU] + cin_head。
    """
    def __init__(self, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.cin_head = nn.Linear(hidden, 2)   # 隐状态 → cin logits

    def forward(self, cout_oh):                          # cout_oh: [B, 2]
        return self.cin_head(self.net(cout_oh))          # [B, 2]

    @torch.no_grad()
    def predict(self, cout):
        """cout (int 0/1) → cin (int 0/1)。组合推理时使用。"""
        cout_oh = torch.zeros(2)
        cout_oh[cout] = 1.0
        logits = self.cin_head(self.net(cout_oh.unsqueeze(0)).squeeze(0))
        return logits.argmax(-1).item()


class StudentAgent(nn.Module):
    """学生加法器：推理模型（DigitAdder + CarryPropagator）+ 独立学习机制。

    推理模型负责做加法；学习机制负责学会做加法，有独立参数集。
    进位传递：前馈 CarryPropagator 学 cout → cin 单步映射。
    """
    def __init__(self, learning_mode='hebbian', adder_hidden=128, carry_hidden=16,
                 init_eta=0.1, init_gamma=0.0):
        super().__init__()
        self.student_adder = DigitAdder(hidden=adder_hidden)
        self.student_carry = FeedForwardCarry(hidden=carry_hidden)
        self.learning_mode = learning_mode

        # 独立学习机制
        learner_cls = LEARNER_CLASSES[learning_mode]
        self.learner = learner_cls(init_eta=init_eta, init_gamma=init_gamma,
                                   adder_hidden=adder_hidden, carry_hidden=carry_hidden)

    def solve_and_learn(self, a_str, b_str):
        """边推理边学习：逐位推理，学习机制即时更新权重。

        进位传递：前馈 CarryPropagator 学 cout → cin。
        训练用 teacher forcing（carry 输入 = 教师上一位 cout），
        保证 carry 干净地学到恒等映射 cout → cin。

        Args:
            a_str: 第一个数（字符串）
            b_str: 第二个数（字符串）
        Returns:
            (answer, correct, info)
        """
        n = max(len(a_str), len(b_str))
        a, b = a_str.zfill(n), b_str.zfill(n)
        out = []
        prev_cout_true = 0   # 最低位无进位输入（教师链）
        total_updates = 0
        last_cout_pred = 0   # 最高位的 cout（用于判断最终进位）

        for i in range(n - 1, -1, -1):              # 低位 → 高位
            ai, bi = int(a[i]), int(b[i])
            cin_true = prev_cout_true               # 当前位 cin = 上一位 cout

            # ---- 学生推理（no_grad，不参与 autograd）----
            with torch.no_grad():
                # CarryPropagator：从上一位 cout 预测当前 cin
                cout_oh = torch.zeros(2)
                cout_oh[prev_cout_true] = 1.0
                h1_c = self.student_carry.net[0](cout_oh.unsqueeze(0))
                h1_act_c = self.student_carry.net[1](h1_c)
                cin_logits = self.student_carry.cin_head(
                    self.student_carry.net[3](self.student_carry.net[2](h1_act_c)))
                cin_pred = cin_logits.argmax(-1).item()

                # DigitAdder：用教师正确进位编码输入
                x = encode_abc(ai, bi, cin_true)                  # [22]
                h1 = self.student_adder.net[0](x.unsqueeze(0))
                h1_act = self.student_adder.net[1](h1)
                d_logits, c_logits = self.student_adder(x.unsqueeze(0))
                digit_pred = d_logits.argmax(-1).item()
                cout_pred = c_logits.argmax(-1).item()
            last_cout_pred = cout_pred

            # ---- 教师纠正 ----
            digit_true = (ai + bi + cin_true) % 10
            cout_true = (ai + bi + cin_true) // 10

            # ---- 学习机制更新权重 ----
            correction = {'digit_true': digit_true, 'cout_true': cout_true, 'cin_true': cin_true}
            if self.learning_mode == 'reward':
                digit_correct = (digit_pred == digit_true)
                cout_correct = (cout_pred == cout_true)
                reward = 1.0 if (digit_correct and cout_correct) else -1.0
                correction['reward'] = reward
                cin_reward = 1.0 if (cin_pred == cin_true) else -1.0
                carry_signal = {'cin_true': cin_true, 'reward': cin_reward}
            else:
                carry_signal = correction

            self.learner.learn_step_adder(self.student_adder, x, h1_act, signal=correction)
            self.learner.learn_step_carry(self.student_carry, cout_oh, h1_act_c, signal=carry_signal)
            total_updates += 1

            # ---- 补充训练 DigitAdder：另一种 cin ----
            other_cin = 1 - cin_true
            with torch.no_grad():
                x_other = encode_abc(ai, bi, other_cin)
                h1_other = self.student_adder.net[0](x_other.unsqueeze(0))
                h1_act_other = self.student_adder.net[1](h1_other)

            digit_true_o = (ai + bi + other_cin) % 10
            cout_true_o = (ai + bi + other_cin) // 10
            correction_o = {'digit_true': digit_true_o, 'cout_true': cout_true_o, 'cin_true': other_cin}
            if self.learning_mode == 'reward':
                correction_o['reward'] = 0.0  # 补充样本中性奖励
            self.learner.learn_step_adder(self.student_adder, x_other, h1_act_other, signal=correction_o)
            total_updates += 1

            # ---- 记录输出，进位传递 ----
            out.append(digit_pred)
            prev_cout_true = cout_true   # 教师进位链

        # ---- 最高位进位 ----
        if last_cout_pred:
            out.append(1)

        answer = ''.join(str(d) for d in reversed(out))
        correct = (answer == str(int(a) + int(b)))
        info = {'updates': total_updates, 'learning_mode': self.learning_mode}
        return answer, correct, info


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

        注意：检查过程中 agent 也在学习（solve_and_learn 有 side effect），
        这是设计意图——没有"纯测试"模式，学习永不停止。

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
            self.current_level = min(self.current_level + 1, self.max_digits + 1)
            return True, acc
        return False, acc


def train_student(max_digits=5, problems_per_level=500,
                  check_interval=50, learning_mode='hebbian', seed=0,
                  samples_per_test=100, max_rounds=10,
                  init_eta=0.1, init_gamma=0.0,
                  adder_hidden=128, carry_hidden=16,
                  consolidation_steps=0):
    """训练学生：课程出题，边推理边学习。

    Args:
        max_digits: 最大位数
        problems_per_level: 每个级别最多做多少题
        check_interval: 每隔多少题检查一次毕业
        learning_mode: 学习模式 ('hebbian'/'reward'/'correction')
        seed: 随机种子
        samples_per_test: 毕业检查时采样多少题
        max_rounds: 每个级别最多尝试几轮（防止无限循环）
        init_eta: 学习机制初始学习率
        init_gamma: 学习机制初始衰减系数
        adder_hidden: DigitAdder 隐层大小
        carry_hidden: CarryPropagator 隐层大小
        consolidation_steps: 毕业后巩固训练步数（0=不巩固）
    Returns:
        (agent, history) — 训练好的学生，学习历史
    """
    torch.manual_seed(seed)
    agent = StudentAgent(learning_mode=learning_mode,
                         init_eta=init_eta, init_gamma=init_gamma,
                         adder_hidden=adder_hidden, carry_hidden=carry_hidden)
    env = CurriculumEnv(max_digits=max_digits, seed=seed,
                        samples_per_test=samples_per_test)
    history = []

    while env.current_level <= max_digits:
        for round_num in range(1, max_rounds + 1):
            level_start = len(history)
            for step in range(problems_per_level):
                a_str, b_str = env.next_problem()
                answer, correct, info = agent.solve_and_learn(a_str, b_str)
                expected = str(int(a_str) + int(b_str))
                history.append({
                    'level': env.current_level,
                    'step': step,
                    'correct': correct,
                    'answer': answer,
                    'expected': expected,
                    'learning_mode': learning_mode,
                })

                if (step + 1) % check_interval == 0:
                    graduated, acc = env.check_graduation(agent)
                    n_done = len(history) - level_start
                    print(f"Level {env.current_level}, 第{round_num}轮, {n_done}题: acc={acc:.3f}"
                          + (" → 毕业升级!" if graduated else ""))
                    if graduated:
                        break
            else:
                # 用完了 problems_per_level 还没毕业
                print(f"Level {env.current_level}, 第{round_num}轮: 未毕业")
                continue  # 下一轮
            break  # 毕业了，跳出 round 循环
        else:
            # max_rounds 轮都没毕业
            print(f"Level {env.current_level}: {max_rounds}轮未毕业，停止训练")
            break

    # 巩固训练：用 max_digits 位题目继续训练
    if consolidation_steps > 0:
        print(f"巩固训练: {consolidation_steps} 步 {max_digits}位加法...")
        rng = random.Random(seed + 1)
        for step in range(consolidation_steps):
            a = rng.randrange(10 ** max_digits)
            b = rng.randrange(10 ** max_digits)
            answer, correct, info = agent.solve_and_learn(str(a), str(b))
            history.append({
                'level': max_digits + 1,  # 标记为巩固阶段
                'step': step,
                'correct': correct,
                'answer': answer,
                'expected': str(a + b),
                'learning_mode': learning_mode,
            })
            if (step + 1) % max(1, consolidation_steps // 5) == 0:
                results = evaluate_agent(agent, max_digits=max_digits, n_per_len=100, seed=999)
                print(f"  巩固 {step+1}/{consolidation_steps}: {results}")

    return agent, history


@torch.no_grad()
def evaluate_agent(agent, max_digits=5, n_per_len=200, seed=12345, eval_lengths=None):
    """评估学生模型在各位数上的准确率（纯推理，不学习）。

    进位传递：前馈 CarryPropagator，学生进位链（free running）。
    """
    if eval_lengths is None:
        eval_lengths = [L for L in [1, 2, 3, 4, 5, 8, 12, 16, 20] if L <= max_digits]
    rng = random.Random(seed)
    results = {}
    agent.eval()
    for L in eval_lengths:
        ok = 0
        for _ in range(n_per_len):
            a = rng.randrange(10 ** L)
            b = rng.randrange(10 ** L)
            n = max(len(str(a)), len(str(b)))
            a_s, b_s = str(a).zfill(n), str(b).zfill(n)
            out = []
            prev_cout = 0
            cout = 0
            for i in range(n - 1, -1, -1):
                cin = agent.student_carry.predict(prev_cout)
                digit, cout = agent.student_adder.predict(int(a_s[i]), int(b_s[i]), cin)
                out.append(digit)
                prev_cout = cout
            if cout:
                out.append(1)
            answer = ''.join(str(d) for d in reversed(out))
            ok += int(answer == str(a + b))
        results[L] = ok / n_per_len
    agent.train()
    return results


def plot_learning_curve(history, save_path=None):
    """绘制学习曲线：accuracy，按 level 分色。"""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    levels = sorted(set(h['level'] for h in history))
    colors = plt.cm.tab10(range(len(levels)))

    for level, color in zip(levels, colors):
        data = [h for h in history if h['level'] == level]
        steps = range(len(data))
        # 准确率（累积）
        corrects = [h['correct'] for h in data]
        cum_acc = []
        for i in range(len(corrects)):
            cum_acc.append(sum(corrects[:i+1]) / (i+1))
        ax.plot(steps, cum_acc, color=color, label=f'{level}位', alpha=0.8)

    ax.set_ylabel('Cumulative Accuracy')
    ax.set_xlabel('Problems within level')
    ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.5, label='毕业线')
    ax.legend()
    ax.grid(True, alpha=0.3)
    mode = history[0].get('learning_mode', 'unknown') if history else 'unknown'
    fig.suptitle(f'在线学习曲线（{mode}模式）')
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"学习曲线已保存到 {save_path}")
    return fig


if __name__ == "__main__":
    print("===== 幼儿式在线学习：从零开始学加法 =====\n")
    agent, history = train_student(
        max_digits=3, problems_per_level=2000,
        check_interval=200, learning_mode='hebbian', seed=0,
        samples_per_test=50, max_rounds=5
    )
    print("\n===== 最终评估 =====")
    results = evaluate_agent(agent, max_digits=3, n_per_len=200, seed=999)
    for L, acc in results.items():
        print(f"  {L:2d} 位: {acc:.4f}")

    print("\n===== 学习曲线 =====")
    import os
    os.makedirs("outputs", exist_ok=True)
    plot_learning_curve(history, save_path="outputs/online_learning_curve.png")
