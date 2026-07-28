# 在线交互式学习：幼儿式加法学习系统设计

## 目标

将模块化加法器从"离线训练+冻结推理"改为"在线交互式学习"：
- 学生模型从空（随机初始化）开始
- 外部环境出题，学生边推理边学习
- 教师提供逐位纠正信号，学生即时更新权重
- 课程从简单到复杂，模拟幼儿学习过程

## 当前架构

```
离线训练 DigitAdder → 离线训练 CarryPropagator → 冻结 → 推理
```

## 新架构

```
StudentAgent (空模型) ←出题— CurriculumEnv —纠正→ Teacher (确定性规则)
  逐位: 学生推理 → 教师纠正 → 即时更新权重 → 下一位用新权重
```

## 三大组件

### 1. StudentAgent

封装空初始化的 DigitAdder + CarryPropagator，以及统一 optimizer。

```python
class StudentAgent(nn.Module):
    def __init__(self, lr=1e-2):
        self.student_adder = DigitAdder(hidden=32)     # 随机初始化
        self.student_carry = CarryPropagator(hidden=8)  # 随机初始化
        self.optimizer = Adam(
            list(self.student_adder.parameters()) +
            list(self.student_carry.parameters()),
            lr=lr
        )
```

核心方法 `solve_and_learn(a_str, b_str)` → `(answer, correct, total_loss)`：

逐位循环（低位→高位）：
1. **学生推理**：cin = student_carry.cin_from_h(h)，digit/cout = student_adder(a,b,cin)
2. **教师纠正**：用确定性规则计算 cin_true, digit_true, cout_true
3. **即时学习**：loss = CE(cin_logits, cin_true) + CE(digit_logits, digit_true) + CE(cout_logits, cout_true)；backward + step
4. **推进**：用学生的 cout 更新 carry 隐状态，教师的 cout 更新 carry_true 给下一位

关键设计决策：
- 学生用自己预测的 cin 做加法（模拟"按自己的理解做题"）
- loss 用教师的正确标签（教师纠正"你这一步应该输出什么"）
- 隐状态用学生的 cout 更新（学生按自己的推理继续）
- carry_true 由教师维护（确保纠正信号链正确）

### 2. Teacher（确定性规则）

不是神经网络，是数学规则：
- `digit_true = (a + b + cin_true) % 10`
- `cout_true = (a + b + cin_true) // 10`
- `cin_true = carry_true`（教师维护的正确进位链）

永远正确，不需要训练。

### 3. CurriculumEnv

课程环境：从简单到复杂出题，学生掌握后升级。

```python
class CurriculumEnv:
    def __init__(self, max_digits=5, graduate_acc=0.95,
                 samples_per_test=100, seed=0):
        self.current_level = 1  # 从1位开始

    def next_problem(self):
        # 当前难度级别的随机加法

    def check_graduation(self, agent):
        # 检查准确率，≥ graduate_acc 则升级
        # 注意：检查过程中 agent 也在学习（solve_and_learn 有 side effect）
        # 这是设计意图——没有"纯测试"模式，学习永不停止
```

课程流程：
1. 从 1 位加法开始
2. 学生反复练习，边做边学
3. 每隔 N 题检查一次准确率
4. 准确率 ≥ 95% → 升级到更多位
5. 直到达到 max_digits

## 训练主循环

```python
def train_student(max_digits=5, problems_per_level=500,
                  check_interval=50, lr=1e-2, seed=0):
    agent = StudentAgent(lr=lr)
    env = CurriculumEnv(max_digits=max_digits, seed=seed)
    history = []

    while env.current_level <= max_digits:
        for step in range(problems_per_level):
            a_str, b_str = env.next_problem()
            answer, correct, loss = agent.solve_and_learn(a_str, b_str)
            history.append({...})

            if (step + 1) % check_interval == 0:
                graduated, acc = env.check_graduation(agent)
                if graduated:
                    break

    return agent, history
```

## 评估与可视化

- 学习曲线：每步的 loss 和 correct，按 level 分色
- 最终评估：1-max_digits 位 exact match
- 与离线训练对比：同样参数量，在线学习需要多少步达到离线效果

## 文件结构

- `online_addition.py` — 新文件：StudentAgent, CurriculumEnv, train_student, 评估/可视化
- `modular_addition.py` — 不变：DigitAdder, CarryPropagator, encode_abc 等被复用
- `tests/test_online.py` — 新文件：StudentAgent 和 CurriculumEnv 的测试

## 关键设计决策

1. **Teacher 是确定性规则而非预训练模型**：永远正确，无需额外训练，更干净
2. **逐步即时学习而非批量**：每做一步就更新权重，下一位立即用新权重，最接近"边做边学"
3. **学生用自己预测的 cin**：模拟真实学习——学生按自己的理解做题，即使理解有误
4. **课程从1位开始**：1位只有 100 种组合，最容易学会；进位在 2 位时自然出现
5. **毕业条件 95%**：允许少量错误，不必 100% 才升级，更接近真实学习节奏
