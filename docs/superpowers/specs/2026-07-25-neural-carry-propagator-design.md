# 纯神经进位传播器设计

## 目标

将模块化加法器中的**程序化进位路由**替换为**学习型进位传播模型**，实现"所有逻辑由模型完成，零确定性程序"的纯神经加法系统。

## 当前架构

```
DigitAdder (MLP, 2188 参数): a+b+cin → digit + cout
add() 函数 (程序化): carry = cout → 下一位 cin
```

## 新架构

```
DigitAdder (MLP, 不变): a+b+cin → digit + cout
CarryPropagator (GRU Cell, ~282 参数): cout → h → cin
```

两个模型协作，逐位循环（低位→高位），进位传递完全由 CarryPropagator 的隐状态学习。

## CarryPropagator 模型

```python
class CarryPropagator(nn.Module):
    def __init__(self, hidden=8):
        self.gru = nn.GRUCell(2, hidden)      # cout one-hot → 隐状态
        self.cin_head = nn.Linear(hidden, 2)   # 隐状态 → cin logits

    def forward(self, cout_seq):
        # cout_seq: [seq_len, 2] 从低位到高位的 cout 序列
        h = torch.zeros(self.gru.hidden_size)
        cins = []
        for cout in cout_seq:
            cin_logits = self.cin_head(h)
            cins.append(cin_logits)
            h = self.gru(cout.unsqueeze(0), h.unsqueeze(0)).squeeze(0)
        return cins  # list of [2] logits
```

参数量：GRUCell(2,8) = 264 + Linear(8,2) = 18 → **总计 ~282 参数**。

## 训练策略：分阶段独立训练

### 阶段 1：训练 DigitAdder（已有）

- 数据：全部 200 种 (a, b, cin) 映射
- 目标：digit_acc=1.0, carry_acc=1.0
- 不变，复用现有 `train_adder()`

### 阶段 2：训练 CarryPropagator

- 数据生成：随机 N 位数加法（1-20 位），用确定性规则计算 cout/cin 序列
  - 注意：cout 来自确定性规则 `(a+b+cin)//10`，不依赖 DigitAdder 预测
  - 这样 CarryPropagator 训练不受 DigitAdder 误差影响
- 输入：cout 序列（one-hot, 从低位到高位）
- 目标：cin 序列（偏移 1 位，首位 cin=0）
- Loss：CrossEntropyLoss on cin predictions
- 训练样本量：~10000-50000 序列

### 阶段 3：组合测试

- 两个模型冻结，组合推理
- 评估 1-20 位 exact match

## 组合推理流程

```python
def neural_add(a_str, b_str, adder, carry_prop):
    n = max(len(a_str), len(b_str))
    a, b = a_str.zfill(n), b_str.zfill(n)
    h = torch.zeros(carry_prop.gru.hidden_size)
    out = []
    for i in range(n-1, -1, -1):  # 低位→高位
        cin = carry_prop.cin_head(h).argmax(-1).item()
        digit, cout = adder.predict(int(a[i]), int(b[i]), cin)
        out.append(digit)
        cout_oh = torch.zeros(2); cout_oh[cout] = 1.0
        h = carry_prop.gru(cout_oh.unsqueeze(0), h.unsqueeze(0)).squeeze(0)
    # 最高位是否还有进位
    cin = carry_prop.cin_head(h).argmax(-1).item()
    if cin:
        out.append(1)
    return ''.join(str(d) for d in reversed(out))
```

## 评估指标

与现有 `evaluate_lengths` 相同：
- 位数范围：1, 2, 3, 4, 5, 8, 12, 16, 20
- 每长度 500 样本
- Exact match 准确率

## 文件结构

- `modular_addition.py` — 修改：新增 CarryPropagator 类、neural_add 函数、训练/评估逻辑
- `tests/test_modular.py` — 修改：新增 CarryPropagator 和 neural_add 的测试

## 关键设计决策

1. **GRU Cell 而非 LSTM**：进位序列极短（≤20 步），GRU 够用且参数更少
2. **隐状态维度 8**：进位只有 0/1 两种，8 维隐状态远超所需，但保持一定冗余便于学习
3. **分阶段训练**：DigitAdder 已验证 100% 准确，无需重新训练；CarryPropagator 独立训练更可控
4. **cout one-hot 编码**：与 DigitAdder 的 carry_head 输出对齐，2 维输入
