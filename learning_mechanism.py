"""内部学习机制：独立参数集，自主修改推理模型权重。

三种学习模式：
  A. RewardLearner  — 奖励调制 delta 规则，类似多巴胺门控
  B. CorrectionLearner — 教师纠正 + 内部规则网络决定更新幅度
  C. HebbianLearner  — 反馈对齐（FA）：用固定随机矩阵传播误差，局部有监督

所有模式都基于 delta 规则（error * pre），确保权重更新有正确方向。
区别在于误差信号的来源和调制方式。

关键设计：学习机制有自己的参数（η, γ, 反馈矩阵 B 等），不占用推理模型的 886 参数。
"""
import torch
import torch.nn as nn


class LearningMechanism(nn.Module):
    """内部学习机制基类：独立参数集，自主修改推理模型权重。"""

    def __init__(self):
        super().__init__()

    def learn_step_adder(self, adder, x, h1_act, signal=None):
        """对 DigitAdder 执行一步学习。

        Args:
            adder: DigitAdder 推理模型
            x: 输入向量 [22]
            h1_act: 第一层激活（post-ReLU）[1, hidden]
            signal: dict {'digit_true': int, 'cout_true': int, 'cin_true': int}
        """
        raise NotImplementedError

    def learn_step_carry(self, carry, cout_oh, h1_act, signal=None):
        """对前馈 CarryPropagator 执行一步学习。

        Args:
            carry: 前馈 CarryPropagator 推理模型
            cout_oh: 输入 cout one-hot [2]
            h1_act: 第一层激活（post-ReLU）[1, carry_hidden]
            signal: dict {'cin_true': int}
        """
        raise NotImplementedError

    def get_param_count(self):
        """返回学习机制自身的参数量。"""
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def _make_target_onehot(logits, true_class, n_classes):
        """构造目标 one-hot 向量（与 logits 同形状的 soft target）。"""
        target = torch.zeros(n_classes)
        target[true_class] = 1.0
        return target


class HebbianLearner(LearningMechanism):
    """反馈对齐（Feedback Alignment）局部学习。

    用固定随机反馈矩阵 B 代替 W^T 传播误差，保持局部性同时提供有监督方向。
    输出层：ΔW = η * (target - softmax(output)) * pre^T     （精确 softmax-CE 梯度）
    隐藏层：ΔW = η * (B @ output_error) * ReLU' * pre^T      （FA 近似反传）

    学习机制参数（独立于推理模型）：
    - η_adder (4), γ_adder (4): DigitAdder 各层学习率/衰减
    - η_carry (3), γ_carry (3): 前馈 CarryPropagator 各层学习率/衰减
    - B_digit [h,10], B_carry [h,2], B2 [h,h]: DigitAdder 反馈矩阵
    - B_carry_cin [h_c,2], B_carry2 [h_c,h_c]: CarryPropagator 反馈矩阵
    """

    def __init__(self, init_eta=0.01, init_gamma=0.001,
                 adder_hidden=16, carry_hidden=16,
                 eta_out=None, eta_hid=None):
        super().__init__()
        # 分层学习率：输出层需要更大 eta（信号强），隐藏层较小（避免不稳定）
        _eta_out = eta_out if eta_out is not None else init_eta * 10
        _eta_hid = eta_hid if eta_hid is not None else init_eta
        # [W1, W2, Wd, Wc] — 前两个隐藏层用 eta_hid，后两个输出层用 eta_out
        self.log_eta_adder = nn.Parameter(torch.log(torch.tensor([
            _eta_hid, _eta_hid, _eta_out, _eta_out
        ])))
        # carry 前馈模型：[net0, net2, cin_head] — 隐藏层用 eta_hid，输出用 eta_out
        self.log_eta_carry = nn.Parameter(torch.log(torch.tensor([
            _eta_hid, _eta_hid, _eta_out
        ])))
        # 衰减系数
        self.log_gamma_adder = nn.Parameter(torch.full((4,), torch.log(torch.tensor(init_gamma))))
        self.log_gamma_carry = nn.Parameter(torch.full((3,), torch.log(torch.tensor(init_gamma))))

        # 固定随机反馈矩阵（FA 的核心：用 B 代替 W^T 传播误差）
        # B_digit: [hidden, 10] — digit_head 误差反馈到 h2
        self.B_digit = nn.Parameter(torch.randn(adder_hidden, 10) * 0.1, requires_grad=False)
        # B_carry: [hidden, 2] — carry_head 误差反馈到 h2
        self.B_carry = nn.Parameter(torch.randn(adder_hidden, 2) * 0.1, requires_grad=False)
        # B2: [hidden, hidden] — h2 误差反馈到 h1
        self.B2 = nn.Parameter(torch.randn(adder_hidden, adder_hidden) * 0.1, requires_grad=False)
        # carry 前馈模型的反馈矩阵
        self.B_carry_cin = nn.Parameter(torch.randn(carry_hidden, 2) * 0.1, requires_grad=False)
        self.B_carry2 = nn.Parameter(torch.randn(carry_hidden, carry_hidden) * 0.1, requires_grad=False)

    @property
    def eta_adder(self):
        return torch.exp(self.log_eta_adder)

    @property
    def eta_carry(self):
        return torch.exp(self.log_eta_carry)

    @property
    def gamma_adder(self):
        return torch.exp(self.log_gamma_adder)

    @property
    def gamma_carry(self):
        return torch.exp(self.log_gamma_carry)

    def learn_step_adder(self, adder, x, h1_act, signal=None):
        """反馈对齐更新 DigitAdder 权重。"""
        if signal is None:
            return
        with torch.no_grad():
            # 前向计算
            h2 = adder.net[2](h1_act)        # pre-ReLU
            h2_act = adder.net[3](h2)        # post-ReLU
            d_logits = adder.digit_head(h2_act)
            c_logits = adder.carry_head(h2_act)

            # 输出层误差：target_onehot - softmax_output
            d_target = self._make_target_onehot(d_logits, signal['digit_true'], 10)
            d_output = torch.softmax(d_logits.squeeze(0), dim=0)
            d_error = d_target - d_output    # [10]

            c_target = self._make_target_onehot(c_logits, signal['cout_true'], 2)
            c_output = torch.softmax(c_logits.squeeze(0), dim=0)
            c_error = c_target - c_output    # [2]

            eta = self.eta_adder
            gamma = self.gamma_adder

            # ---- 输出层：标准 delta 规则 ----
            # Wd: ΔW = η * d_error * h2_act^T - γ * W
            adder.digit_head.weight.data += (
                eta[2] * d_error.unsqueeze(1) * h2_act.squeeze(0).unsqueeze(0)
                - gamma[2] * adder.digit_head.weight.data
            )
            adder.digit_head.bias.data += eta[2] * d_error - gamma[2] * adder.digit_head.bias.data

            # Wc: ΔW = η * c_error * h2_act^T - γ * W
            adder.carry_head.weight.data += (
                eta[3] * c_error.unsqueeze(1) * h2_act.squeeze(0).unsqueeze(0)
                - gamma[3] * adder.carry_head.weight.data
            )
            adder.carry_head.bias.data += eta[3] * c_error - gamma[3] * adder.carry_head.bias.data

            # ---- 隐藏层：反馈对齐 ----
            # h2 的误差 = B_digit @ d_error + B_carry @ c_error
            h2_error = self.B_digit @ d_error + self.B_carry @ c_error  # [hidden]
            # ReLU 导数
            h2_relu_grad = (h2.squeeze(0) > 0).float()  # pre-ReLU > 0
            h2_delta = h2_error * h2_relu_grad            # [hidden]

            # W2: ΔW = η * h2_delta * h1_act^T - γ * W
            adder.net[2].weight.data += (
                eta[1] * h2_delta.unsqueeze(1) * h1_act.squeeze(0).unsqueeze(0)
                - gamma[1] * adder.net[2].weight.data
            )
            adder.net[2].bias.data += eta[1] * h2_delta - gamma[1] * adder.net[2].bias.data

            # h1 的误差 = B2 @ h2_delta
            h1_error = self.B2 @ h2_delta  # [hidden]
            h1_relu_grad = (adder.net[0](x.unsqueeze(0)).squeeze(0) > 0).float()
            h1_delta = h1_error * h1_relu_grad

            # W1: ΔW = η * h1_delta * x^T - γ * W
            adder.net[0].weight.data += (
                eta[0] * h1_delta.unsqueeze(1) * x.unsqueeze(0)
                - gamma[0] * adder.net[0].weight.data
            )
            adder.net[0].bias.data += eta[0] * h1_delta - gamma[0] * adder.net[0].bias.data

    def learn_step_carry(self, carry, cout_oh, h1_act, signal=None):
        """反馈对齐更新前馈 CarryPropagator 权重（cout → cin 单步映射）。

        Args:
            carry: 前馈 CarryPropagator（net: [Linear, ReLU, Linear, ReLU], cin_head）
            cout_oh: 输入 cout one-hot [2]
            h1_act: 第一层激活（post-ReLU）[1, carry_hidden]
            signal: dict {'cin_true': int}
        """
        if signal is None:
            return
        with torch.no_grad():
            h2 = carry.net[2](h1_act)        # pre-ReLU
            h2_act = carry.net[3](h2)        # post-ReLU
            cin_logits = carry.cin_head(h2_act)

            cin_target = self._make_target_onehot(cin_logits, signal.get('cin_true', 0), 2)
            cin_output = torch.softmax(cin_logits.squeeze(0), dim=0)
            cin_error = cin_target - cin_output  # [2]

            eta = self.eta_carry
            gamma = self.gamma_carry

            # ---- 输出层 cin_head：标准 delta 规则 ----
            carry.cin_head.weight.data += (
                eta[2] * cin_error.unsqueeze(1) * h2_act.squeeze(0).unsqueeze(0)
                - gamma[2] * carry.cin_head.weight.data
            )
            carry.cin_head.bias.data += eta[2] * cin_error - gamma[2] * carry.cin_head.bias.data

            # ---- 隐藏层：反馈对齐 ----
            h2_error = self.B_carry_cin @ cin_error      # [carry_hidden]
            h2_relu_grad = (h2.squeeze(0) > 0).float()
            h2_delta = h2_error * h2_relu_grad

            carry.net[2].weight.data += (
                eta[1] * h2_delta.unsqueeze(1) * h1_act.squeeze(0).unsqueeze(0)
                - gamma[1] * carry.net[2].weight.data
            )
            carry.net[2].bias.data += eta[1] * h2_delta - gamma[1] * carry.net[2].bias.data

            h1_error = self.B_carry2 @ h2_delta          # [carry_hidden]
            h1_relu_grad = (carry.net[0](cout_oh.unsqueeze(0)).squeeze(0) > 0).float()
            h1_delta = h1_error * h1_relu_grad

            carry.net[0].weight.data += (
                eta[0] * h1_delta.unsqueeze(1) * cout_oh.unsqueeze(0)
                - gamma[0] * carry.net[0].weight.data
            )
            carry.net[0].bias.data += eta[0] * h1_delta - gamma[0] * carry.net[0].bias.data


class RewardLearner(LearningMechanism):
    """奖励门控 delta 规则学习。

    奖励门控学习强度，方向始终由 error = target - softmax(output) 决定：
    - 正确 (r=+1): gate=1.0（标准更新）
    - 错误 (r=-1): gate=2.0（增大更新幅度，需要更多学习）
    类似多巴胺：奖励信号不改变学习方向，而是调节"学多少"。

    参数：与 HebbianLearner 相同的 η/γ/B，外加 β_adder(4) + β_carry(3) 调制因子。
    """

    def __init__(self, init_eta=0.01, init_gamma=0.001, init_beta=1.0,
                 adder_hidden=16, carry_hidden=16,
                 eta_out=None, eta_hid=None):
        super().__init__()
        _eta_out = eta_out if eta_out is not None else init_eta * 10
        _eta_hid = eta_hid if eta_hid is not None else init_eta
        self.log_eta_adder = nn.Parameter(torch.log(torch.tensor([
            _eta_hid, _eta_hid, _eta_out, _eta_out
        ])))
        self.log_eta_carry = nn.Parameter(torch.log(torch.tensor([
            _eta_hid, _eta_hid, _eta_out
        ])))
        self.log_gamma_adder = nn.Parameter(torch.full((4,), torch.log(torch.tensor(init_gamma))))
        self.log_gamma_carry = nn.Parameter(torch.full((3,), torch.log(torch.tensor(init_gamma))))
        self.log_beta_adder = nn.Parameter(torch.full((4,), torch.log(torch.tensor(init_beta))))
        self.log_beta_carry = nn.Parameter(torch.full((3,), torch.log(torch.tensor(init_beta))))

        # 反馈矩阵
        self.B_digit = nn.Parameter(torch.randn(adder_hidden, 10) * 0.1, requires_grad=False)
        self.B_carry = nn.Parameter(torch.randn(adder_hidden, 2) * 0.1, requires_grad=False)
        self.B2 = nn.Parameter(torch.randn(adder_hidden, adder_hidden) * 0.1, requires_grad=False)
        self.B_carry_cin = nn.Parameter(torch.randn(carry_hidden, 2) * 0.1, requires_grad=False)
        self.B_carry2 = nn.Parameter(torch.randn(carry_hidden, carry_hidden) * 0.1, requires_grad=False)

    @property
    def eta_adder(self):
        return torch.exp(self.log_eta_adder)

    @property
    def eta_carry(self):
        return torch.exp(self.log_eta_carry)

    @property
    def gamma_adder(self):
        return torch.exp(self.log_gamma_adder)

    @property
    def gamma_carry(self):
        return torch.exp(self.log_gamma_carry)

    @property
    def beta_adder(self):
        return torch.exp(self.log_beta_adder)

    @property
    def beta_carry(self):
        return torch.exp(self.log_beta_carry)

    def learn_step_adder(self, adder, x, h1_act, signal=None):
        """奖励门控 delta 规则更新 DigitAdder。

        奖励门控学习强度，方向始终由 error = target - softmax(output) 决定：
        - 错误 (r<0):  gate=2.0（增大更新幅度，需要更多学习）
        - 正确/中性:   gate=1.0（标准更新幅度）
        """
        if signal is None:
            return
        r = signal.get('reward', 0.0) if isinstance(signal, dict) else (signal if signal is not None else 0.0)
        correction = signal if isinstance(signal, dict) else {}

        # 奖励门控：只调节强度，不改变方向
        gate = 2.0 if r < 0 else 1.0

        with torch.no_grad():
            h2 = adder.net[2](h1_act)
            h2_act = adder.net[3](h2)
            d_logits = adder.digit_head(h2_act)
            c_logits = adder.carry_head(h2_act)

            d_target = self._make_target_onehot(d_logits, correction['digit_true'], 10)
            d_output = torch.softmax(d_logits.squeeze(0), dim=0)
            d_error = d_target - d_output

            c_target = self._make_target_onehot(c_logits, correction['cout_true'], 2)
            c_output = torch.softmax(c_logits.squeeze(0), dim=0)
            c_error = c_target - c_output

            eta = self.eta_adder
            gamma = self.gamma_adder
            beta = self.beta_adder

            # 输出层：η * gate * β * error * pre（方向由 error 决定，gate 只调强度）
            adder.digit_head.weight.data += (
                eta[2] * gate * beta[2] * d_error.unsqueeze(1) * h2_act.squeeze(0).unsqueeze(0)
                - gamma[2] * adder.digit_head.weight.data
            )
            adder.digit_head.bias.data += eta[2] * gate * beta[2] * d_error - gamma[2] * adder.digit_head.bias.data

            adder.carry_head.weight.data += (
                eta[3] * gate * beta[3] * c_error.unsqueeze(1) * h2_act.squeeze(0).unsqueeze(0)
                - gamma[3] * adder.carry_head.weight.data
            )
            adder.carry_head.bias.data += eta[3] * gate * beta[3] * c_error - gamma[3] * adder.carry_head.bias.data

            # 隐藏层：FA + 奖励门控
            h2_error = self.B_digit @ d_error + self.B_carry @ c_error
            h2_relu_grad = (h2.squeeze(0) > 0).float()
            h2_delta = h2_error * h2_relu_grad

            adder.net[2].weight.data += (
                eta[1] * gate * beta[1] * h2_delta.unsqueeze(1) * h1_act.squeeze(0).unsqueeze(0)
                - gamma[1] * adder.net[2].weight.data
            )
            adder.net[2].bias.data += eta[1] * gate * beta[1] * h2_delta - gamma[1] * adder.net[2].bias.data

            h1_error = self.B2 @ h2_delta
            h1_relu_grad = (adder.net[0](x.unsqueeze(0)).squeeze(0) > 0).float()
            h1_delta = h1_error * h1_relu_grad

            adder.net[0].weight.data += (
                eta[0] * gate * beta[0] * h1_delta.unsqueeze(1) * x.unsqueeze(0)
                - gamma[0] * adder.net[0].weight.data
            )
            adder.net[0].bias.data += eta[0] * gate * beta[0] * h1_delta - gamma[0] * adder.net[0].bias.data

    def learn_step_carry(self, carry, cout_oh, h1_act, signal=None):
        """奖励门控 delta 规则更新前馈 CarryPropagator（cout → cin）。

        奖励门控学习强度，方向始终由 error = target - output 决定。
        """
        if signal is None:
            return
        r = signal.get('reward', 0.0) if isinstance(signal, dict) else (signal if signal is not None else 0.0)
        correction = signal if isinstance(signal, dict) else {}

        # 奖励门控：只调节强度，不改变方向
        gate = 2.0 if r < 0 else 1.0

        with torch.no_grad():
            cin_true = correction.get('cin_true', 0)
            h2 = carry.net[2](h1_act)
            h2_act = carry.net[3](h2)
            cin_logits = carry.cin_head(h2_act)

            cin_target = self._make_target_onehot(cin_logits, cin_true, 2)
            cin_output = torch.softmax(cin_logits.squeeze(0), dim=0)
            cin_error = cin_target - cin_output

            eta = self.eta_carry
            gamma = self.gamma_carry
            beta = self.beta_carry

            # cin_head: delta 规则
            carry.cin_head.weight.data += (
                eta[2] * gate * beta[2] * cin_error.unsqueeze(1) * h2_act.squeeze(0).unsqueeze(0)
                - gamma[2] * carry.cin_head.weight.data
            )
            carry.cin_head.bias.data += eta[2] * gate * beta[2] * cin_error - gamma[2] * carry.cin_head.bias.data

            # 隐藏层：FA + 奖励门控
            h2_error = self.B_carry_cin @ cin_error
            h2_relu_grad = (h2.squeeze(0) > 0).float()
            h2_delta = h2_error * h2_relu_grad

            carry.net[2].weight.data += (
                eta[1] * gate * beta[1] * h2_delta.unsqueeze(1) * h1_act.squeeze(0).unsqueeze(0)
                - gamma[1] * carry.net[2].weight.data
            )
            carry.net[2].bias.data += eta[1] * gate * beta[1] * h2_delta - gamma[1] * carry.net[2].bias.data

            h1_error = self.B_carry2 @ h2_delta
            h1_relu_grad = (carry.net[0](cout_oh.unsqueeze(0)).squeeze(0) > 0).float()
            h1_delta = h1_error * h1_relu_grad

            carry.net[0].weight.data += (
                eta[0] * gate * beta[0] * h1_delta.unsqueeze(1) * cout_oh.unsqueeze(0)
                - gamma[0] * carry.net[0].weight.data
            )
            carry.net[0].bias.data += eta[0] * gate * beta[0] * h1_delta - gamma[0] * carry.net[0].bias.data


class CorrectionLearner(LearningMechanism):
    """教师纠正 + 确定性门控 delta 规则学习。

    教师纠正信号提供 error 方向（target - softmax(output)），
    内部门控逻辑根据逐位预测是否正确调节更新强度：
    - 预测正确: gate=1.0（标准更新）
    - 预测错误: gate=2.0（增大更新）

    与 RewardLearner 的区别：门控信号来自逐位纠正（更精细），
    而非整体奖励标量。

    参数：η + γ + 反馈矩阵 B（与 HebbianLearner 相同结构）
    """

    def __init__(self, init_eta=0.01, init_gamma=0.001,
                 adder_hidden=16, carry_hidden=16,
                 eta_out=None, eta_hid=None):
        super().__init__()
        _eta_out = eta_out if eta_out is not None else init_eta * 10
        _eta_hid = eta_hid if eta_hid is not None else init_eta
        self.log_eta_adder = nn.Parameter(torch.log(torch.tensor([
            _eta_hid, _eta_hid, _eta_out, _eta_out
        ])))
        self.log_eta_carry = nn.Parameter(torch.log(torch.tensor([
            _eta_hid, _eta_hid, _eta_out
        ])))
        self.log_gamma_adder = nn.Parameter(torch.full((4,), torch.log(torch.tensor(init_gamma))))
        self.log_gamma_carry = nn.Parameter(torch.full((3,), torch.log(torch.tensor(init_gamma))))

        # 反馈矩阵
        self.B_digit = nn.Parameter(torch.randn(adder_hidden, 10) * 0.1, requires_grad=False)
        self.B_carry = nn.Parameter(torch.randn(adder_hidden, 2) * 0.1, requires_grad=False)
        self.B2 = nn.Parameter(torch.randn(adder_hidden, adder_hidden) * 0.1, requires_grad=False)
        self.B_carry_cin = nn.Parameter(torch.randn(carry_hidden, 2) * 0.1, requires_grad=False)
        self.B_carry2 = nn.Parameter(torch.randn(carry_hidden, carry_hidden) * 0.1, requires_grad=False)

    @property
    def eta_adder(self):
        return torch.exp(self.log_eta_adder)

    @property
    def eta_carry(self):
        return torch.exp(self.log_eta_carry)

    @property
    def gamma_adder(self):
        return torch.exp(self.log_gamma_adder)

    @property
    def gamma_carry(self):
        return torch.exp(self.log_gamma_carry)

    def learn_step_adder(self, adder, x, h1_act, signal=None):
        """纠正门控 delta 规则更新 DigitAdder。

        门控逻辑：预测正确时减小更新幅度，错误时增大。
        方向始终由 error = target - output 决定。
        """
        if signal is None:
            return
        with torch.no_grad():
            h2 = adder.net[2](h1_act)
            h2_act = adder.net[3](h2)
            d_logits = adder.digit_head(h2_act)
            c_logits = adder.carry_head(h2_act)

            # 输出层误差
            d_target = self._make_target_onehot(d_logits, signal['digit_true'], 10)
            d_output = torch.softmax(d_logits.squeeze(0), dim=0)
            d_error = d_target - d_output

            c_target = self._make_target_onehot(c_logits, signal['cout_true'], 2)
            c_output = torch.softmax(c_logits.squeeze(0), dim=0)
            c_error = c_target - c_output

            # 确定性门控：逐位判断正确/错误
            digit_correct = (d_logits.argmax(-1).item() == signal['digit_true'])
            cout_correct = (c_logits.argmax(-1).item() == signal['cout_true'])
            # 输出层门控：错误时大幅增加更新，正确时标准更新
            d_gate = 1.0 if digit_correct else 2.0
            c_gate = 1.0 if cout_correct else 2.0
            # 隐藏层门控：任一输出错误则大更新
            hid_gate = 2.0 if not (digit_correct and cout_correct) else 1.0

            eta = self.eta_adder
            gamma = self.gamma_adder

            # 输出层：η * gate * error * pre
            adder.digit_head.weight.data += (
                eta[2] * d_gate * d_error.unsqueeze(1) * h2_act.squeeze(0).unsqueeze(0)
                - gamma[2] * adder.digit_head.weight.data
            )
            adder.digit_head.bias.data += eta[2] * d_gate * d_error - gamma[2] * adder.digit_head.bias.data

            adder.carry_head.weight.data += (
                eta[3] * c_gate * c_error.unsqueeze(1) * h2_act.squeeze(0).unsqueeze(0)
                - gamma[3] * adder.carry_head.weight.data
            )
            adder.carry_head.bias.data += eta[3] * c_gate * c_error - gamma[3] * adder.carry_head.bias.data

            # 隐藏层：FA + 门控
            h2_error = self.B_digit @ d_error + self.B_carry @ c_error
            h2_relu_grad = (h2.squeeze(0) > 0).float()
            h2_delta = h2_error * h2_relu_grad

            adder.net[2].weight.data += (
                eta[1] * hid_gate * h2_delta.unsqueeze(1) * h1_act.squeeze(0).unsqueeze(0)
                - gamma[1] * adder.net[2].weight.data
            )
            adder.net[2].bias.data += eta[1] * hid_gate * h2_delta - gamma[1] * adder.net[2].bias.data

            h1_error = self.B2 @ h2_delta
            h1_relu_grad = (adder.net[0](x.unsqueeze(0)).squeeze(0) > 0).float()
            h1_delta = h1_error * h1_relu_grad

            adder.net[0].weight.data += (
                eta[0] * hid_gate * h1_delta.unsqueeze(1) * x.unsqueeze(0)
                - gamma[0] * adder.net[0].weight.data
            )
            adder.net[0].bias.data += eta[0] * hid_gate * h1_delta - gamma[0] * adder.net[0].bias.data

    def learn_step_carry(self, carry, cout_oh, h1_act, signal=None):
        """纠正门控 delta 规则更新前馈 CarryPropagator（cout → cin）。"""
        if signal is None:
            return
        with torch.no_grad():
            cin_true = signal.get('cin_true', 0)
            h2 = carry.net[2](h1_act)
            h2_act = carry.net[3](h2)
            cin_logits = carry.cin_head(h2_act)

            cin_target = self._make_target_onehot(cin_logits, cin_true, 2)
            cin_output = torch.softmax(cin_logits.squeeze(0), dim=0)
            cin_error = cin_target - cin_output

            # 确定性门控
            cin_correct = (cin_logits.argmax(-1).item() == cin_true)
            gate = 1.0 if cin_correct else 2.0

            eta = self.eta_carry
            gamma = self.gamma_carry

            # cin_head: delta 规则
            carry.cin_head.weight.data += (
                eta[2] * gate * cin_error.unsqueeze(1) * h2_act.squeeze(0).unsqueeze(0)
                - gamma[2] * carry.cin_head.weight.data
            )
            carry.cin_head.bias.data += eta[2] * gate * cin_error - gamma[2] * carry.cin_head.bias.data

            # 隐藏层：FA + 门控
            h2_error = self.B_carry_cin @ cin_error
            h2_relu_grad = (h2.squeeze(0) > 0).float()
            h2_delta = h2_error * h2_relu_grad

            carry.net[2].weight.data += (
                eta[1] * gate * h2_delta.unsqueeze(1) * h1_act.squeeze(0).unsqueeze(0)
                - gamma[1] * carry.net[2].weight.data
            )
            carry.net[2].bias.data += eta[1] * gate * h2_delta - gamma[1] * carry.net[2].bias.data

            h1_error = self.B_carry2 @ h2_delta
            h1_relu_grad = (carry.net[0](cout_oh.unsqueeze(0)).squeeze(0) > 0).float()
            h1_delta = h1_error * h1_relu_grad

            carry.net[0].weight.data += (
                eta[0] * gate * h1_delta.unsqueeze(1) * cout_oh.unsqueeze(0)
                - gamma[0] * carry.net[0].weight.data
            )
            carry.net[0].bias.data += eta[0] * gate * h1_delta - gamma[0] * carry.net[0].bias.data
