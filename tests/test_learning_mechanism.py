import torch
from learning_mechanism import LearningMechanism, HebbianLearner, RewardLearner, CorrectionLearner
from modular_addition import DigitAdder
from online_addition import FeedForwardCarry


def test_learning_mechanism_base():
    lm = LearningMechanism()
    assert lm.get_param_count() == 0


def test_hebbian_learner_init():
    hl = HebbianLearner()
    # 默认 adder_hidden=16, carry_hidden=16
    # η+γ: 4+3+4+3 = 14
    # B_digit(16*10=160) + B_carry(16*2=32) + B2(16*16=256)        adder 反馈 = 448
    # B_carry_cin(16*2=32) + B_carry2(16*16=256)                    carry 反馈 = 288
    # 合计 14 + 448 + 288 = 750
    assert hl.get_param_count() == 750


def test_hebbian_learner_learn_step_adder():
    """FA 学习应能修改 DigitAdder 权重。"""
    adder = DigitAdder(hidden=16)
    hl = HebbianLearner()
    w_before = adder.digit_head.weight.data.clone()
    x = torch.zeros(22); x[3] = 1.0; x[14] = 1.0; x[20] = 1.0
    with torch.no_grad():
        h1 = adder.net[0](x.unsqueeze(0))
        h1_act = adder.net[1](h1)
    correction = {'digit_true': 7, 'cout_true': 0, 'cin_true': 0}
    hl.learn_step_adder(adder, x, h1_act, signal=correction)
    assert not torch.equal(w_before, adder.digit_head.weight.data)


def test_hebbian_learner_learn_step_carry():
    """FA 学习应能修改前馈 CarryPropagator 权重。"""
    carry = FeedForwardCarry(hidden=16)
    hl = HebbianLearner(carry_hidden=16)
    w_before = carry.cin_head.weight.data.clone()
    cout_oh = torch.zeros(2); cout_oh[1] = 1.0   # cout=1
    with torch.no_grad():
        h1 = carry.net[0](cout_oh.unsqueeze(0))
        h1_act = carry.net[1](h1)
    hl.learn_step_carry(carry, cout_oh, h1_act, signal={'cin_true': 1})
    assert not torch.equal(w_before, carry.cin_head.weight.data)


def test_hebbian_learner_carry_learns_identity():
    """前馈 CarryPropagator 经 FA 训练后应学会 cout→cin 恒等映射。"""
    carry = FeedForwardCarry(hidden=16)
    hl = HebbianLearner(carry_hidden=16, init_eta=0.1, init_gamma=0.0)
    # 训练：cout=0→cin=0, cout=1→cin=1
    for _ in range(500):
        for cout_val in [0, 1]:
            cout_oh = torch.zeros(2); cout_oh[cout_val] = 1.0
            with torch.no_grad():
                h1 = carry.net[0](cout_oh.unsqueeze(0))
                h1_act = carry.net[1](h1)
            hl.learn_step_carry(carry, cout_oh, h1_act, signal={'cin_true': cout_val})
    # 验证恒等映射
    assert carry.predict(0) == 0, "cout=0 应预测 cin=0"
    assert carry.predict(1) == 1, "cout=1 应预测 cin=1"


def test_reward_learner_init():
    rl = RewardLearner()
    # Hebbian(750) + β_adder(4) + β_carry(3) = 757
    assert rl.get_param_count() == 757


def test_reward_learner_with_reward_and_correction():
    """奖励+纠正应能修改权重。"""
    adder = DigitAdder(hidden=16)
    rl = RewardLearner()
    w_before = adder.digit_head.weight.data.clone()
    x = torch.zeros(22); x[3] = 1.0; x[14] = 1.0; x[20] = 1.0
    with torch.no_grad():
        h1 = adder.net[0](x.unsqueeze(0))
        h1_act = adder.net[1](h1)
    signal = {'digit_true': 7, 'cout_true': 0, 'cin_true': 0, 'reward': 1.0}
    rl.learn_step_adder(adder, x, h1_act, signal=signal)
    assert not torch.equal(w_before, adder.digit_head.weight.data)


def test_correction_learner_init():
    cl = CorrectionLearner()
    assert cl.get_param_count() > 0


def test_correction_learner_with_correction():
    """教师纠正信号应能修改权重。"""
    adder = DigitAdder(hidden=16)
    cl = CorrectionLearner()
    w_before = adder.net[0].weight.data.clone()
    x = torch.zeros(22); x[3] = 1.0; x[14] = 1.0; x[20] = 1.0
    with torch.no_grad():
        h1 = adder.net[0](x.unsqueeze(0))
        h1_act = adder.net[1](h1)
    correction = {'digit_true': 7, 'cout_true': 0, 'cin_true': 0}
    cl.learn_step_adder(adder, x, h1_act, signal=correction)
    assert not torch.equal(w_before, adder.net[0].weight.data)
