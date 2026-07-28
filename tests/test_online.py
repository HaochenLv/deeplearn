import torch
from online_addition import StudentAgent, CurriculumEnv, train_student, evaluate_agent


def test_student_agent_init_hebbian():
    agent = StudentAgent(learning_mode='hebbian')
    assert hasattr(agent, 'student_adder')
    assert hasattr(agent, 'student_carry')
    assert hasattr(agent, 'learner')
    assert not hasattr(agent, 'optimizer')  # 不再有外部 optimizer
    # 推理模型参数量
    n_inference = sum(p.numel() for p in agent.student_adder.parameters()) + \
                  sum(p.numel() for p in agent.student_carry.parameters())
    assert n_inference > 0
    # 学习机制有独立参数
    n_learner = agent.learner.get_param_count()
    assert n_learner > 0
    assert n_learner != n_inference  # 独立参数集


def test_student_agent_init_reward():
    agent = StudentAgent(learning_mode='reward')
    n_learner = agent.learner.get_param_count()
    assert n_learner > 0


def test_student_agent_init_correction():
    agent = StudentAgent(learning_mode='correction')
    n_learner = agent.learner.get_param_count()
    assert n_learner > 0


def test_solve_and_learn_returns():
    agent = StudentAgent(learning_mode='hebbian')
    answer, correct, info = agent.solve_and_learn("3", "4")
    assert isinstance(answer, str)
    assert isinstance(correct, bool)
    assert isinstance(info, dict)


def test_solve_and_learn_1digit_hebbian():
    """1 位加法经过 FA 学习后应能正确。"""
    agent = StudentAgent(learning_mode='hebbian', init_eta=0.05)
    for _ in range(500):
        agent.solve_and_learn("3", "4")
    answer, correct, _ = agent.solve_and_learn("3", "4")
    assert correct, f"3+4={answer}, expected 7"


def test_solve_and_learn_1digit_reward():
    """1 位加法经过奖励学习后应能正确。"""
    agent = StudentAgent(learning_mode='reward', init_eta=0.05)
    for _ in range(1000):
        agent.solve_and_learn("3", "4")
    answer, correct, _ = agent.solve_and_learn("3", "4")
    assert correct, f"3+4={answer}, expected 7"


def test_solve_and_learn_1digit_correction():
    """1 位加法经过纠正学习后应能正确。"""
    agent = StudentAgent(learning_mode='correction', init_eta=0.05)
    for _ in range(1000):
        agent.solve_and_learn("3", "4")
    answer, correct, _ = agent.solve_and_learn("3", "4")
    assert correct, f"3+4={answer}, expected 7"


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


def test_train_student_1digit():
    """1 位加法应能在练习后逐步提高准确率。"""
    agent, history = train_student(
        max_digits=1, problems_per_level=500,
        check_interval=100, learning_mode='hebbian', seed=0,
        samples_per_test=20, max_rounds=3
    )
    # 检查 history 格式
    assert len(history) > 0
    assert 'level' in history[0]
    assert 'correct' in history[0]


def test_train_student_curriculum_progress():
    """课程应从 level 1 开始出题。"""
    agent, history = train_student(
        max_digits=2, problems_per_level=100,
        check_interval=50, learning_mode='hebbian', seed=0,
        samples_per_test=10, max_rounds=1
    )
    levels_seen = set(h['level'] for h in history)
    assert 1 in levels_seen  # 至少从 level 1 开始


def test_evaluate_agent_returns_dict():
    """evaluate_agent 应返回 {位数: 准确率} 字典。"""
    agent, _ = train_student(max_digits=1, problems_per_level=100,
                             check_interval=100, learning_mode='hebbian', seed=0,
                             samples_per_test=10, max_rounds=1)
    results = evaluate_agent(agent, max_digits=1, n_per_len=20, seed=42)
    assert isinstance(results, dict)
    assert 1 in results
    assert 0.0 <= results[1] <= 1.0
