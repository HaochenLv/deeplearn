"""三种学习模式对比实验：Hebbian / Reward / Correction。

3 个模型配合：DigitAdder(前馈) + CarryPropagator(前馈) + LearningMechanism
进位传递用前馈模型（cout→cin 单步映射），FA 能完整训练，N 位泛化。
对比 3 种学习方案，选出最好的一个。
"""
import os
from online_addition import StudentAgent, train_student, evaluate_agent, plot_learning_curve

MODES = ['hebbian', 'reward', 'correction']
MAX_DIGITS = 5
RESULTS_DIR = 'outputs'

os.makedirs(RESULTS_DIR, exist_ok=True)

all_results = {}

for mode in MODES:
    print(f"\n{'='*60}")
    print(f"  学习模式: {mode}")
    print(f"{'='*60}")

    agent = StudentAgent(learning_mode=mode, init_eta=0.1, init_gamma=0.0)
    n_inference = sum(p.numel() for p in agent.student_adder.parameters()) + \
                  sum(p.numel() for p in agent.student_carry.parameters())
    n_learner = agent.learner.get_param_count()
    print(f"推理模型参数: {n_inference}  学习机制参数: {n_learner}")

    agent, history = train_student(
        max_digits=MAX_DIGITS, problems_per_level=2000,
        check_interval=200, learning_mode=mode, seed=0,
        samples_per_test=50, max_rounds=5,
        init_eta=0.1, init_gamma=0.0,
        consolidation_steps=5000,
    )

    # 评估：1-5 位 + 泛化到 8/12/16/20/30/50 位
    eval_lengths = [1, 2, 3, 4, 5, 8, 12, 16, 20, 30, 50]
    results = evaluate_agent(agent, max_digits=MAX_DIGITS, n_per_len=200, seed=999,
                             eval_lengths=eval_lengths)
    print(f"\n最终评估 ({mode}):")
    for L, acc in results.items():
        print(f"  {L:2d} 位: {acc:.4f}")

    all_results[mode] = {
        'inference_params': n_inference,
        'learner_params': n_learner,
        'accuracy': results,
    }

    plot_learning_curve(history, save_path=f"{RESULTS_DIR}/learning_{mode}.png")

# ---- 汇总对比 ----
print(f"\n{'='*60}")
print("  三种模式对比汇总（前馈 CarryPropagator）")
print(f"{'='*60}")
header_lengths = [1, 2, 3, 4, 5, 8, 12, 16, 20, 30, 50]
print(f"{'模式':<12} {'推理参数':>8} {'学习参数':>8}  ", end='')
for L in header_lengths:
    print(f"{L}位   ", end='')
print()
print('-' * 120)
for mode in MODES:
    r = all_results[mode]
    print(f"{mode:<12} {r['inference_params']:>8} {r['learner_params']:>8}  ", end='')
    for L in header_lengths:
        acc = r['accuracy'].get(L, 0)
        print(f"{acc:.3f} ", end='')
    print()

print(f"\n图表保存在 {RESULTS_DIR}/ 目录")
