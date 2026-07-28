import os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import Tokenizer
from model import AdditionTransformer
from train import train_stage
from eval import evaluate_lengths

OUT_DIR = "outputs"
CKPT_DIR = "checkpoints"


def plot_extrapolation(lengths, accs, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(lengths, accs, marker="o")
    plt.axvline(4.5, color="r", linestyle="--", label="训练长度上限 (4)")
    plt.xlabel("位数")
    plt.ylabel("答案正确率")
    plt.title("长度外推曲线")
    plt.ylim(-0.05, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_forgetting(history, lengths, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.figure(figsize=(6, 4))
    steps = list(range(len(history)))
    for L in lengths:
        accs = [h[2].get(L) for h in history]
        plt.plot(steps, accs, marker="o", label=f"len {L}")
    plt.xlabel("训练步（跨阶段）")
    plt.ylabel("token 准确率")
    plt.title("遗忘探针")
    plt.ylim(-0.05, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = Tokenizer()
    model = AdditionTransformer(tok.vocab_size).to(device)

    all_history = []
    for stage_d in range(1, 5):
        print(f"\n=== 课程阶段 {stage_d} ===")
        hist = train_stage(model, tok, stage_d, device, probe_lengths=(1, 2, 3, 4))
        all_history.extend(hist)
        torch.save(model.state_dict(), os.path.join(CKPT_DIR, f"stage{stage_d}.pt"))

    print("\n=== 评估 1–8 位 ===")
    results = evaluate_lengths(model, tok, range(1, 9), n=200, device=device)
    for L in range(1, 9):
        r = results[L]
        print(f"len {L}: answer_acc={r['answer_acc']:.3f} exact_match={r['exact_match']:.3f}")

    lengths = list(range(1, 9))
    accs = [results[L]["answer_acc"] for L in lengths]
    plot_extrapolation(lengths, accs, os.path.join(OUT_DIR, "extrapolation.png"))
    plot_forgetting(all_history, [1, 2, 3], os.path.join(OUT_DIR, "forgetting.png"))
    print(f"\n图已保存到 {OUT_DIR}/")


if __name__ == "__main__":
    main()
