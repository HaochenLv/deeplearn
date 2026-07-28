"""临时诊断脚本：加载最终模型，对指定长度复现测试集，定位错误模式。

用法: python3 diag_len.py [L] [ckpt]   默认 L=3, ckpt=checkpoints/stage4.pt
"""
import sys
import random
import torch

from data import Tokenizer, sample_pair, format_prompt, make_target
from model import AdditionTransformer
from eval import greedy_decode, extract_answer, column_accuracy


def main():
    L = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    ckpt = sys.argv[2] if len(sys.argv) > 2 else "checkpoints/stage4.pt"

    tok = Tokenizer()
    model = AdditionTransformer(tok.vocab_size)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    device = torch.device("cpu")

    rng = random.Random(12345)  # 与 evaluate_lengths 同种子，复现同一批测试样本
    n = 200
    correct = 0
    carry_sum = 0.0
    em = 0
    errors = []
    for _ in range(n):
        a, b = sample_pair(L, rng)
        prompt = format_prompt(a, b, L)
        out = greedy_decode(model, tok, prompt, device, max_new_tokens=20 * L + 16)
        true_ans = str(a + b)
        pred_ans = extract_answer(out)
        ans_ok = pred_ans == true_ans
        em_ok = out.strip() == make_target(a, b).strip()
        ca = column_accuracy(out, a, b)
        correct += ans_ok
        em += em_ok
        carry_sum += ca
        if len(errors) < 10 and not ans_ok:
            errors.append((a, b, true_ans, pred_ans, ca, out))

    print(f"len {L}: answer_acc={correct/n:.3f} exact_match={em/n:.3f} carry_acc={carry_sum/n:.3f}")
    print(f"\n=== {len(errors)} 个错误样本示例（答案错的前 10 个）===")
    for (a, b, ta, pa, ca, out) in errors:
        print(f"\n{a:0{L}d}+{b:0{L}d}={ta} | 预测={pa} | 列进位正确率={ca:.2f}")
        print(f"  生成草稿: {out!r}")
        print(f"  正确草稿: {make_target(a, b)!r}")


if __name__ == "__main__":
    main()
