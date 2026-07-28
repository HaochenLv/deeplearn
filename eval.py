import re
import torch

from data import Tokenizer, sample_pair, format_prompt, make_target, make_columns


@torch.no_grad()
def greedy_decode(model, tok: Tokenizer, prompt_str: str, device, max_new_tokens=128):
    """返回模型生成的 target 文本（不含 bos/prompt/eos）。"""
    model.eval()
    prompt_ids = tok.encode(prompt_str)
    ids = [tok.bos_id] + prompt_ids
    generated = []
    for _ in range(max_new_tokens):
        x = torch.tensor([ids], device=device, dtype=torch.long)
        logits = model(x)
        next_id = int(logits[0, -1].argmax().item())
        if next_id == tok.eos_id:
            break
        generated.append(next_id)
        ids.append(next_id)
    return tok.decode(generated)


def column_accuracy(text: str, a: int, b: int):
    """草稿每列 d<digit> c<carry> 与真值的逐列匹配率（进位诊断）。"""
    pairs = re.findall(r"d(\d)\s*c(\d)", text)
    true = [(dout, cout) for (_, _, _, dout, cout) in make_columns(a, b)]
    if not true:
        return 1.0
    correct = sum(1 for i, (d, c) in enumerate(pairs)
                  if i < len(true) and (int(d), int(c)) == true[i])
    return correct / len(true)


def extract_answer(text: str):
    """从生成文本末尾的 '=答案' 提取答案数字串；无则 None。"""
    idx = text.rfind("=")
    if idx < 0:
        return None
    tail = text[idx + 1:]
    ans = "".join(ch for ch in tail if ch.isdigit())
    return ans if ans else None


@torch.no_grad()
def evaluate_lengths(model, tok: Tokenizer, lengths, n: int, device, seed=12345):
    import random
    rng = random.Random(seed)
    results = {}
    model.eval()
    for L in lengths:
        correct, em, carry_tot, total = 0, 0, 0.0, 0
        for _ in range(n):
            a, b = sample_pair(L, rng)
            prompt = format_prompt(a, b, L)
            true_ans = str(a + b)
            out = greedy_decode(model, tok, prompt, device, max_new_tokens=20 * L + 16)
            pred_ans = extract_answer(out)
            total += 1
            if pred_ans == true_ans:
                correct += 1
            if out.strip() == make_target(a, b).strip():
                em += 1
            carry_tot += column_accuracy(out, a, b)
        results[L] = {
            "answer_acc": correct / total,
            "exact_match": em / total,
            "carry_acc": carry_tot / total,
        }
    return results
