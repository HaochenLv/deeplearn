import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import (
    Tokenizer, AdditionDataset, build_curriculum_examples, build_length_examples,
    make_collate_fn, format_prompt, make_target,
)
from eval import greedy_decode


def compute_loss(model, input_ids, labels, pad_mask):
    logits = model(input_ids, pad_mask)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
    )
    return loss, logits


@torch.no_grad()
def teacher_forced_accuracy(model, dataset, tok: Tokenizer, device, batch_size=64):
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=make_collate_fn(tok))
    correct, total = 0, 0
    model.eval()
    for input_ids, labels, pad_mask in loader:
        logits = model(input_ids.to(device), pad_mask.to(device))
        pred = logits.argmax(-1).cpu()
        mask = labels != -100
        correct += (pred[mask] == labels[mask]).sum().item()
        total += int(mask.sum().item())
    model.train()
    return correct / total if total else 0.0


@torch.no_grad()
def exact_match_accuracy(model, examples, tok: Tokenizer, device, max_new_per_digit=20):
    """自回归 exact-match：对每个样本贪婪解码，整串 target 完全匹配才算对。

    比逐 token 准确率严格得多，反映真实生成能力（含错误累积）。
    """
    model.eval()
    correct = 0
    for (a, b, d) in examples:
        prompt = format_prompt(a, b, d)
        out = greedy_decode(model, tok, prompt, device,
                            max_new_tokens=max_new_per_digit * d + 16)
        if out.strip() == make_target(a, b).strip():
            correct += 1
    model.train()
    return correct / len(examples) if examples else 0.0


def train_stage(model, tok: Tokenizer, stage_d: int, device,
                n_examples=4000, batch_size=64, max_epochs=30,
                lr=1e-3, grad_clip=1.0, grad_threshold_em=0.9, seed=0,
                n_val_em=32, probe_lengths=(1, 2, 3, 4, 5), log_every=1,
                checkpoint_path=None):
    """训练第 stage_d 阶段（课程数据）。

    毕业判据：自回归 exact-match ≥ grad_threshold_em（而非 teacher-forced token 准确率——
    后者会因生成时的错误累积而“假性毕业”）。history 记录 (epoch, em_acc, {L: token_acc}, mean_loss)。
    若提供 checkpoint_path，阶段结束时保存 model.state_dict() 到该路径。
    """
    collate = make_collate_fn(tok)
    train_ds = AdditionDataset(tok, build_curriculum_examples(stage_d, n_examples, seed=seed))
    val_examples = build_length_examples(stage_d, n_val_em, seed=seed + 1000)
    val_ds = AdditionDataset(tok, val_examples)
    probe_ds = {L: AdditionDataset(tok, build_length_examples(L, 64, seed=9000 + L))
                for L in probe_lengths}
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    history = []  # [(epoch, em_acc, {L: token_acc}, mean_loss)]
    model.train()
    for epoch in range(1, max_epochs + 1):
        ep_loss, nb = 0.0, 0
        for input_ids, labels, pad_mask in loader:
            loss, _ = compute_loss(model, input_ids.to(device), labels.to(device), pad_mask.to(device))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            ep_loss += loss.item(); nb += 1
        mean_loss = ep_loss / max(nb, 1)
        token_acc = teacher_forced_accuracy(model, val_ds, tok, device)
        em_acc = exact_match_accuracy(model, val_examples, tok, device)
        probe = {L: teacher_forced_accuracy(model, probe_ds[L], tok, device) for L in probe_lengths}
        history.append((epoch, em_acc, probe, mean_loss))
        if epoch % log_every == 0:
            print(f"[stage {stage_d}] epoch {epoch}: loss={mean_loss:.3f} token_acc={token_acc:.3f} em_acc={em_acc:.3f}")
        if em_acc >= grad_threshold_em:
            print(f"[stage {stage_d}] 毕业 @ epoch {epoch} (em_acc={em_acc:.3f})")
            break

    if checkpoint_path is not None:
        import os
        os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        print(f"[stage {stage_d}] checkpoint saved → {checkpoint_path}")

    return history
