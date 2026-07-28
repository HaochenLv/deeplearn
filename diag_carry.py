"""诊断：MLP（无显式进位监督）在「涉及进位」vs「不涉及进位」样本上的表现差异。

若进位样本准确率明显低于非进位样本 → 进位是弱点，教学有价值。
"""
import random
import torch
from mlp_addition import train, encode_input, decode_digits

MD = 4
print(f"训练 {MD} 位 MLP（hidden=256, 20 epoch）...")
model = train(MD, epochs=20, hidden=256, n_per_epoch=40000)
model.eval()


def has_carry(a, b, md):
    """任意一位产生向高位的进位 → True。"""
    x, y, cin = a, b, 0
    for _ in range(md):
        s = (x % 10) + (y % 10) + cin
        if s >= 10:
            return True
        cin = 0
        x //= 10
        y //= 10
    return False


rng = random.Random(4321)
hi = 10 ** MD
cats = {"carry": [0, 0], "nocarry": [0, 0]}   # [correct, total]
with torch.no_grad():
    for _ in range(5000):
        a = rng.randrange(hi)
        b = rng.randrange(hi)
        pred = model(encode_input(a, b, MD).unsqueeze(0)).argmax(-1)[0].tolist()
        ok = decode_digits(pred) == a + b
        k = "carry" if has_carry(a, b, MD) else "nocarry"
        cats[k][0] += ok
        cats[k][1] += 1

print("\n=== 进位诊断（5000 样本）===")
for k, (c, t) in cats.items():
    print(f"  {k:8s}: {c/t:.4f}   ({t} 样本)")
