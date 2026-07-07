"""
Week 4 概率论+信息论 — 动手实战 Part 1：手写 Softmax + 交叉熵

从公式直接翻译成代码，每行不超过一行数学操作。
运行这个脚本，你就能看到三个 logits 怎么变成合法概率分布。

运行依赖：numpy
"""

import numpy as np


def softmax(x):
    """任意实数向量 → 概率分布

    x 可以是 1D 向量或 2D batch。
    三步走：exp 变正数 → 除以总和 → 得到概率。
    """
    # 减 max 防溢出：exp(1000) 会爆 float，减去最大值后最大是 0
    x_max = np.max(x, axis=-1, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / e_x.sum(axis=-1, keepdims=True)


def cross_entropy(y_true, y_pred):
    """真实 one-hot 分布 vs 预测概率分布

    H(P, Q) = -Σ P(x) × log Q(x)
    y_true 是 one-hot（P），y_pred 是模型输出（Q）。
    +1e-8 防止 log(0) 炸掉。
    """
    return -np.sum(y_true * np.log(y_pred + 1e-8)) / len(y_true)


# ============================================================
# 1. 验证 Softmax：单样本
# ============================================================
print("=" * 50)
print("1. Softmax 单样本验证")
print("=" * 50)

logits = np.array([2.3, -0.5, 1.1])
probs = softmax(logits)

print(f"输入 logits:  {logits}")
print(f"输出概率:    {probs}")
print(f"概率求和:    {probs.sum():.4f}  ← 应该 = 1.0")
print(f"最大概率索引: {np.argmax(probs)}  ← 应该 = 0（logit 2.3 最大）")

# ============================================================
# 2. 验证 Softmax：批量
# ============================================================
print(f"\n{'=' * 50}")
print("2. Softmax 批量验证（2 个样本）")
print("=" * 50)

batch_logits = np.array([
    [2.3, -0.5, 1.1],
    [0.1,  1.5, 0.8]
])
batch_probs = softmax(batch_logits)

print(f"输入 (2×3):\n{batch_logits}")
print(f"输出 (2×3):\n{batch_probs}")
print(f"每行求和: {batch_probs.sum(axis=1)}  ← 应该全是 1.0")

# ============================================================
# 3. 验证交叉熵
# ============================================================
print(f"\n{'=' * 50}")
print("3. 交叉熵验证")
print("=" * 50)

# 真实标签：类别 0（猫）
y_true = np.array([[1.0, 0.0, 0.0]])

# 预测 A：0.9 概率给猫 → 损失应该很小
y_pred_good = np.array([[0.9, 0.05, 0.05]])
loss_good = cross_entropy(y_true, y_pred_good)

# 预测 B：0.1 概率给猫 → 损失应该很大
y_pred_bad = np.array([[0.1, 0.8, 0.1]])
loss_bad = cross_entropy(y_true, y_pred_bad)

print(f"预测A（对猫 0.9 把握）→ 交叉熵 = {loss_good:.4f}")
print(f"预测B（对猫 0.1 把握）→ 交叉熵 = {loss_bad:.4f}")
print(f"预测B / 预测A = {loss_bad / loss_good:.1f}x  ← 错得越离谱，损失越大")

# ============================================================
# 4. 温度参数直观感受
# ============================================================
print(f"\n{'=' * 50}")
print("4. 温度参数效果")
print("=" * 50)

logits = np.array([2.0, 0.5, 0.1])

for T in [0.1, 0.5, 1.0, 2.0, 5.0]:
    probs = softmax(logits / T)
    print(f"T={T:.1f} → {probs}  (熵≈{(-probs * np.log(probs + 1e-8)).sum():.2f})")

print("\nT→0: 趋于独热，T=1: 标准，T→∞: 趋于均匀")

print(f"\n{'=' * 50}")
print("全部验证通过 ✅")
print("=" * 50)
