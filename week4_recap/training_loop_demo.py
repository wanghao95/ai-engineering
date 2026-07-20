"""
一个极简三层网络的完整训练循环 Demo
=====================================
对应文章：Phase 1 数学基础总复盘 - "用 20 行 NumPy 亲手跑一遍"
四块拼图协同：线性代数(前向变换+参数更新) + 概率论(Softmax) + 信息论(交叉熵) + 微积分(反向传播)

网络结构：输入(2) → 隐藏层(2,无激活) → 输出(2,Softmax)
任务：根据花瓣特征判断鸢尾花种类（二分类）
"""

import numpy as np

# === 初始化（和文中完全一致） ===
x = np.array([1.0, 0.5])               # 输入：标准化后的花瓣长度、宽度
y_true = np.array([1.0, 0.0])          # 真实标签 one-hot：[类别0, 类别1]

W1 = np.array([[1.0, 0.0], [0.0, 1.0]])  # 2×2 单位矩阵（第一层权重）
b1 = np.array([0.0, 0.0])                # 第一层偏置
W2 = np.array([[2.0, -1.0], [-1.0, 1.0]])  # 2×2 矩阵（第二层权重，随机初始化）
b2 = np.array([0.0, 0.0])                # 第二层偏置
lr = 0.1                               # 学习率


def softmax(z):
    """数值稳定的 Softmax 实现"""
    exp_z = np.exp(z - np.max(z))
    return exp_z / exp_z.sum()


# === 训练 50 步 ===
print("=" * 55)
print("四块拼图训练循环 Demo")
print("=" * 55)
print(f"输入 x: {x}")
print(f"真实标签: {y_true}")
print(f"初始 W1:\n{W1}")
print(f"初始 W2:\n{W2}")
print("-" * 55)

for step in range(50):
    # ── 前向传播 ──
    h = W1 @ x + b1                     # 【线性代数】隐藏层：矩阵乘法 + 向量加法
    logits = W2 @ h + b2                # 【线性代数】输出层 logits
    probs = softmax(logits)             # 【概率论】Softmax → 概率分布

    # ── 损失计算 ──
    loss = -np.sum(y_true * np.log(probs + 1e-8))  # 【信息论】交叉熵损失

    # ── 反向传播 ──
    d_logits = probs - y_true           # 【微积分+信息论+概率论】魔法抵消：∂L/∂(logits)
    dW2 = np.outer(h, d_logits)         # 【微积分+线性代数】∂L/∂W2
    db2 = d_logits                      # 【微积分】∂L/∂b2
    dh = d_logits @ W2.T                # 【微积分+线性代数】∂L/∂h
    dW1 = np.outer(x, dh)               # 【微积分+线性代数】∂L/∂W1
    db1 = dh                            # 【微积分】∂L/∂b1

    # ── 参数更新 ──
    W1 -= lr * dW1                      # 【线性代数】W1 := W1 - lr × ∂L/∂W1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

    if step % 10 == 0:
        print(f"Step {step:2d} | Loss: {loss:.4f} | "
              f"probs: [{probs[0]:.3f}, {probs[1]:.3f}]")

print("-" * 55)
print(f"最终 W1:\n{W1}")
print(f"最终 W2:\n{W2}")
print(f"预测: 类别{np.argmax(probs)} (置信度 {probs[np.argmax(probs)]:.3f})")
print("=" * 55)
print("预期：Loss 从 0.128 → 0.05 以下，probs → [1.0, 0.0]")
print("运行 `python training_loop_demo.py` 查看完整训练过程")
print("试着改一改 lr、W2 初始值、训练步数，观察收敛变化")
