 """
50 行 NumPy 手写神经网络 —— MNIST 手写数字识别
不依赖 PyTorch / TensorFlow，只用 NumPy 实现前向传播 + 反向传播。
"""
import numpy as np

# ---------- 1. 加载数据 ----------
import os
from sklearn.datasets import fetch_openml
DATA_HOME = os.path.join(os.path.dirname(__file__), 'data')   # 缓存到项目目录
mnist = fetch_openml('mnist_784', version=1, as_frame=False,
                     parser='liac-arff', data_home=DATA_HOME)
X, y = mnist.data / 255.0, mnist.target.astype(int)          # 归一化到 [0,1]
X_train, y_train = X[:60000], y[:60000]
X_test,  y_test  = X[60000:], y[60000:]

# ---------- 2. 初始化参数（He 初始化）----------
np.random.seed(42)
H = 128                                        # 隐藏层神经元数
W1 = np.random.randn(784, H) * np.sqrt(2/784)
b1 = np.zeros(H)
W2 = np.random.randn(H, 10) * np.sqrt(2/H)
b2 = np.zeros(10)

def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

# ---------- 3. 训练循环 ----------
lr, epochs, batch = 0.1, 20, 64
history = []
for epoch in range(epochs):
    idx = np.random.permutation(len(X_train))
    for start in range(0, len(idx), batch):
        bi = idx[start:start+batch]
        Xb, yb = X_train[bi], y_train[bi]
        N = len(Xb)

        # 前向传播
        h = np.maximum(0, Xb @ W1 + b1)        # 隐藏层 + ReLU
        logits = h @ W2 + b2                   # 输出层
        probs = softmax(logits)                # 概率
        loss = -np.log(probs[np.arange(N), yb] + 1e-12).mean()

        # 反向传播（链式法则逐层回传）
        dlogits = probs.copy()
        dlogits[np.arange(N), yb] -= 1
        dlogits /= N                           # softmax+交叉熵的梯度
        dW2 = h.T @ dlogits
        db2 = dlogits.sum(axis=0)
        dh = dlogits @ W2.T
        dh[h <= 0] = 0                         # ReLU 梯度：负数清零
        dW1 = Xb.T @ dh
        db1 = dh.sum(axis=0)

        # 参数更新
        W1 -= lr*dW1; b1 -= lr*db1
        W2 -= lr*dW2; b2 -= lr*db2

    # 每个 epoch 记录一次测试准确率
    h = np.maximum(0, X_test @ W1 + b1)
    pred = (h @ W2 + b2).argmax(axis=1)
    acc = (pred == y_test).mean()
    history.append((epoch+1, loss, acc))
    print(f"epoch {epoch+1:2d} | loss {loss:.4f} | test acc {acc*100:.2f}%")

print("\n最终测试准确率:", f"{history[-1][2]*100:.2f}%")
