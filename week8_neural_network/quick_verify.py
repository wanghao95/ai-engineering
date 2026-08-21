"""快速验证：5 个 epoch，确认代码正确性"""
import numpy as np, os, sys

NPZ = os.path.join(os.path.dirname(__file__), 'mnist.npz')
d = np.load(NPZ)
X, y = d['X'] / 255.0, d['y']
X_train, y_train = X[:60000], y[:60000]
X_test,  y_test  = X[60000:], y[60000:]

np.random.seed(42)
H = 128
W1 = np.random.randn(784, H) * np.sqrt(2/784)
b1 = np.zeros(H)
W2 = np.random.randn(H, 10) * np.sqrt(2/H)
b2 = np.zeros(10)

def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

lr, epochs, batch = 0.1, 5, 64
for epoch in range(epochs):
    idx = np.random.permutation(len(X_train))
    for start in range(0, len(idx), batch):
        bi = idx[start:start+batch]
        Xb, yb = X_train[bi], y_train[bi]
        N = len(Xb)
        h = np.maximum(0, Xb @ W1 + b1)
        logits = h @ W2 + b2
        probs = softmax(logits)
        loss = -np.log(probs[np.arange(N), yb] + 1e-12).mean()
        dlogits = probs.copy()
        dlogits[np.arange(N), yb] -= 1
        dlogits /= N
        dW2 = h.T @ dlogits
        db2 = dlogits.sum(axis=0)
        dh = dlogits @ W2.T
        dh[h <= 0] = 0
        dW1 = Xb.T @ dh
        db1 = dh.sum(axis=0)
        W1 -= lr*dW1; b1 -= lr*db1
        W2 -= lr*dW2; b2 -= lr*db2
    h = np.maximum(0, X_test @ W1 + b1)
    pred = (h @ W2 + b2).argmax(axis=1)
    acc = (pred == y_test).mean()
    print(f"epoch {epoch+1} | loss {loss:.4f} | test acc {acc*100:.2f}%", flush=True)
