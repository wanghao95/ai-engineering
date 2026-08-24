"""快速解析 MNIST arff.gz（sklearn 的 liac-arff 解析器太慢）"""
import gzip, numpy as np, os

ARFF = os.path.join(os.path.dirname(__file__),
                    'data/openml/openml.org/data/v1/download/52667/mnist_784.arff.gz')
NPZ = os.path.join(os.path.dirname(__file__), 'mnist.npz')

rows = []
with gzip.open(ARFF, 'rt') as f:
    in_data = False
    for line in f:
        if not in_data:
            if line.strip().lower() == '@data':
                in_data = True
            continue
        line = line.strip()
        if not line:
            continue
        rows.append(line.split(','))

arr = np.array(rows, dtype=np.float64)
X, y = arr[:, :784], arr[:, 784].astype(int)
print('loaded:', X.shape, y.shape)
np.savez_compressed(NPZ, X=X, y=y)
print('saved to', NPZ)
