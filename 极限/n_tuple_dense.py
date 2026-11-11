# 极限/n_tuple_dense.py
"""密集 n-tuple 网络 —— numpy 向量化，全表存储

与稀疏版不同，16^6 = 1678 万个权重全部存储为 float32 数组，
查表用 numpy 向量索引，不依赖 Python dict。
4 张表 × 16.7M × 4 字节 ≈ 256 MB 内存。
"""
import os
import pickle
import numpy as np

N_TUPLE_SIZE = 6
VALUE_RANGE = 16        # 0~15 覆盖到 32768 方块
TABLE_SIZE = VALUE_RANGE ** N_TUPLE_SIZE  # 16,777,216

# 索引基数：[16^5, 16^4, 16^3, 16^2, 16, 1]
_BASE = np.array([VALUE_RANGE ** i for i in reversed(range(N_TUPLE_SIZE))],
                 dtype=np.int64)

# ---- 同构变换（与 n_tuple.py 保持一致） ----

def _get_isomorphic():
    idx = np.arange(16).reshape((4, 4))
    transforms = [
        idx,
        idx[::-1, :],
        idx.T,
        idx.T[:, ::-1],
        idx[:, ::-1],
        idx[::-1, ::-1],
        idx[:, ::-1].T,
        idx.T[::-1, :],
    ]
    return [t.flatten() for t in transforms]

_ISOMORPHIC = _get_isomorphic()

_PATTERNS = [
    [0, 1, 2, 3, 4, 5],
    [4, 5, 6, 7, 8, 9],
    [0, 1, 2, 4, 5, 6],
    [4, 5, 6, 8, 9, 10],
]


def _build_pattern_indexes():
    result = []
    for pat in _PATTERNS:
        variants = np.array([iso[pat] for iso in _ISOMORPHIC], dtype=np.int64)
        result.append(variants)
    return result


PATTERN_INDEXES = _build_pattern_indexes()
N_PATTERNS = len(_PATTERNS)
N_ISOMORPHIC = 8


# ---- 密集 n 元组网络 ----

class DenseNTuple:
    """密集 n-tuple —— numpy 向量化实现

    4 张 float32 数组，每张 16.7M 元素，总计 ~256 MB。
    查表纯 numpy 操作，比稀疏 dict 快 5~10 倍。
    """

    def __init__(self, path=None):
        self.tables = [np.zeros(TABLE_SIZE, dtype=np.float32)
                       for _ in range(N_PATTERNS)]
        if path and os.path.exists(path):
            self.load(path)

    def evaluate(self, board_1d):
        """评估棋盘 —— 纯 numpy 向量化"""
        board = np.asarray(board_1d, dtype=np.int64)
        total = np.float32(0.0)
        for pi in range(N_PATTERNS):
            # (8, 6) 位置索引 → (8,) 复合索引 → 求和
            idx = board[PATTERN_INDEXES[pi]] @ _BASE
            total += self.tables[pi][idx].sum()
        return float(total)

    def update(self, board_1d, error, lr):
        """TD 更新 —— 纯 numpy 向量化"""
        board = np.asarray(board_1d, dtype=np.int64)
        delta = np.float32(lr * error / (N_PATTERNS * N_ISOMORPHIC))
        for pi in range(N_PATTERNS):
            idx = board[PATTERN_INDEXES[pi]] @ _BASE
            self.tables[pi][idx] += delta

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.tables, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.tables = pickle.load(f)

    @property
    def total_weights(self):
        return sum(t.size for t in self.tables)

    def stats(self):
        """各表的统计信息"""
        rows = []
        for i, t in enumerate(self.tables):
            nz = int((t != 0).sum())
            rows.append((i, nz, float(t.min()), float(t.max()), float(t.mean())))
        return rows


# ---- 快速测试 ----

if __name__ == "__main__":
    net = DenseNTuple()
    board = [0, 1, 2, 3, 0, 5, 0, 7, 0, 0, 0, 1, 0, 0, 0, 0]
    print(f"评估: {net.evaluate(board):.4f}")
    net.update(board, 100.0, 0.1)
    print(f"更新后评估: {net.evaluate(board):.4f}")
    print(f"总权重: {net.total_weights:,}")
