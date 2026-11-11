# n_tuple.py
"""n-tuple 网络 —— 稀疏字典实现，内存友好

参考: Szubert & Jaśkowski, "Temporal difference learning of
      n-tuple networks for the game 2048", CIG 2014

棋盘 tile 值 = 显示数字（1,2,3,...），0 = 空格
与你的游戏完全一致，无需转换。
"""
import os
import pickle
import numpy as np

# ---- 同构变换 ----

def _get_isomorphic_indexes():
    """生成 D4 二面体群（8 个对称变换）的索引映射

    棋盘坐标 (4×4)：
         0  1  2  3
         4  5  6  7
         8  9 10 11
        12 13 14 15
    """
    idx = np.arange(16).reshape((4, 4))

    transforms = [
        idx,                          # 原始
        idx[::-1, :],                 # 垂直翻转
        idx.T,                        # 转置
        idx.T[:, ::-1],               # 转置 + 水平翻转
        idx[:, ::-1],                 # 水平翻转
        idx[::-1, ::-1],              # 180° 旋转
        idx[:, ::-1].T,               # 水平翻转 + 转置
        idx.T[::-1, :],               # 转置 + 垂直翻转
    ]
    return [t.flatten() for t in transforms]


_ISOMORPHIC = _get_isomorphic_indexes()

# ---- 特征模式（4 组 6 元组） ----

_PATTERNS = [
    [0, 1, 2, 3, 4, 5],       # 左上角 2×3
    [4, 5, 6, 7, 8, 9],       # 中上 2×3
    [0, 1, 2, 4, 5, 6],       # 左上 2×3 偏左
    [4, 5, 6, 8, 9, 10],      # 中上 2×3 偏左
]


def _build_pattern_indexes():
    """为每个模式生成 8 个同构变体的棋盘位置索引"""
    result = []
    for pattern in _PATTERNS:
        variants = []
        for iso in _ISOMORPHIC:
            variants.append(iso[pattern])
        result.append(np.array(variants))
    return result


PATTERN_INDEXES = _build_pattern_indexes()
N_PATTERNS = len(_PATTERNS)
N_ISOMORPHIC = 8  # D4 群的元素数


# ---- n 元组网络 ----

class NTupleNet:
    """稀疏 n 元组网络 —— 用 Python dict 只存非零权重"""

    def __init__(self, model_path=None):
        # 4 张表，每张是 dict: {tuple(6 个 tile 值): float 权重}
        self.tables = [{} for _ in range(N_PATTERNS)]

        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def evaluate(self, board_1d):
        """评估棋盘（1D 16 元素数组）"""
        total = 0.0
        for pi, table in enumerate(self.tables):
            for indexes in PATTERN_INDEXES[pi]:
                key = tuple(int(board_1d[i]) for i in indexes)
                total += table.get(key, 0.0)
        return total

    def update(self, board_1d, error, lr):
        """TD 更新所有权重

        Args:
            board_1d: 1D 16 元素数组
            error: TD 误差
            lr: 学习率
        """
        delta = lr * error / (N_PATTERNS * N_ISOMORPHIC)
        for pi, table in enumerate(self.tables):
            for indexes in PATTERN_INDEXES[pi]:
                key = tuple(int(board_1d[i]) for i in indexes)
                table[key] = table.get(key, 0.0) + delta

    def save(self, path):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.tables, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.tables = pickle.load(f)

    @property
    def total_weights(self):
        return sum(len(t) for t in self.tables)
