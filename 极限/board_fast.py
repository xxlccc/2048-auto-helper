# 极限/board_fast.py
"""Numba JIT 加速棋盘操作 —— 1D numpy 数组 + 编译到机器码"""
import numpy as np
from numba import njit

SIZE = 4
DIRECTIONS = ("up", "down", "left", "right")


@njit
def _shift_left(row):
    """滑动一行（4 元素），返回 (新行, 得分)"""
    # 去掉 0
    nz = row[row != 0]
    n = len(nz)
    result = np.zeros(4, dtype=np.int64)
    score = 0
    pos = 0
    i = 0
    while i < n:
        if i + 1 < n and nz[i] == nz[i + 1]:
            result[pos] = nz[i] + 1
            score += 1 << (nz[i] + 1)
            i += 2
        else:
            result[pos] = nz[i]
            i += 1
        pos += 1
    return result, score


@njit
def _mirror(b):
    """水平翻转"""
    r = np.zeros(16, dtype=np.int64)
    for i in range(4):
        s = i * 4
        r[s] = b[s + 3]
        r[s + 1] = b[s + 2]
        r[s + 2] = b[s + 1]
        r[s + 3] = b[s]
    return r


@njit
def _transpose(b):
    """转置"""
    r = np.zeros(16, dtype=np.int64)
    for i in range(4):
        for j in range(4):
            r[i * 4 + j] = b[j * 4 + i]
    return r


@njit
def _move_left(b):
    """整个棋盘左滑"""
    r = np.zeros(16, dtype=np.int64)
    sc = 0
    for i in range(4):
        s = i * 4
        row, rs = _shift_left(b[s:s + 4])
        r[s:s + 4] = row
        sc += rs
    return r, sc


@njit
def _equal(a, b):
    """比较两个 1D 棋盘是否相同"""
    for i in range(16):
        if a[i] != b[i]:
            return False
    return True


def move_b(board_1d, direction):
    """Numba 加速滑动。返回 (新棋盘, 得分, 是否有移动)"""
    if direction == "left":
        r, sc = _move_left(board_1d)
        moved = not _equal(r, board_1d)
    elif direction == "right":
        r, sc = _move_left(_mirror(board_1d))
        r = _mirror(r)
        moved = not _equal(r, board_1d)
    elif direction == "up":
        r, sc = _move_left(_transpose(board_1d))
        r = _transpose(r)
        moved = not _equal(r, board_1d)
    else:  # down
        m = _mirror(_transpose(board_1d))
        r, sc = _move_left(m)
        r = _transpose(_mirror(r))
        moved = not _equal(r, board_1d)
    return r, sc, moved


@njit
def _random_popup(board_1d, spawn_low=1, spawn_high=2, spawn_low_prob=0.9):
    """随机生成棋子（修改传入数组）"""
    empty = np.where(board_1d == 0)[0]
    if len(empty) == 0:
        return board_1d
    idx = empty[np.random.randint(0, len(empty))]
    val = spawn_low if np.random.random() < spawn_low_prob else spawn_high
    board_1d[idx] = val
    return board_1d, idx, val


def random_popup(board_1d, spawn_low=1, spawn_high=2, spawn_low_prob=0.9):
    """随机生成棋子，返回新棋盘（不修改原数组）"""
    b = board_1d.copy()
    empty = np.where(b == 0)[0]
    if len(empty) == 0:
        return b
    idx = np.random.choice(empty)
    b[idx] = spawn_low if np.random.random() < spawn_low_prob else spawn_high
    return b


# ---- 期望搜索（训练时用浅搜索提升数据质量） ----

def expectimax_move(board_1d, net, depth, spawn_low=1, spawn_high=2, spawn_low_prob=0.9):
    """Expectimax 选最佳移动"""
    best_after = None
    best_val = float("-inf")
    best_reward = 0

    for d in DIRECTIONS:
        after, reward, moved = move_b(board_1d, d)
        if not moved:
            continue

        if depth <= 0:
            val = net.evaluate(after)
        else:
            empty = np.where(after == 0)[0]
            if len(empty) == 0:
                val = net.evaluate(after)
            else:
                if len(empty) > 8:
                    empty = np.random.choice(empty, 8, replace=False)
                total = 0.0
                for pos in empty:
                    b1 = after.copy()
                    b1[pos] = spawn_low
                    s1, _ = _eval_deep(b1, net, depth - 1, spawn_low, spawn_high, spawn_low_prob)

                    b2 = after.copy()
                    b2[pos] = spawn_high
                    s2, _ = _eval_deep(b2, net, depth - 1, spawn_low, spawn_high, spawn_low_prob)

                    total += spawn_low_prob * s1 + (1 - spawn_low_prob) * s2
                val = total / len(empty)

        val += reward
        if val > best_val:
            best_val = val
            best_after = after.copy()
            best_reward = reward

    return best_after, best_reward


def _eval_deep(board_1d, net, depth, spawn_low, spawn_high, spawn_low_prob):
    """递归评估（player 节点：选最佳移动）"""
    best_val = float("-inf")
    best_move = None

    for d in DIRECTIONS:
        after, reward, moved = move_b(board_1d, d)
        if not moved:
            continue
        val = net.evaluate(after) + reward
        if val > best_val:
            best_val = val
            best_move = d

    if best_move is None:
        return net.evaluate(board_1d), None
    return best_val, best_move


# ---- 完整一局 ----

def play_game_fast(net, search_depth=0, spawn_low=1, spawn_high=2, spawn_low_prob=0.9):
    """Numba 加速游戏模拟"""
    board = np.zeros(16, dtype=np.int64)
    board = random_popup(board, spawn_low, spawn_high, spawn_low_prob)
    board = random_popup(board, spawn_low, spawn_high, spawn_low_prob)

    path = []
    score = 0

    while True:
        if search_depth <= 0:
            best_after = None
            best_val = float("-inf")
            best_reward = 0
            for d in DIRECTIONS:
                after, reward, moved = move_b(board, d)
                if not moved:
                    continue
                v = net.evaluate(after)
                if reward + v > best_val:
                    best_val = reward + v
                    best_after = after
                    best_reward = reward
        else:
            best_after, best_reward = expectimax_move(
                board, net, search_depth, spawn_low, spawn_high, spawn_low_prob
            )
            if best_after is not None:
                best_val = net.evaluate(best_after) + best_reward

        if best_after is None:
            break

        path.append((best_after.copy(), best_val - best_reward))
        board = random_popup(best_after, spawn_low, spawn_high, spawn_low_prob)
        score += best_reward

    max_tile = int(board.max())
    return path, score, max_tile
