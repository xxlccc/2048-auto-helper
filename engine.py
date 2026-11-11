# engine.py
"""2048 决策引擎 —— Expectimax + 置换表 + n-tuple 评估"""
import math
import os
import random
from n_tuple import NTupleNet

SIZE = 4
DIRECTIONS = ["up", "down", "left", "right"]

# n-tuple 网络（全局单例，加载后生效）
_ntuple = None
_ntuple_path = "data/model/n_tuple.pkl"


def load_ntuple(path=None):
    """加载 n-tuple 权重，之后 evaluate() 自动使用"""
    global _ntuple
    p = path or _ntuple_path
    if os.path.exists(p):
        _ntuple = NTupleNet(p)
        print(f"n-tuple 已加载: {_ntuple.total_weights} 个非零权重")
    else:
        print(f"n-tuple 权重不存在 ({p})，使用蛇形评估")


def load_extreme(path="极限/model/extreme.pkl"):
    """加载极限密集 n-tuple 模型"""
    global _ntuple
    if os.path.exists(path):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "极限"))
        from n_tuple_dense import DenseNTuple
        _ntuple = DenseNTuple(path)
        print(f"极限模型已加载: {_ntuple.total_weights:,} 个权重")
    else:
        print(f"极限模型不存在 ({path})，请先运行 python 极限/train_extreme.py")


# ---------- 棋盘操作（mirror / transpose 复用） ----------

def _mirror(board):
    return [row[::-1] for row in board]


def _transpose(board):
    return [list(col) for col in zip(*board)]


def _slide_row_left(row):
    non_zero = [v for v in row if v != 0]
    merged = []
    score = 0
    i = 0
    while i < len(non_zero):
        if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
            merged.append(non_zero[i] + 1)
            score += 1 << (non_zero[i] + 1)
            i += 2
        else:
            merged.append(non_zero[i])
            i += 1
    merged.extend([0] * (SIZE - len(merged)))
    return merged, score


def _move_left(board):
    rows = []
    total = 0
    for row in board:
        r, s = _slide_row_left(row)
        rows.append(r)
        total += s
    return rows, total


def move(board, direction):
    if direction == "left":
        result, score = _move_left(board)
        moved = result != board
    elif direction == "right":
        result, score = _move_left(_mirror(board))
        result = _mirror(result)
        moved = result != board
    elif direction == "up":
        result, score = _move_left(_transpose(board))
        result = _transpose(result)
        moved = result != board
    else:  # down
        result, score = _move_left(_mirror(_transpose(board)))
        result = _transpose(_mirror(result))
        moved = result != board
    return result, score, moved


def get_empty_cells(board):
    return [(r, c) for r in range(SIZE) for c in range(SIZE) if board[r][c] == 0]


def spawn_tile(board, position, value):
    r, c = position
    nb = [row[:] for row in board]
    nb[r][c] = value
    return nb


def _board_1d(board_2d):
    """2D 棋盘 → 1D 数组（n-tuple 输入格式）"""
    return [v for row in board_2d for v in row]


# ---------- 评估函数 ----------

# 社区最优蛇形矩阵（来自 caub 的 SO 答案，非均匀衰减）
_SNAKE_RAW = [
    [10,   8,   7,   6.5],
    [0.5,  0.7, 1,   3  ],
    [-0.5, -1.5, -1.8, -2 ],
    [-3.8, -3.7, -3.5, -3 ],
]

# 取 exp 映射到正值权重空间
import math as _math
_SNAKE_BL = [[_math.exp(v) for v in row] for row in _SNAKE_RAW]
_SNAKE_BR = [row[::-1] for row in _SNAKE_BL]
_SNAKE_TL = _SNAKE_BL[::-1]
_SNAKE_TR = [row[::-1] for row in _SNAKE_TL]
_ALL_SNAKES = [_SNAKE_BL, _SNAKE_BR, _SNAKE_TL, _SNAKE_TR]


def _snake_eval(board):
    """蛇形启发式评估（n-tuple 未加载时的回退方案）

    4 因子（权重参考 nneonneo + caub）：
    1. 蛇形角位加权（主导）
    2. 空格奖励
    3. 平滑度
    4. 合并潜力（相邻相同值越多越好）
    """
    empty = len(get_empty_cells(board))

    # 蛇形加权
    best_snake = 0.0
    for snake in _ALL_SNAKES:
        s = 0.0
        for r in range(SIZE):
            for c in range(SIZE):
                v = board[r][c]
                if v > 0:
                    s += (1 << v) * snake[r][c]
        best_snake = max(best_snake, s)

    # 平滑度
    smooth = 0.0
    for r in range(SIZE):
        for c in range(SIZE):
            v = board[r][c]
            if v == 0:
                continue
            if c + 1 < SIZE and board[r][c + 1] != 0:
                smooth -= abs(v - board[r][c + 1])
            if r + 1 < SIZE and board[r + 1][c] != 0:
                smooth -= abs(v - board[r + 1][c])

    # 合并潜力：数相邻相同值对数（nneonneo 启发式）
    merges = 0
    for r in range(SIZE):
        for c in range(SIZE - 1):
            if board[r][c] != 0 and board[r][c] == board[r][c + 1]:
                merges += 1
    for r in range(SIZE - 1):
        for c in range(SIZE):
            if board[r][c] != 0 and board[r][c] == board[r + 1][c]:
                merges += 1

    return best_snake * 1.0 + empty * 5.0 + smooth * 0.5 + merges * 3.0


def evaluate(board):
    """评估棋盘（优先用 n-tuple，否则蛇形）"""
    if _ntuple is not None:
        return _ntuple.evaluate(_board_1d(board))
    return _snake_eval(board)


# ---------- 置换表 ----------

def _board_key(board):
    return tuple(tuple(row) for row in board)


_trans_table = {}


def _tt_lookup(board, depth):
    key = _board_key(board)
    if key in _trans_table:
        cd, cs = _trans_table[key]
        if cd >= depth:
            return cs
    return None


def _tt_store(board, depth, score):
    key = _board_key(board)
    if key not in _trans_table or _trans_table[key][0] < depth:
        _trans_table[key] = (depth, score)


def clear_tt():
    _trans_table.clear()


# ---------- Expectimax ----------

def _order_moves(board):
    scored = []
    for d in DIRECTIONS:
        nb, gain, moved = move(board, d)
        if moved:
            scored.append((evaluate(nb) + gain, d))
    scored.sort(key=lambda x: -x[0])
    return [d for _, d in scored]


def _sample_spawn(board, pos, depth, spawns):
    """对空格 pos，按 spawns 的 (值, 概率) 列表加权求期望"""
    total = 0.0
    for val, prob in spawns:
        b = spawn_tile(board, pos, val)
        s, _ = expectimax(b, depth, True, spawns)
        total += prob * s
    return total


def expectimax(board, depth, is_player_turn, spawns=None):
    """Expectimax 搜索。spawns = [(值, 概率), ...]"""
    if spawns is None:
        spawns = [(1, 0.90), (2, 0.08), (3, 0.02)]

    cached = _tt_lookup(board, depth)
    if cached is not None:
        return cached, None

    if depth == 0:
        ev = evaluate(board)
        _tt_store(board, 0, ev)
        return ev, None

    if is_player_turn:
        best_score = float("-inf")
        best_dir = None
        for direction in _order_moves(board):
            nb, gain, moved = move(board, direction)
            if not moved:
                continue
            cs, _ = expectimax(nb, depth - 1, False, spawns)
            cs += gain
            if cs > best_score:
                best_score = cs
                best_dir = direction
        if best_dir is None:
            ev = evaluate(board)
            _tt_store(board, 0, ev)
            return ev, None
        _tt_store(board, depth, best_score)
        return best_score, best_dir
    else:
        empty = get_empty_cells(board)
        if not empty:
            ev = evaluate(board)
            _tt_store(board, 0, ev)
            return ev, None

        if len(empty) > 8:
            sampled = random.sample(empty, 8)
        else:
            sampled = empty

        total = 0.0
        for pos in sampled:
            total += _sample_spawn(board, pos, depth - 1, spawns)

        return total / len(sampled), None


# ---------- 公开接口 ----------

def get_move(board_4x4, strategy="standard", spawns=None):
    if spawns is None:
        spawns = [(1, 0.90), (2, 0.08), (3, 0.02)]

    board = [list(row) for row in board_4x4]
    empty = len(get_empty_cells(board))
    clear_tt()

    if strategy == "conservative":
        depth = 3 if empty >= 6 else 4
    elif strategy == "aggressive":
        if empty >= 8:
            depth = 5
        elif empty >= 5:
            depth = 7
        elif empty >= 3:
            depth = 9
        else:
            depth = 11
    else:  # standard
        if empty >= 8:
            depth = 3
        elif empty >= 5:
            depth = 4
        elif empty >= 3:
            depth = 6
        else:
            depth = 8

    _, best_dir = expectimax(board, depth, True, spawns)
    if best_dir is None:
        return random.choice(DIRECTIONS)
    return best_dir


def get_available_strategies():
    eval_name = "n-tuple" if _ntuple else "snake+merges"
    return {
        "conservative": f"保守型：深度 3~4（评估: {eval_name}）",
        "standard": f"标准型：深度 3~8 自适应（评估: {eval_name}）",
        "aggressive": f"激进型：深度 5~11（评估: {eval_name}）",
    }
