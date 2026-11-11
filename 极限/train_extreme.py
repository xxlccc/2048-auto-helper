# 极限/train_extreme.py
"""密集 n-tuple 训练 —— 追求最高分

用法: python 极限/train_extreme.py --games 200000

与稀疏版 tdl_train.py 的区别：
- 全表存储（256 MB），查表 numpy 向量化
- 训练速度快 5~10 倍
- 所有组合都有权重，泛化更强
- 覆盖到 32768 方块
"""
import sys
import os
import argparse
import random
import time
import numpy as np

# 确保能 import 项目根目录的 engine
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from n_tuple_dense import DenseNTuple

# 尝试加载 Numba 加速引擎
try:
    from board_fast import play_game_fast as _play_fast, move_b, random_popup
    _USE_FAST = True
except ImportError:
    _USE_FAST = False
    print("⚠ Numba 加速未启用，使用普通引擎")


# ---- 游戏模拟 ----

def _board_1d(board_2d):
    return [v for row in board_2d for v in row]


def play_game(net, search_depth=0, spawn_low=1, spawn_high=2, spawn_low_prob=0.9):
    """模拟一局"""
    if _USE_FAST:
        return _play_fast(net, search_depth, spawn_low, spawn_high, spawn_low_prob)
    else:
        return _play_slow(net, search_depth, spawn_low, spawn_high, spawn_low_prob)


def _play_slow(net, search_depth, spawn_low, spawn_high, spawn_low_prob):
    """回退：普通 Python 游戏模拟"""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from engine import move, get_empty_cells, spawn_tile, DIRECTIONS

    board = [[0] * 4 for _ in range(4)]

    def popup(b):
        empty = get_empty_cells(b)
        if not empty:
            return b
        r, c = random.choice(empty)
        val = spawn_low if random.random() < spawn_low_prob else spawn_high
        return spawn_tile(b, (r, c), val)

    board = popup(board)
    board = popup(board)
    path = []
    score = 0

    while True:
        best_after = None
        best_val = float("-inf")
        best_reward = 0
        for d in DIRECTIONS:
            after, reward, moved = move(board, d)
            if not moved:
                continue
            v = net.evaluate(_board_1d(after))
            if reward + v > best_val:
                best_val = reward + v
                best_after = after
                best_reward = reward

        if best_after is None:
            break

        path.append((_board_1d(best_after), best_val - best_reward))
        board = popup(best_after)
        score += best_reward

    max_tile = max(max(row) for row in board)
    return path, score, max_tile


# ---- TD(0) 学习 ----

def td_learn(path, net, lr):
    if len(path) < 2:
        return
    net.update(path[-1][0], 0 - path[-1][1], lr)
    for i in range(len(path) - 2, -1, -1):
        error = path[i + 1][1] - path[i][1]
        net.update(path[i][0], error, lr)


# ---- 训练主循环 ----

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=200000)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--search-depth", type=int, default=0,
                        help="训练搜索深度: 0=贪心(推荐)")
    parser.add_argument("--model", default=os.path.join(os.path.dirname(__file__),
                             "model", "extreme.pkl"))
    args = parser.parse_args()

    if args.resume and os.path.exists(args.model):
        print(f"从 {args.model} 继续训练")
        net = DenseNTuple(args.model)
    else:
        print("从头训练 密集 n-tuple 网络")
        print(f"表大小: 4 × 16^6 = {4 * 16**6:,} 个权重 ({4*16**6*4/1024/1024:.0f} MB)")
        net = DenseNTuple()

    # 训练固定用标准分布（通用模型），游戏时 config.yaml 的 level 控制实际生成概率
    spawn_low, spawn_high, spawn_low_prob = 1, 2, 0.9

    print(f"训练局数: {args.games:,}")
    print(f"学习率: {args.lr}（三段衰减）")
    print(f"搜索深度: {args.search_depth}")
    print(f"加速引擎: {'Numba JIT' if _USE_FAST else 'Python（慢）'}")
    print(f"模型路径: {args.model}")
    print()

    scores = []
    max_tiles = []
    report_interval = min(200, max(10, args.games // 50))

    t0 = time.time()
    for game in range(1, args.games + 1):
        # 学习率衰减：前 1/3 保持，中 1/3 减半，后 1/3 再减半
        progress = game / args.games
        if progress < 0.33:
            current_lr = args.lr
        elif progress < 0.66:
            current_lr = args.lr * 0.5
        else:
            current_lr = args.lr * 0.25

        path, score, max_tile = play_game(
            net, args.search_depth, spawn_low, spawn_high, spawn_low_prob
        )
        td_learn(path, net, current_lr)

        scores.append(score)
        max_tiles.append(max_tile)

        if game % report_interval == 0 or game == 1:
            elapsed = time.time() - t0
            speed = game / elapsed if elapsed > 0 else 0
            eta = (args.games - game) / speed if speed > 0 else 0

            recent = scores[-report_interval:]
            recent_tiles = max_tiles[-report_interval:]
            avg_score = int(np.mean(recent))
            max_score = int(np.max(recent))
            max_t = max(recent_tiles)

            tile_counts = {}
            for t in recent_tiles:
                tile_counts[t] = tile_counts.get(t, 0) + 1

            tile_str = " | ".join(
                f"{t}:{tile_counts[t]/report_interval*100:.0f}%"
                for t in sorted(tile_counts.keys(), reverse=True)[:5]
            )

            print(
                f"[{game:7d}/{args.games}] "
                f"avg={avg_score:7d}  max={max_score:8d}  "
                f"best={max_t}  lr={current_lr:.3f}  "
                f"({tile_str})  "
                f"d={args.search_depth}  "
                f"{elapsed:.0f}s  eta:{eta:.0f}s"
            )

        if game % args.save_every == 0:
            net.save(args.model)
            print(f"  → 已保存: {args.model}")

    net.save(args.model)
    elapsed = time.time() - t0
    print(f"\n✅ 训练完成！{args.games:,} 局 / {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    print(f"   模型: {args.model}")

    # 打印非零权重统计
    stats = net.stats()
    print(f"\n   表统计:")
    for i, nz, mn, mx, avg in stats:
        print(f"   [{i}] 非零: {nz:,} / {16**6:,}  "
              f"范围: [{mn:.3f}, {mx:.3f}]  均值: {avg:.6f}")


if __name__ == "__main__":
    main()
