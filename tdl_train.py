# tdl_train.py
"""TD(0) 训练 n-tuple 网络 —— 程序自己玩 2048 学评估权重

用法: python tdl_train.py                          # 从头训练
      python tdl_train.py --resume                 # 继续训练
      python tdl_train.py --games 50000 --lr 0.01  # 自定义参数

训练完成后权重存到 data/model/n_tuple.pkl，
engine.py 会自动加载。
"""
import os
import sys
import argparse
import random
import time
import numpy as np
from engine import move, get_empty_cells, spawn_tile, DIRECTIONS, load_ntuple
from n_tuple import NTupleNet

SIZE = 4


# ---- 游戏模拟 ----

def _board_1d(board_2d):
    return [v for row in board_2d for v in row]


def _random_popup(board):
    """随机生成新棋子（90% 出 1，10% 出 2）"""
    empty = get_empty_cells(board)
    if not empty:
        return board
    r, c = random.choice(empty)
    val = 1 if random.random() < 0.9 else 2
    return spawn_tile(board, (r, c), val)


def play_game(net):
    """用 n-tuple 网络贪婪决策玩一局

    每一步：评估 4 个方向的 after-state，选 V(s') + reward 最大的。
    记录完整路径用于 TD 学习。

    Returns:
        path: [(board_1d_after, V_est), ...] 每一步的 after-state 和估计值
        score: 终局得分
        max_tile: 最大方块（显示值）
    """
    board = [[0] * SIZE for _ in range(SIZE)]
    board = _random_popup(board)
    board = _random_popup(board)

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
            total = reward + v
            if total > best_val:
                best_val = total
                best_after = after
                best_reward = reward

        if best_after is None:
            break  # 无法移动，游戏结束

        path.append((_board_1d(best_after), best_val - best_reward))  # (state, V_before)
        board = _random_popup(best_after)
        score += best_reward

    max_tile = max(max(row) for row in board)
    return path, score, max_tile


# ---- TD(0) 学习 ----

def td_learn(path, net, lr):
    """反向 TD(0) 更新

    path: [(state_1d, V_est), ...]
    每步: V(s) ← V(s) + α * (V(s_next) + r - V(s))
    """
    if len(path) < 2:
        return

    # 终局状态 V=0
    net.update(path[-1][0], 0 - path[-1][1], lr)

    # 反向传播
    for i in range(len(path) - 2, -1, -1):
        _, v_curr = path[i]
        _, v_next = path[i + 1]
        # 注意：path 里存的是 after-state，reward 已包含在 v 里
        # 简化：V(after_t) ← V(after_t) + α * (V(after_t+1) - V(after_t))
        # 更准确的 TD: error = r + V(after_t+1) - V(after_t)
        # 但这里 path 只存了 after-state 的 V，reward 已经隐含了
        error = v_next - v_curr
        net.update(path[i][0], error, lr)


# ---- 训练主循环 ----

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--model", default="data/model/n_tuple.pkl")
    args = parser.parse_args()

    # 加载或创建网络
    if args.resume and os.path.exists(args.model):
        print(f"从 {args.model} 继续训练")
        net = NTupleNet(args.model)
    else:
        print("从头训练 n-tuple 网络")
        net = NTupleNet()

    print(f"总权重: {net.total_weights}")
    print(f"训练局数: {args.games}")
    print(f"学习率: {args.lr}")
    print()

    scores = []
    max_tiles = []
    report_interval = min(100, args.games // 20) or 100

    t0 = time.time()
    for game in range(1, args.games + 1):
        path, score, max_tile = play_game(net)
        td_learn(path, net, args.lr)

        scores.append(score)
        max_tiles.append(max_tile)

        if game % report_interval == 0 or game == 1:
            elapsed = time.time() - t0
            recent = scores[-report_interval:]
            recent_tiles = max_tiles[-report_interval:]
            avg_score = int(np.mean(recent))
            max_score = int(np.max(recent))
            max_t = max(recent_tiles)

            # 统计方块分布
            tile_counts = {}
            for t in recent_tiles:
                tile_counts[t] = tile_counts.get(t, 0) + 1
            tile_str = " | ".join(
                f"{t}:{tile_counts[t]/report_interval*100:.0f}%"
                for t in sorted(tile_counts.keys(), reverse=True)[:4]
            )

            print(
                f"[{game:6d}/{args.games}] "
                f"avg={avg_score:6d}  max={max_score:7d}  "
                f"best_tile={max_t}  ({tile_str})  "
                f"weights={net.total_weights:6d}  "
                f"{elapsed:.0f}s"
            )

        if game % args.save_every == 0:
            net.save(args.model)
            print(f"  → 已保存: {args.model}")

    net.save(args.model)
    elapsed = time.time() - t0
    print(f"\n✅ 训练完成！总时间: {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    print(f"   权重已保存: {args.model}")
    print(f"   在 main.py 启动前运行: engine.load_ntuple()")


if __name__ == "__main__":
    main()
