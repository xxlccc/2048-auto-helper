# main.py
"""2048 自动助手 —— 识别 + 决策 + 执行 + 修正重训"""
import os
import sys
import json
import time
import yaml
import cv2
import numpy as np
from adb_utils import adb_screenshot
from board_selector import select_board_region
from board_splitter import crop_board, split_board
from infer import TileRecognizer
from engine import get_move, get_available_strategies, load_ntuple
from executor import execute_swipe

BOARD_CACHE = "data/model/board_rect.json"
CONFIG_PATH = "config.yaml"
MAX_DISPLAY_W = 600
MAX_DISPLAY_H = 900


def _load_config():
    """加载 config.yaml"""
    if not os.path.exists(CONFIG_PATH):
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _save_config(cfg):
    """保存 config.yaml"""
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)


# ---------- 缓存 ----------

def save_board_rect(rect):
    os.makedirs(os.path.dirname(BOARD_CACHE), exist_ok=True)
    with open(BOARD_CACHE, "w") as f:
        json.dump(rect, f)


def load_board_rect():
    if os.path.exists(BOARD_CACHE):
        with open(BOARD_CACHE, "r") as f:
            return json.load(f)
    return None


# ---------- 显示 ----------

def _fit_to_screen(img):
    h, w = img.shape[:2]
    scale = min(MAX_DISPLAY_W / w, MAX_DISPLAY_H / h, 1.0)
    if scale >= 1.0:
        return img, 1.0
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h)), scale


def _print_board(board, confs, uncertain_mask=None):
    """格式化打印棋盘，不确定格用 ? 标记"""
    sep = "┌" + "┬".join(["──────"] * 4) + "┐"
    mid = "├" + "┼".join(["──────"] * 4) + "┤"
    end = "└" + "┴".join(["──────"] * 4) + "┘"

    print(sep)
    for r in range(4):
        parts = []
        for c in range(4):
            val = board[r, c]
            cf = confs[r, c]
            if val == 0:
                s = f"  ·   "
            else:
                marker = "?" if (uncertain_mask is not None and uncertain_mask[r, c]) else " "
                s = f"{marker}{val}({cf:.0%})"
            parts.append(s.ljust(6)[:6])
        print("│" + "│".join(parts) + "│")
        if r < 3:
            print(mid)
    print(end)


# ---------- 不确定格子处理 ----------

def _handle_uncertain(screenshot, board_rect, board, confs, threshold):
    """找出低置信度格子，弹出窗口让用户修正

    Returns:
        corrections: [(row, col, old_value, new_value, cell_img), ...]
    """
    cells = split_board(crop_board(screenshot, board_rect))
    uncertain_mask = np.zeros((4, 4), dtype=bool)
    corrections = []

    for r in range(4):
        for c in range(4):
            if confs[r, c] < threshold and board[r, c] != 0:
                uncertain_mask[r, c] = True

    if not uncertain_mask.any():
        return [], uncertain_mask

    print(f"\n🔍 发现 {uncertain_mask.sum()} 个不确定格子")
    for r in range(4):
        for c in range(4):
            if not uncertain_mask[r, c]:
                continue

            cell = cells[r][c]
            h, w = cell.shape[:2]
            display = cv2.resize(cell, (w * 4, h * 4), interpolation=cv2.INTER_NEAREST)
            window_name = f"[{r+1},{c+1}] 猜测={board[r,c]} (置信度{confs[r,c]:.0%})"
            cv2.imshow(window_name, display)
            cv2.waitKey(1)

            label = input(
                f"  [{r+1},{c+1}] 猜测={board[r,c]} "
                f"→ 实际数字(回车=保留, 0=空格, q=跳过): "
            ).strip()
            cv2.destroyAllWindows()

            if label.lower() == "q":
                continue
            if label == "":
                continue  # 保留当前猜测
            if label == "0":
                label = "empty"

            try:
                new_val = 0 if label.lower() == "empty" else int(label)
            except ValueError:
                print(f"    无效输入 '{label}'，保留原值")
                continue

            corrections.append((r, c, board[r, c], new_val, cell))

    return corrections, uncertain_mask


def _save_and_retrain(corrections):
    """保存修正的图片到 ImageFolder 并重新训练"""
    from collect import save_cell_image
    import subprocess

    for r, c, old_val, new_val, cell_img in corrections:
        label = "empty" if new_val == 0 else str(new_val)
        path = save_cell_image(cell_img, label)
        print(f"  [{r+1},{c+1}] {old_val}→{new_val} 已保存: {path}")

    print("\n⏳ 正在重新训练模型...")
    ret = subprocess.run(
        [sys.executable, "train.py"],
        capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    # train.py 输出较多，只打印关键行
    for line in ret.stdout.split("\n"):
        line = line.strip()
        if "准确" in line or "完成" in line or "Saved" in line or "类别" in line:
            print(f"  {line}")
    if ret.returncode == 0:
        print("✅ 模型已更新")
        return True
    else:
        print(f"⚠ 训练失败:\n{ret.stderr}")
        return False


# ---------- 主循环 ----------

def main():
    print("=" * 50)
    print("  2048 自动助手")
    print("=" * 50)

    # 检查 ADB
    screenshot = adb_screenshot()
    if screenshot is None:
        print("❌ 无法获取手机截图，请检查 ADB 连接")
        return

    # 框选棋盘
    board_rect = load_board_rect()
    if board_rect is None:
        print("\n请用鼠标框选 2048 棋盘区域，Enter 确认...")
        board_rect = select_board_region(screenshot)
        if board_rect is None:
            print("未选择棋盘区域，退出")
            return
        save_board_rect(board_rect)

    # 从 config.yaml 加载配置
    config = _load_config()
    strategy = config.get("strategy", "standard")
    mode = config.get("mode", "semi")
    delay = config.get("swipe_delay", 0.6)
    post_delay = config.get("post_swipe_delay", 0.3)
    engine_model = config.get("engine_model", "snake")

    # 加载识别器
    try:
        recognizer = TileRecognizer()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return

    # 加载决策模型
    if engine_model == "extreme":
        from engine import load_extreme
        load_extreme()
    elif engine_model == "ntuple":
        load_ntuple()
    else:
        print("决策模型: 蛇形启发式（config.yaml 可改为 ntuple 或 extreme）")
    threshold_conf = config.get("confidence_threshold", 0.60)

    # 生成概率配置
    level = config.get("level", 1)
    if level == 1:
        spawns = [(1, 0.87), (2, 0.11), (3, 0.02)]
    elif level == 2:
        spawns = [(2, 0.75), (3, 0.20), (4, 0.05)]
    else:
        spawns = [(3, 0.65), (4, 0.25), (5, 0.10)]
    print(f"游戏等级: {level}  生成权重: {dict(spawns)}")

    # 自动统计实际生成概率
    track_spawn = config.get("track_spawn", False)
    spawn_counts = {}  # {value: count}
    prev_board = None
    last_direction = None

    strategies = get_available_strategies()
    direction_names = {"up": "↑ 上", "down": "↓ 下", "left": "← 左", "right": "→ 右"}

    print(f"类别: {recognizer.classes}")
    print(f"策略: {strategy} ({strategies[strategy]})")
    print()
    print("  [回车] 截屏识别 + 建议方向")
    print("  [c]    手动修正识别错误的格子")
    print("  [w/a/s/d] 手动方向")
    print("  [1/2/3] 切换策略(保守/标准/激进)")
    print("  [m] 切换半自动/全自动")
    print("  [r] 重新框选棋盘")
    print("  [q] 退出")
    print()

    while True:
        # 截屏
        screenshot = adb_screenshot()
        if screenshot is None:
            print("截图失败，重试...")
            time.sleep(0.5)
            continue

        # 识别
        board_img = crop_board(screenshot, board_rect)
        cells = split_board(board_img)
        board, confs = recognizer.predict_board(cells)

        # 处理不确定格子
        corrections, uncertain_mask = _handle_uncertain(
            screenshot, board_rect, board, confs, threshold_conf
        )

        if corrections:
            # 有修正 → 重新训练 → 重新加载识别器 → 重新识别
            _save_and_retrain(corrections)
            recognizer = TileRecognizer()  # 重新加载新模型
            board, confs = recognizer.predict_board(cells)
            # 再检查一遍是否还有不确定
            corrections2, uncertain_mask = _handle_uncertain(
                screenshot, board_rect, board, confs, threshold_conf
            )
            if corrections2:
                _save_and_retrain(corrections2)
                recognizer = TileRecognizer()
                board, confs = recognizer.predict_board(cells)

        # 显示
        os.system("cls" if os.name == "nt" else "clear")
        print("=" * 50)
        print(f"  2048 自动助手  |  策略: {strategy}  |  模式: {'全自动' if mode == 'auto' else '半自动'}")
        print("=" * 50)
        print()
        _print_board(board, confs, uncertain_mask)
        avg_conf = confs.mean()
        print(f"\n平均置信度: {avg_conf:.1%}")

        # 自动统计新棋子生成概率
        if track_spawn and prev_board is not None and last_direction is not None:
            from engine import move as engine_move
            expected, _, moved = engine_move(
                [list(row) for row in prev_board], last_direction
            )
            if moved:
                # 找预期为空但实际非空的格子（候选刷新点）
                candidates = []
                for r in range(4):
                    for c in range(4):
                        if expected[r][c] == 0 and board[r][c] != 0:
                            candidates.append((int(board[r][c]), r, c))
                if len(candidates) == 0:
                    pass  # 无滑动或滑动无变化
                elif len(candidates) == 1:
                    v = candidates[0][0]
                    spawn_counts[v] = spawn_counts.get(v, 0) + 1
                else:
                    print(f"  ⚠ 新棋子检测异常：预期 {len(candidates)} 个候选 "
                          f"{[c[0] for c in candidates]}，可能有识别误差，已跳过")
            last_direction = None  # 清空，避免同一次反复计

        # 记录本次棋盘，供下次对比
        prev_board = board.copy()

        if track_spawn and sum(spawn_counts.values()) >= 20:
            total = sum(spawn_counts.values())
            stats = " | ".join(
                f"{v}:{spawn_counts.get(v,0)/total*100:.0f}%"
                for v in sorted(spawn_counts.keys())
            )
            print(f"📊 新棋子统计({total}次): {stats}")

        # 决策
        best_dir = get_move(board, strategy, spawns)
        print(f"建议方向: {direction_names[best_dir]}")

        # 图像显示
        vis = screenshot.copy()
        x1, y1, x2, y2 = board_rect
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        w, h = x2 - x1, y2 - y1
        for i in range(1, 4):
            cv2.line(vis, (x1 + i * w // 4, y1), (x1 + i * w // 4, y2), (255, 255, 0), 1)
            cv2.line(vis, (x1, y1 + i * h // 4), (x2, y1 + i * h // 4), (255, 255, 0), 1)
        for r in range(4):
            for c in range(4):
                tx = x1 + c * w // 4 + 5
                ty = y1 + r * h // 4 + h // 8 + 5
                val = board[r, c]
                cf = confs[r, c]
                color = (0, 0, 255) if (uncertain_mask is not None and uncertain_mask[r, c]) else \
                        (0, 200, 0) if cf > 0.7 else (0, 180, 255)
                text = f"{val}({cf:.0%})" if val else "."
                cv2.putText(vis, text, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        vis_disp, _ = _fit_to_screen(vis)
        cv2.imshow("2048", vis_disp)

        # 模式分支
        print()
        if mode == "auto":
            print(f"⏳ 自动执行 {direction_names[best_dir]}（{delay}秒后）...")
            cv2.waitKey(1)
            time.sleep(delay)
            try:
                execute_swipe(best_dir, board_rect=board_rect)
                last_direction = best_dir
                print(f"✅ 已执行: {direction_names[best_dir]}")
            except Exception as e:
                print(f"❌ 执行失败: {e}")
            time.sleep(post_delay)
        else:
            print("> ", end="", flush=True)
            cmd = input().strip().lower()
            if cmd == "q":
                break
            elif cmd == "":
                try:
                    execute_swipe(best_dir, board_rect=board_rect)
                    last_direction = best_dir
                except Exception as e:
                    print(f"❌ 执行失败: {e}")
            elif cmd in ("w", "up"):
                execute_swipe("up", board_rect=board_rect)
                last_direction = "up"
            elif cmd in ("s", "down"):
                execute_swipe("down", board_rect=board_rect)
                last_direction = "down"
            elif cmd in ("a", "left"):
                execute_swipe("left", board_rect=board_rect)
                last_direction = "left"
            elif cmd in ("d", "right"):
                execute_swipe("right", board_rect=board_rect)
                last_direction = "right"
            elif cmd == "1":
                strategy = "conservative"
                config["strategy"] = strategy
                _save_config(config)
                print(f"→ 已切换并保存: {strategies[strategy]}")
            elif cmd == "2":
                strategy = "standard"
                config["strategy"] = strategy
                _save_config(config)
                print(f"→ 已切换并保存: {strategies[strategy]}")
            elif cmd == "3":
                strategy = "aggressive"
                config["strategy"] = strategy
                _save_config(config)
                print(f"→ 已切换并保存: {strategies[strategy]}")
            elif cmd == "m":
                mode = "auto" if mode == "semi" else "semi"
                config["mode"] = mode
                _save_config(config)
                print(f"→ 已切换并保存: {'全自动' if mode == 'auto' else '半自动'}")
            elif cmd == "c":
                # 手动修正：逐格确认，回车保留，输入新值覆盖
                print("  手动修正模式（回车=保留, 0=空格, q=跳过）")
                cells = split_board(board_img)
                manual_corrections = []
                for cr in range(4):
                    for cc in range(4):
                        val = board[cr, cc]
                        if val == 0:
                            continue
                        cell = cells[cr][cc]
                        h, w = cell.shape[:2]
                        display = cv2.resize(cell, (w * 4, h * 4), interpolation=cv2.INTER_NEAREST)
                        cv2.imshow(f"[{cr+1},{cc+1}] 当前={val}", display)
                        cv2.waitKey(1)
                        new = input(f"  [{cr+1},{cc+1}] 当前={val} → 新值: ").strip()
                        cv2.destroyAllWindows()
                        if new.lower() == "q":
                            continue
                        if new == "":
                            continue
                        if new == "0":
                            new = "empty"
                        try:
                            new_val = 0 if new.lower() == "empty" else int(new)
                        except ValueError:
                            continue
                        if new_val != val:
                            manual_corrections.append((cr, cc, val, new_val, cell))
                if manual_corrections:
                    _save_and_retrain(manual_corrections)
                    recognizer = TileRecognizer()
                    board, confs = recognizer.predict_board(cells)
                    print("✅ 已修正并重训，识别结果已更新")
                else:
                    print("  无修改")
            elif cmd == "r":
                board_rect = select_board_region(screenshot)
                if board_rect:
                    save_board_rect(board_rect)
            else:
                print(f"未知命令: {cmd}")
            time.sleep(post_delay)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n已退出。")
