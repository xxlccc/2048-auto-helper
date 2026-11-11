# executor.py
"""ADB 滑动执行"""
import subprocess
import time


class ExecutorError(Exception):
    pass


def execute_swipe(direction, board_rect=None, duration_ms=100):
    """在棋盘区域内执行滑动

    Args:
        direction: "up" | "down" | "left" | "right"
        board_rect: (x1, y1, x2, y2) 棋盘包围盒（像素坐标），
                    传 None 则用屏幕比例滑动
        duration_ms: 滑动持续时间（毫秒）
    """
    if board_rect:
        x1, y1, x2, y2 = board_rect
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # 棋盘内留 15% 边距
        margin_x = int((x2 - x1) * 0.15)
        margin_y = int((y2 - y1) * 0.15)
        il, ir = x1 + margin_x, x2 - margin_x
        it, ib = y1 + margin_y, y2 - margin_y

        if direction == "up":
            fx, fy, tx, ty = cx, ib, cx, it
        elif direction == "down":
            fx, fy, tx, ty = cx, it, cx, ib
        elif direction == "left":
            fx, fy, tx, ty = ir, cy, il, cy
        elif direction == "right":
            fx, fy, tx, ty = il, cy, ir, cy
        else:
            raise ExecutorError(f"无效方向: {direction}")
    else:
        # 回退：屏幕比例滑动
        try:
            result = subprocess.run(
                ["adb", "shell", "wm", "size"],
                capture_output=True, text=True, timeout=5
            )
            line = result.stdout.strip()
            if ":" in line:
                size_str = line.split(":")[-1].strip()
                w, h = size_str.split("x")
                sw, sh = int(w), int(h)
            else:
                sw, sh = 1080, 1920
        except Exception:
            sw, sh = 1080, 1920

        ratios = {
            "up":    (0.5, 0.8, 0.5, 0.2),
            "down":  (0.5, 0.2, 0.5, 0.8),
            "left":  (0.8, 0.5, 0.2, 0.5),
            "right": (0.2, 0.5, 0.8, 0.5),
        }
        if direction not in ratios:
            raise ExecutorError(f"无效方向: {direction}")
        rx1, ry1, rx2, ry2 = ratios[direction]
        fx, fy = int(rx1 * sw), int(ry1 * sh)
        tx, ty = int(rx2 * sw), int(ry2 * sh)

    cmd = ["adb", "shell", "input", "swipe",
           str(fx), str(fy), str(tx), str(ty), str(duration_ms)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)

    if result.returncode != 0:
        raise ExecutorError(f"滑动失败: {result.stderr.strip()}")
