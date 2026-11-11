# collect.py
"""训练数据采集工具 —— 截图 → 框选棋盘 → 逐格赋值 → 存入 ImageFolder"""
import os
import sys
import json
import cv2
from adb_utils import adb_screenshot
from board_selector import select_board_region
from board_splitter import crop_board, split_board
from config import TILES_DATA_DIR

BOARD_CACHE = "data/model/board_rect.json"


def save_board_rect(rect):
    os.makedirs(os.path.dirname(BOARD_CACHE), exist_ok=True)
    with open(BOARD_CACHE, "w") as f:
        json.dump(rect, f)


def load_board_rect():
    if os.path.exists(BOARD_CACHE):
        with open(BOARD_CACHE, "r") as f:
            return json.load(f)
    return None


def save_cell_image(cell_img, label):
    """保存格子图片到 ImageFolder 目录，自动编号不覆盖"""
    label_dir = os.path.join(TILES_DATA_DIR, str(label))
    os.makedirs(label_dir, exist_ok=True)

    # 自动编号
    idx = 1
    while True:
        if idx == 1:
            path = os.path.join(label_dir, "1.png")
        else:
            path = os.path.join(label_dir, f"{idx}.png")
        if not os.path.exists(path):
            break
        idx += 1

    cv2.imwrite(path, cell_img)
    return path


def main():
    print("=" * 50)
    print("  训练数据采集工具")
    print("  逐个查看格子 → 输入数字 → 自动归档")
    print("=" * 50)
    print()

    # 1. 截图
    screenshot = adb_screenshot()
    if screenshot is None:
        print("❌ 无法获取手机截图，请检查 ADB 连接")
        return

    # 2. 框选棋盘
    board_rect = load_board_rect()
    if board_rect is None:
        print("请用鼠标框选 2048 棋盘区域，按 Enter 确认...")
        board_rect = select_board_region(screenshot)
        if board_rect is None:
            print("未选择棋盘区域，退出")
            return
        save_board_rect(board_rect)
    else:
        print(f"使用缓存区域: {board_rect}（按 r 重新框选）")

    # 3. 切分格子
    board_img = crop_board(screenshot, board_rect)
    cells = split_board(board_img)

    # 4. 逐格赋值
    print("\n" + "=" * 50)
    print("  逐格赋值（空格子输入 0 或直接回车）")
    print("  输入 q 跳过本格  |  quit 退出")
    print("=" * 50)

    total_saved = 0

    for row in range(4):
        for col in range(4):
            cell = cells[row][col]

            # 放大显示格子图像
            h, w = cell.shape[:2]
            display = cv2.resize(cell, (w * 4, h * 4), interpolation=cv2.INTER_NEAREST)
            cv2.imshow(f"第{row+1}行 第{col+1}列", display)
            cv2.waitKey(1)

            label = input(f"  [{row+1},{col+1}] 数字(0=空格): ").strip()
            cv2.destroyAllWindows()

            if label.lower() == "quit":
                print(f"\n已退出。本次保存 {total_saved} 张图片")
                print(f"数据目录: {TILES_DATA_DIR}")
                return
            if label.lower() == "q":
                continue
            if label == "":
                label = "empty"
            if label == "0":
                label = "empty"

            path = save_cell_image(cell, label)
            total_saved += 1
            print(f"    → 已保存: {path}")

    print(f"\n✅ 采集完成！本次保存 {total_saved} 张")
    print(f"   数据目录: {TILES_DATA_DIR}")
    print(f"   下一步: python train.py")


if __name__ == "__main__":
    main()
