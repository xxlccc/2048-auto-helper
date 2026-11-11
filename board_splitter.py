# board_splitter.py
import cv2
import numpy as np
from config import BOARD_GRID_SIZE, CELL_MARGIN_RATIO

def split_board(board_img, grid_size=BOARD_GRID_SIZE, margin_ratio=CELL_MARGIN_RATIO):
    h, w = board_img.shape[:2]
    cell_h = h / grid_size
    cell_w = w / grid_size

    cells = []
    for r in range(grid_size):
        row = []
        for c in range(grid_size):
            x1 = int(c * cell_w)
            y1 = int(r * cell_h)
            x2 = int((c + 1) * cell_w)
            y2 = int((r + 1) * cell_h)

            cell = board_img[y1:y2, x1:x2]

            mh = int((y2 - y1) * margin_ratio)
            mw = int((x2 - x1) * margin_ratio)

            if cell.shape[0] > 2 * mh and cell.shape[1] > 2 * mw:
                cell = cell[mh:cell.shape[0]-mh, mw:cell.shape[1]-mw]

            row.append(cell)
        cells.append(row)

    return cells

def crop_board(image, board_rect):
    x1, y1, x2, y2 = board_rect
    return image[y1:y2, x1:x2]
