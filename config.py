# config.py

BOARD_GRID_SIZE = 4
TILE_INPUT_SIZE = 64

# 数据路径
TILES_DATA_DIR = "data/tiles"
MODEL_PATH = "data/model/tile_cnn.pth"

# 识别时对单格四周裁掉的比例，避免边框干扰
CELL_MARGIN_RATIO = 0.05
