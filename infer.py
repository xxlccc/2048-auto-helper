# infer.py
"""推理模块 - 加载训练好的模型，识别单格或整盘棋子"""
import os
import json
import cv2
import torch
import numpy as np
from torchvision import transforms
from model import TileCNN
from config import MODEL_PATH, TILE_INPUT_SIZE


class TileRecognizer:
    """棋子识别器"""

    def __init__(self, model_path=MODEL_PATH, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 加载类别映射（训练时保存的）
        mapping_path = model_path.replace(".pth", "_classes.json")
        if not os.path.exists(mapping_path):
            raise FileNotFoundError(
                f"类别映射不存在: {mapping_path}\n请先运行 python train.py"
            )
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        self.classes = mapping["classes"]  # ["empty", "2", "4", "8", ...]
        num_classes = len(self.classes)

        # 加载模型
        self.model = TileCNN(num_classes=num_classes).to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((TILE_INPUT_SIZE, TILE_INPUT_SIZE)),
            transforms.ToTensor(),
        ])

    def preprocess_cell(self, cell_img):
        """预处理单格图像"""
        if cell_img is None or cell_img.size == 0:
            return None

        # 统一转 RGB
        if len(cell_img.shape) == 2:
            cell_img = cv2.cvtColor(cell_img, cv2.COLOR_GRAY2BGR)

        cell_rgb = cv2.cvtColor(cell_img, cv2.COLOR_BGR2RGB)
        tensor = self.transform(cell_rgb).unsqueeze(0).to(self.device)
        return tensor

    @torch.no_grad()
    def predict(self, cell_img):
        """预测单格棋子数字

        Returns:
            (int_value, float_confidence)
            例: (2, 0.95) 表示数字 2，置信度 95%
                (0, 1.0)  表示空格子
        """
        x = self.preprocess_cell(cell_img)
        if x is None:
            return 0, 1.0

        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(torch.argmax(probs).item())
        conf = float(probs[pred_idx].item())

        # 索引 → 类别名 → 实际数字值
        class_name = self.classes[pred_idx]

        if class_name.lower() in ("empty", "0", "none", "bg"):
            return 0, conf
        else:
            try:
                value = int(class_name)
            except ValueError:
                value = 0
            return value, conf

    def predict_board(self, cells_4x4):
        """预测整盘 4×4 数字矩阵

        Args:
            cells_4x4: 4×4 的格子图像列表

        Returns:
            board: 4×4 整数矩阵（0=空格）
            confs: 4×4 置信度矩阵
        """
        board = np.zeros((4, 4), dtype=int)
        confs = np.zeros((4, 4), dtype=float)

        for r in range(4):
            for c in range(4):
                value, conf = self.predict(cells_4x4[r][c])
                board[r, c] = value
                confs[r, c] = conf

        return board, confs
