# export_model.py
"""将 TileCNN 导出为 TorchScript，供 Android PyTorch Mobile 加载"""
import os
import json
import torch
from model import TileCNN

MODEL_PATH = "data/model/tile_cnn.pth"
CLASSES_PATH = MODEL_PATH.replace(".pth", "_classes.json")
OUTPUT_DIR = "android_assets"


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 模型不存在: {MODEL_PATH}，请先运行 python train.py")
        return
    if not os.path.exists(CLASSES_PATH):
        print(f"❌ 类别映射不存在: {CLASSES_PATH}")
        return

    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    classes = mapping["classes"]
    num_classes = len(classes)

    model = TileCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()

    example = torch.randn(1, 3, 64, 64)
    scripted = torch.jit.trace(model, example)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_pt = os.path.join(OUTPUT_DIR, "model.pt")
    scripted.save(out_pt)

    out_json = os.path.join(OUTPUT_DIR, "classes.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(mapping, f)

    print(f"✅ 导出完成: {OUTPUT_DIR}/")
    print(f"   model.pt ({num_classes} 类)")
    print(f"   classes.json")
    print(f"   类别: {classes}")


if __name__ == "__main__":
    main()
