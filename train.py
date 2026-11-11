# train.py
"""用法: python train.py [数据目录] [可选的类别列表]

例: python train.py                          # 用 config 里的路径
    python train.py data/tiles               # 指定目录
    python train.py data/tiles 2,4,8,16,32   # 只训指定类别
"""
import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
from model import TileCNN
from config import TILES_DATA_DIR, MODEL_PATH, TILE_INPUT_SIZE


def main():
    # 确定数据目录
    data_dir = sys.argv[1] if len(sys.argv) > 1 else TILES_DATA_DIR
    if not os.path.isdir(data_dir):
        print(f"❌ 数据目录不存在: {data_dir}")
        print(f"   请创建目录结构，例如:")
        print(f"     {data_dir}/")
        print(f"       ├── empty/   (背景/空格子)")
        print(f"       ├── 2/       (数字 2)")
        print(f"       ├── 4/       (数字 4)")
        print(f"       └── ...")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")
    print(f"数据目录: {data_dir}")

    # 数据增强：不翻转数字（2 翻过来还是 2，但某些数字不对）
    transform = transforms.Compose([
        transforms.Resize((TILE_INPUT_SIZE, TILE_INPUT_SIZE)),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)

    if len(dataset) == 0:
        print("❌ 数据集为空，请放入图片后重试")
        return

    print(f"\n总样本数: {len(dataset)}")
    print(f"类别数: {len(dataset.classes)}")
    print(f"类别索引映射 (模型输出 → 实际值):")
    for i, cls_name in enumerate(dataset.classes):
        count = sum(1 for _, label in dataset.samples if label == i)
        print(f"  索引 {i} → '{cls_name}' ({count} 张)")

    # 检查每类至少 1 张
    class_counts = {}
    for _, label in dataset.samples:
        cls_name = dataset.classes[label]
        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
    min_count = min(class_counts.values())
    if min_count < 2:
        print(f"\n⚠ 有些类别只有 {min_count} 张图，训练可能不理想，但可以试试")

    # 分割训练/验证集
    train_size = max(1, int(0.85 * len(dataset)))
    val_size = max(1, len(dataset) - train_size)
    train_set, val_set = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    bsz = min(8, len(train_set))
    print(f"\nbatch_size: {bsz} (总样本少时自动缩小)")

    train_loader = DataLoader(train_set, batch_size=bsz, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=bsz, shuffle=False, num_workers=0)

    num_classes = len(dataset.classes)
    model = TileCNN(num_classes=num_classes).to(device)

    # 对少样本降低学习率
    lr = 1e-3 if len(dataset) >= 50 else (5e-4 if len(dataset) >= 20 else 1e-4)
    print(f"学习率: {lr}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    epochs = 20 if len(dataset) < 30 else 30
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)

        train_acc = correct / total if total else 0.0

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += imgs.size(0)

        val_acc = val_correct / val_total if val_total else 0.0

        print(f"Epoch {epoch+1:02d}/{epochs} | loss={total_loss/total:.4f} | "
              f"train_acc={train_acc:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # 保存模型 + 类别映射
            torch.save(model.state_dict(), MODEL_PATH)
            mapping = {"classes": dataset.classes, "class_to_idx": dataset.class_to_idx}
            mapping_path = MODEL_PATH.replace(".pth", "_classes.json")
            with open(mapping_path, "w", encoding="utf-8") as f:
                json.dump(mapping, f, ensure_ascii=False, indent=2)
            print(f"  → 已保存: {MODEL_PATH} + 类别映射")

    print(f"\n✅ 训练完成，最佳验证准确率: {best_val_acc:.1%}")
    print(f"   模型: {MODEL_PATH}")
    print(f"   映射: {mapping_path}")


if __name__ == "__main__":
    main()
