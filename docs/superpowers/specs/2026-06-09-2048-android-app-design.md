# 2048 安卓 App 设计方案

## 概述

将现有 PC 端 2048 自动助手打包为独立 Android APK，安装即用，无需电脑。

## 核心原则

- 完整复用已有 PC 端的 CNN 模型、棋盘坐标、蛇形决策引擎
- 手机端不训练，只推理
- 支持手动替换模型文件

## 架构

```
┌─────────────────────────────────────────┐
│                 Android APK              │
│                                          │
│  ┌──────────┐  ┌──────────────────────┐  │
│  │ 无障碍服务 │  │  PyTorch Mobile      │  │
│  │ (截图+滑动)│  │  (CNN 识图, .pt 模型) │  │
│  └────┬─────┘  └──────────┬───────────┘  │
│       │                   │               │
│       └───────┬───────────┘               │
│               ▼                           │
│  ┌────────────────────────┐              │
│  │  engine (完整移植)      │              │
│  │  蛇形评估 + Expectimax  │              │
│  │  + 置换表, 深度 3~11    │              │
│  └───────────┬────────────┘              │
│              ▼                            │
│  ┌────────────────────────┐              │
│  │  悬浮窗                 │              │
│  │  启动/暂停/状态指示      │              │
│  └────────────────────────┘              │
└─────────────────────────────────────────┘
```

## 模块详述

### 1. 无障碍服务

- 通过 Android AccessibilityService API 实现
- 截图：调用 `takeScreenshot()` 获取 Bitmap
- 滑动：构造 `GestureDescription` 模拟滑动
- 滑动坐标复用 `board_rect.json` 计算（棋盘中心区域）

### 2. CNN 识图

- 使用 PyTorch Mobile Android 库
- 模型格式：`model.pt`（从现有 `model.py` 导出的 TorchScript）
- 类别映射：`classes.json`（从现有 `tile_cnn_classes.json`）
- 加载优先级：`/sdcard/2048/model.pt` > APK 内置 assets
- 推理流程：截图 → 透视矫正 → 4×4 切格 → 逐格 CNN → 输出矩阵

### 3. 决策引擎

- 完整移植 `engine.py` 到 Kotlin：
  - `_slide_row_left` → 滑动合并逻辑
  - `_mirror` / `_transpose` → 棋盘变换
  - `move()` → 四个方向操作
  - 蛇形评估 → `_SNAKE_RAW` 矩阵权重（与 PC 版一致）
  - Expectimax + 置换表 → 递归搜索 + HashMap 缓存
  - 深度策略 → conservative 3~4 / standard 3~8 / aggressive 5~11
- 不加载 n-tuple 模型

### 4. 棋盘坐标

- 内置现有 `board_rect.json` 的内容（硬编码或 assets 文件）
- 截图分辨率与电脑 ADB 截图一致，坐标直接复用
- 无需用户标定

### 5. 配置

- `config.yaml` 参数硬编码初始值，在简易设置页可调：
  - 等级 level（1/2/3）
  - 策略 strategy（conservative/standard/aggressive）
  - 置信度阈值 confidence_threshold
  - 滑动前后延迟

### 6. 模型替换

- 内置模型：APK `assets/model.pt` + `assets/classes.json`
- 外部模型优先：`/sdcard/2048/model.pt` 存在时加载外部
- 替换方式：电脑训好新模型后，复制到手机该目录即可

### 7. 悬浮窗

- 半透明小圆点（约 40dp），可拖动，不遮挡游戏操作区
- 颜色指示：
  - 灰色 = 初始化/待机
  - 绿色 = 自动运行中
  - 红色 = 出错
- 交互：
  - 点击 = 启动/暂停自动模式
  - 长按 2 秒 = 打开简易设置面板（等级/策略/阈值/延迟）

## 数据流

```
无障碍截图(Bitmap) → 透视矫正(内置坐标) → 4×4 切格
    → PyTorch 推理 → 16 个 (值, 置信度)
    → 低置信度格 → 标记但不弹窗（手机端不修正）
    → engine.get_move() → 最优方向
    → 无障碍滑动(棋盘内坐标)
    → 延迟 post_swipe_delay
    → 循环
```

## 与 PC 版的差异

| 功能 | PC 版 | 手机版 |
|---|---|---|
| 手动修正错误格子 | ✅ 弹窗交互 | ❌ 不在手机上做 |
| 自动重训 CNN | ✅ subprocess 调用 | ❌ 不支持 |
| n-tuple 模型 | ✅ 可选加载 | ❌ 不包含 |
| 模型替换 | N/A | `/sdcard/2048/model.pt` |
| 新棋子统计 | ✅ 终端打印 | ❌ 不包含 |
| 可视化窗口 | ✅ OpenCV 窗口 | ✅ 无（仅悬浮窗） |

## 技术选型

- 语言：Kotlin
- 最低 SDK：Android 8.0 (API 26)
- 深度学习：PyTorch Mobile 2.x
- 辅助功能：AccessibilityService API

## 模型导出

PC 端需要将 `model.py` 的 TileCNN 导出为 TorchScript：

```python
import torch
from model import TileCNN
model = TileCNN(num_classes=N)
model.load_state_dict(torch.load("data/model/tile_cnn.pth"))
model.eval()
scripted = torch.jit.script(model)
scripted.save("model.pt")
```
