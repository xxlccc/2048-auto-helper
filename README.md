# 2048 自动助手（PC 端 Python + Android App）

通过 **ADB 截屏 → CNN 识别棋盘 → Expectimax 决策 → 自动滑动**，让程序替你玩手机上的 2048。
仓库包含一套完整的 PC 端 Python 工具链（识图、决策、训练、自动执行），以及一个可独立安装的 Android App。

```
┌──────────┐   ┌───────────┐   ┌──────────────┐   ┌──────────┐
│  截图     │──►│ CNN 识图   │──►│ Expectimax   │──►│ 滑动执行  │
│ ADB/无障碍│   │ 逐格数字   │   │ 蛇形/n-tuple │   │ ADB/手势  │
└──────────┘   └───────────┘   └──────────────┘   └──────────┘
```

## 功能

- **识图**：小型 CNN（PyTorch）识别 4×4 棋盘每格数字，置信度低于阈值时弹窗人工修正，并可用修正样本增量重训
- **决策**：Expectimax + 置换表 + 蛇形启发式评估；可选稀疏 n-tuple、稠密 n-tuple（性能最强）
- **执行**：PC 端走 ADB 滑动，可半自动逐步确认，也可全自动连续下棋
- **训练**：监督学习训练识图模型；TD(0) 自对弈训练 n-tuple 评估网络
- **Android**：PyTorch Mobile 端侧推理 + 无障碍服务截图/滑动 + 悬浮窗控制，手机独立运行，无需电脑

## 目录结构

```
2048/
├── main.py              # PC 端主程序（识别 → 决策 → 执行 主循环）
├── config.yaml          # 策略 / 模式 / 阈值 / 游戏等级等配置
├── config.py            # 路径与识别常量
├── adb_utils.py         # ADB 截屏
├── board_selector.py    # 鼠标框选棋盘区域
├── board_splitter.py    # 整盘切分为 4×4 单格
├── infer.py             # 模型加载与单格/整盘识别
├── model.py             # TileCNN 网络结构
├── engine.py            # Expectimax 决策引擎（snake / ntuple / extreme）
├── n_tuple.py           # 稀疏 n-tuple 网络
├── executor.py          # ADB 滑动执行
├── collect.py           # 训练数据采集（截图 → 框选 → 逐格标注）
├── train.py             # 识图模型监督训练
├── tdl_train.py         # TD(0) 训练稀疏 n-tuple
├── export_model.py      # 导出 TorchScript 供 Android 使用
├── 极限/                 # 稠密 n-tuple（更强，训练更慢）
│   ├── board_fast.py
│   ├── n_tuple_dense.py
│   └── train_extreme.py
├── data/
│   ├── model/           # 训练好的模型：tile_cnn.pth / n_tuple.pkl / 类别表 / 棋盘坐标缓存
│   └── tiles/           # 识图训练样本（ImageFolder 格式，按数字分类）
├── android/             # Android App（Kotlin，PyTorch Mobile）
├── android_assets/      # 打包进 APK 的模型与坐标
└── docs/                # 设计文档与实现计划
```

## 环境要求

- Python 3.10+（开发使用 3.11）
- 已开启 USB 调试并连接电脑的 Android 手机，已安装 2048 游戏
- `adb` 可执行文件在 PATH 中

```bash
pip install -r requirements.txt
```

## PC 端使用

```bash
adb devices          # 确认手机已连接
python main.py       # 启动
```

首次运行会在截图上用鼠标框选棋盘区域，坐标缓存在 `data/model/board_rect.json`，之后无需重复框选。

运行中快捷键：

| 按键 | 作用 |
| --- | --- |
| `回车` | 截屏识别 + 给出建议方向 |
| `c` | 手动修正识别错误的格子（可触发增量重训） |
| `w/a/s/d` | 手动指定方向 |
| `1/2/3` | 切换策略（保守 / 标准 / 激进） |
| `m` | 切换半自动 / 全自动 |
| `r` | 重新框选棋盘 |
| `q` | 退出 |

## 配置（`config.yaml`）

| 配置项 | 说明 |
| --- | --- |
| `strategy` | `conservative` / `standard` / `aggressive`，影响搜索深度与风险偏好 |
| `mode` | `semi` 每步确认 / `auto` 全自动 |
| `confidence_threshold` | 识别置信度阈值，低于此值提示人工修正 |
| `swipe_delay` / `post_swipe_delay` | 滑动前后等待时间（秒） |
| `max_auto_steps` | 全自动最大步数，0 为不限 |
| `level` | 游戏等级 1/2/3，决定新棋子的生成概率 |
| `engine_model` | `snake`（默认，免训练）/ `ntuple` / `extreme` |
| `track_spawn` | 运行时统计实际生成概率 |

## 训练

```bash
# 1) 采集识图样本：截图 → 框选棋盘 → 逐格标注，按数字存入 data/tiles/
python collect.py

# 2) 训练识图 CNN，输出 data/model/tile_cnn.pth
python train.py

# 3) TD(0) 自对弈训练稀疏 n-tuple，输出 data/model/n_tuple.pkl
python tdl_train.py --games 50000 --lr 0.01

# 4) 稠密 n-tuple（更强），输出 极限/model/extreme.pkl
python 极限/train_extreme.py

# 5) 导出 TorchScript 供 Android 端使用
python export_model.py
```

## Android App

`android/` 是一个独立的 Kotlin 工程（minSdk 26 / targetSdk 34，依赖 PyTorch Mobile 2.2.0）：

1. 用 Android Studio 打开 `android/` 目录
2. 首次构建会下载 Gradle 与依赖，直接 Run 安装到手机
3. 手机端开启无障碍服务、授予悬浮窗权限，打开悬浮窗即可开始

模型与棋盘坐标随 APK 打包在 `android/app/src/main/assets/` 下，可直接替换为重新训练的文件。

## 关于模型文件

- **仓库内包含**：`data/model/tile_cnn.pth`（识图，约 0.4 MB）、`data/model/n_tuple.pkl`（约 18 MB）、Android 用的 `model.pt`
- **未包含**：`极限/model/extreme.pkl`（**256 MB**，超过 GitHub 单文件 100 MB 硬上限）。需要时请自行运行 `python 极限/train_extreme.py` 生成，或用 Git LFS / Release 附件分发

## 参考

- [What is the optimal algorithm for the game 2048? — Stack Overflow](https://stackoverflow.com/questions/22342854/)（讨论存档见 `so_2048.md`）
- [nneonneo/2048-ai](https://github.com/nneonneo/2048-ai)
- Szubert & Jaśkowski, *Temporal difference learning of n-tuple networks for the game 2048*, CIG 2014

## 许可

尚未添加 LICENSE（待定）。
