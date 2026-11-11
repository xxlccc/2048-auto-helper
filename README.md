# 2048 自动助手（PC 端 Python + Android App）

通过 **ADB 截屏 → CNN 识别棋盘 → Expectimax 决策 → 自动滑动**，让程序替你玩手机上的 2048。
当前可用的是 PC 端 Python 工具链（识图、决策、训练、自动执行）。Android App 仍在开发中、暂不可用，详见「已知问题」。

```
┌──────────┐   ┌───────────┐   ┌──────────────┐   ┌──────────┐
│  截图     │──►│ CNN 识图   │──►│ Expectimax   │──►│ 滑动执行  │
│ ADB/无障碍│   │ 逐格数字   │   │ 蛇形/n-tuple │   │ ADB/手势  │
└──────────┘   └───────────┘   └──────────────┘   └──────────┘
```

## 功能

- **识图**：小型 CNN（PyTorch）识别 4×4 棋盘每格数字，置信度低于阈值时弹窗人工修正，并可用修正样本增量重训
- **决策**：Expectimax + 置换表 + 蛇形启发式评估；可选稀疏 n-tuple；稠密 n-tuple（`极限/`）仍在调试，见「已知问题」
- **执行**：PC 端走 ADB 滑动，可半自动逐步确认，也可全自动连续下棋
- **训练**：监督学习训练识图模型；TD(0) 自对弈训练 n-tuple 评估网络
- **Android**（开发中，暂不可用）：PyTorch Mobile 端侧推理 + 无障碍服务截图/滑动 + 悬浮窗控制，目标是手机独立运行、无需电脑

## 已知问题

> ⚠️ 目前**只有 PC 端（Python + ADB）可用**，下面两块都还不能用。

### 极限分支（`极限/`，dense n-tuple）—— 仍有问题，暂不可用

请保持 `config.yaml` 里的 `engine_model: snake`（默认值）或改用 `ntuple`；下文「训练」第 4 步仅供参考，暂不建议执行。

- **模型文件缺失时不会报错，而是静默降级**：`极限/model/extreme.pkl` 有 256 MB，超过 GitHub 单文件上限，没有随仓库分发。把 `engine_model` 设成 `extreme` 后，程序只打印一行中文提示就继续跑，实际仍在用蛇形评估——看起来"能用"，其实不是极限模型在决策。
- **依赖 numba，环境容易装不上**：`极限/board_fast.py` 需要 numba 做 JIT 加速。numba 目前不支持 Python 3.14，此时训练会退回纯 Python 实现，速度远达不到设计中的"快 5~10 倍"。
- **内存占用大**：模型常驻约 256 MB（4 张 16.7M 的 float32 表），再加 Python 本身的开销，低配机器容易吃掉大半内存。
- **端到端链路未验证通过**：从「训练 → 保存 → 加载 → 实战」整条路径尚未跑通并确认效果，暂不保证可用。

### Android App（`android/`）—— 尚未完成，不可用

**Android 端还没有做完，处于不可用状态。** 仓库里只有源码，没有随附 APK，也没有 Gradle Wrapper，功能未打通；请暂时只在 PC 端运行。

- **没有可安装的产物**：仓库里不含 `.apk`，克隆下来没法直接装到手机上试。
- **缺少 Gradle Wrapper**：`android/` 下只有 `build.gradle.kts`、`settings.gradle.kts`、`gradle.properties`，没有 `gradlew`，命令行无法直接构建，需要先用 Android Studio 补全 wrapper。
- **功能未验证**：无障碍服务截图、手势滑动、PyTorch Mobile 推理这条链路尚未在真机上跑通确认。

修复进展会更新在本节；遇到具体报错欢迎提 issue。

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
├── 极限/                 # 稠密 n-tuple（实验中，暂不可用，见「已知问题」）
│   ├── board_fast.py
│   ├── n_tuple_dense.py
│   └── train_extreme.py
├── data/
│   ├── model/           # 训练好的模型：tile_cnn.pth / n_tuple.pkl / 类别表 / 棋盘坐标缓存
│   └── tiles/           # 识图训练样本（ImageFolder 格式，按数字分类）
├── android/             # Android App（Kotlin，开发中，暂不可用）
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
| `engine_model` | `snake`（默认，免训练）/ `ntuple`（推荐）/ `extreme`（**暂不可用**） |
| `track_spawn` | 运行时统计实际生成概率 |

## 训练

```bash
# 1) 采集识图样本：截图 → 框选棋盘 → 逐格标注，按数字存入 data/tiles/
python collect.py

# 2) 训练识图 CNN，输出 data/model/tile_cnn.pth
python train.py

# 3) TD(0) 自对弈训练稀疏 n-tuple，输出 data/model/n_tuple.pkl
python tdl_train.py --games 50000 --lr 0.01

# 4) 稠密 n-tuple —— ⚠ 暂时有问题，勿用于实战，见「已知问题」
#    python 极限/train_extreme.py

# 5) 导出 TorchScript 供 Android 端使用
python export_model.py
```

## Android App（开发中，暂不可用）

> 🚧 **尚未完成，当前不可用**，请先使用 PC 端。以下是开发中的工程说明，供后续继续开发参考。

`android/` 是一个独立的 Kotlin 工程（minSdk 26 / targetSdk 34，依赖 PyTorch Mobile 2.2.0），目标是让手机独立运行、不再需要电脑：

- 用 Android Studio 打开 `android/` 目录（工程未附带 Gradle Wrapper，需由 IDE 生成后才能构建）
- 端侧用 PyTorch Mobile 2.2.0 做 CNN 推理，通过无障碍服务截屏、手势滑动
- 模型与棋盘坐标放在 `android/app/src/main/assets/` 下，可直接替换为重新训练的文件

上述链路尚未在真机上验证通过，暂不保证可用。

## 关于模型文件

- **仓库内包含**：`data/model/tile_cnn.pth`（识图，约 0.4 MB）、`data/model/n_tuple.pkl`（约 18 MB）、Android 用的 `model.pt`
- **未包含**：`极限/model/extreme.pkl`（**256 MB**，超过 GitHub 单文件 100 MB 硬上限）。需要时请自行运行 `python 极限/train_extreme.py` 生成，或用 Git LFS / Release 附件分发

## 参考

- [What is the optimal algorithm for the game 2048? — Stack Overflow](https://stackoverflow.com/questions/22342854/)（讨论存档见 `so_2048.md`）
- [nneonneo/2048-ai](https://github.com/nneonneo/2048-ai)
- Szubert & Jaśkowski, *Temporal difference learning of n-tuple networks for the game 2048*, CIG 2014

## 许可

尚未添加 LICENSE（待定）。
