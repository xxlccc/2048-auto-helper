# 2048 Android APK Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** 将现有 PC 端 2048 自动助手打包为独立 Android APK

**Architecture:** Kotlin 原生 Android 项目，无障碍服务截图+滑动，PyTorch Mobile 加载 TorchScript CNN 模型，纯 Kotlin 移植 Expectimax 决策引擎，悬浮窗交互

**Tech Stack:** Kotlin, Android API 26+, PyTorch Mobile 2.x, AccessibilityService

---

## 前置任务：模型导出

### Task 0: 导出 CNN 模型为 TorchScript

**Files:**
- Create: `export_model.py`

- [ ] **Step 1: 写导出脚本**

```python
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
    # 加载类别映射
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    classes = mapping["classes"]
    num_classes = len(classes)

    # 构建模型并加载权重
    model = TileCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()

    # 导出为 TorchScript
    example = torch.randn(1, 3, 64, 64)  # (batch, channels, h, w)
    scripted = torch.jit.trace(model, example)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    scripted.save(os.path.join(OUTPUT_DIR, "model.pt"))
    
    # 复制类别映射
    with open(os.path.join(OUTPUT_DIR, "classes.json"), "w") as f:
        json.dump(mapping, f)

    print(f"导出完成: {OUTPUT_DIR}/model.pt ({num_classes} 类)")
    print(f"类别: {classes}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行导出**

```bash
cd C:/Users/xxx/Desktop/2048
python export_model.py
```

预期输出：`android_assets/model.pt` 和 `android_assets/classes.json`

- [ ] **Step 3: 验证 TorchScript 模型可加载**

```bash
python -c "
import torch
m = torch.jit.load('android_assets/model.pt')
m.eval()
out = m(torch.randn(1, 3, 64, 64))
print('输出形状:', out.shape)
print('OK')
"
```

预期：`输出形状: torch.Size([1, N])`（N 为类别数）

---

## 第一阶段：Android 项目骨架

### Task 1: 创建 Android 项目

**Files:**
- Create: `android/` 目录下完整的 Android 项目结构

- [ ] **Step 1: 使用 Android Studio 或手动创建项目**

创建目录结构：
```
android/
  app/
    build.gradle.kts
    src/main/
      AndroidManifest.xml
      java/com/game2048/ai/
        MainActivity.kt
        App.kt
      assets/
        model.pt          (从 android_assets/ 复制)
        classes.json      (从 android_assets/ 复制)
        board_rect.json   (复制现有文件)
      res/
        values/strings.xml
        layout/activity_main.xml
  build.gradle.kts
  settings.gradle.kts
  gradle.properties
```

- [ ] **Step 2: 配置 `app/build.gradle.kts`**

```kotlin
plugins {
    id("com.android.application")
    kotlin("android")
}

android {
    namespace = "com.game2048.ai"
    compileSdk = 34
    defaultConfig {
        applicationId = "com.game2048.ai"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"
    }
}

dependencies {
    implementation("org.pytorch:pytorch_android_lite:2.2.0")
    implementation("org.pytorch:pytorch_android_torchvision_lite:2.2.0")
    implementation("androidx.core:core-ktx:1.12.0")
}
```

- [ ] **Step 3: 配置 `AndroidManifest.xml`**

```xml
<manifest xmlns:android="http://schemas.android.com/apk/res/android">
    <uses-permission android:name="android.permission.SYSTEM_ALERT_WINDOW" />
    
    <application>
        <activity android:name=".MainActivity"
            android:exported="true">
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />
                <category android:name="android.intent.category.LAUNCHER" />
            </intent-filter>
        </activity>
        
        <service android:name=".service.GameService"
            android:permission="android.permission.BIND_ACCESSIBILITY_SERVICE"
            android:exported="true">
            <intent-filter>
                <action android:name="android.accessibilityservice.AccessibilityService" />
            </intent-filter>
            <meta-data
                android:name="android.accessibilityservice"
                android:resource="@xml/accessibility_config" />
        </service>
    </application>
</manifest>
```

- [ ] **Step 4: 创建无障碍服务配置 `res/xml/accessibility_config.xml`**

```xml
<accessibility-service
    xmlns:android="http://schemas.android.com/apk/res/android"
    android:accessibilityFeedbackType="feedbackGeneric"
    android:canTakeScreenshot="true"
    android:canPerformGestures="true"
    android:notificationTimeout="100" />
```

- [ ] **Step 5: 创建 Application 类 `App.kt`**

```kotlin
class App : Application() {
    companion object {
        lateinit var instance: App
    }
    override fun onCreate() {
        super.onCreate()
        instance = this
    }
}
```

注册到 AndroidManifest.xml 的 `<application android:name=".App">`。

- [ ] **Step 6: Commit**

```bash
git add android/
git commit -m "feat: create Android project skeleton"
```

---

## 第二阶段：核心引擎（Kotlin 移植 engine.py）

### Task 2: 棋盘操作引擎 `Board.kt`

**Files:**
- Create: `android/app/src/main/java/com/game2048/ai/engine/Board.kt`

- [ ] **Step 1: 实现棋盘滑动逻辑**

```kotlin
package com.game2048.ai.engine

object Board {
    const val SIZE = 4
    val DIRECTIONS = listOf("up", "down", "left", "right")

    data class MoveResult(
        val board: Array<IntArray>,
        val score: Int,
        val moved: Boolean
    )

    fun move(board: Array<IntArray>, direction: String): MoveResult {
        return when (direction) {
            "left" -> moveLeft(board)
            "right" -> {
                val (r, s, _) = moveLeft(mirror(board))
                MoveResult(mirror(r), s, !boardsEqual(r, board))
            }
            "up" -> {
                val (r, s, _) = moveLeft(transpose(board))
                MoveResult(transpose(r), s, !boardsEqual(r, board))
            }
            else -> {
                val m = mirror(transpose(board))
                val (r, s, _) = moveLeft(m)
                MoveResult(transpose(mirror(r)), s, !boardsEqual(r, board))
            }
        }
    }

    fun mirror(board: Array<IntArray>): Array<IntArray> {
        return Array(SIZE) { r -> board[r].reversedArray() }
    }

    fun transpose(board: Array<IntArray>): Array<IntArray> {
        return Array(SIZE) { r -> Array(SIZE) { c -> board[c][r] }.toIntArray() }
    }

    private fun moveLeft(board: Array<IntArray>): MoveResult {
        val result = Array(SIZE) { IntArray(SIZE) }
        var totalScore = 0
        for (r in 0 until SIZE) {
            val (row, score) = slideRowLeft(board[r])
            result[r] = row
            totalScore += score
        }
        return MoveResult(result, totalScore, !boardsEqual(result, board))
    }

    private fun slideRowLeft(row: IntArray): Pair<IntArray, Int> {
        val nonZero = row.filter { it != 0 }.toMutableList()
        val merged = IntArray(SIZE)
        var score = 0
        var pos = 0
        var i = 0
        while (i < nonZero.size) {
            if (i + 1 < nonZero.size && nonZero[i] == nonZero[i + 1]) {
                merged[pos] = nonZero[i] + 1
                score += 1 shl (nonZero[i] + 1)
                i += 2
            } else {
                merged[pos] = nonZero[i]
                i += 1
            }
            pos++
        }
        return Pair(merged, score)
    }

    fun boardsEqual(a: Array<IntArray>, b: Array<IntArray>): Boolean {
        for (r in 0 until SIZE) {
            for (c in 0 until SIZE) {
                if (a[r][c] != b[r][c]) return false
            }
        }
        return true
    }
}
```

- [ ] **Step 2: 单元测试**

创建 `android/app/src/test/java/com/game2048/ai/engine/BoardTest.kt`：

```kotlin
class BoardTest {
    @Test fun testSlideLeft() {
        val board = arrayOf(
            intArrayOf(1, 1, 0, 0),
            intArrayOf(0, 0, 0, 0),
            intArrayOf(0, 0, 0, 0),
            intArrayOf(0, 0, 0, 0)
        )
        val result = Board.move(board, "left")
        assert(result.moved)
        assertEquals(2, result.board[0][0])  // 1+1→2
        assertEquals(0, result.board[0][1])
    }
}
```

- [ ] **Step 3: Commit**

```bash
git add android/app/src/main/java/com/game2048/ai/engine/Board.kt
git commit -m "feat: add board operations engine"
```

### Task 3: 蛇形评估 + Expectimax `Engine.kt`

**Files:**
- Create: `android/app/src/main/java/com/game2048/ai/engine/Engine.kt`

- [ ] **Step 1: 实现蛇形评估函数**

```kotlin
package com.game2048.ai.engine

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max

object Engine {
    // 蛇形权重矩阵 (与 PC 版 engine.py 完全一致)
    private val SNAKE_RAW = arrayOf(
        doubleArrayOf(10.0, 8.0, 7.0, 6.5),
        doubleArrayOf(0.5, 0.7, 1.0, 3.0),
        doubleArrayOf(-0.5, -1.5, -1.8, -2.0),
        doubleArrayOf(-3.8, -3.7, -3.5, -3.0)
    )
    private val SNAKE_BL = SNAKE_RAW.map { row -> row.map { exp(it) }.toDoubleArray() }.toTypedArray()
    private val SNAKE_BR = SNAKE_BL.map { it.reversedArray() }.toTypedArray()
    private val SNAKE_TL = SNAKE_BL.reversedArray()
    private val SNAKE_TR = SNAKE_TL.map { it.reversedArray() }.toTypedArray()
    private val ALL_SNAKES = arrayOf(SNAKE_BL, SNAKE_BR, SNAKE_TL, SNAKE_TR)

    fun evaluate(board: Array<IntArray>): Double {
        val empty = board.sumOf { row -> row.count { it == 0 } }

        var bestSnake = 0.0
        for (snake in ALL_SNAKES) {
            var s = 0.0
            for (r in 0 until 4) {
                for (c in 0 until 4) {
                    val v = board[r][c]
                    if (v > 0) s += (1 shl v) * snake[r][c]
                }
            }
            bestSnake = max(bestSnake, s)
        }

        var smooth = 0.0
        for (r in 0 until 4) {
            for (c in 0 until 4) {
                val v = board[r][c]
                if (v == 0) continue
                if (c + 1 < 4 && board[r][c + 1] != 0)
                    smooth -= abs(v - board[r][c + 1]).toDouble()
                if (r + 1 < 4 && board[r + 1][c] != 0)
                    smooth -= abs(v - board[r + 1][c]).toDouble()
            }
        }

        var merges = 0
        for (r in 0 until 4) {
            for (c in 0 until 3) {
                if (board[r][c] != 0 && board[r][c] == board[r][c + 1]) merges++
            }
        }
        for (r in 0 until 3) {
            for (c in 0 until 4) {
                if (board[r][c] != 0 && board[r][c] == board[r + 1][c]) merges++
            }
        }

        return bestSnake * 1.0 + empty * 5.0 + smooth * 0.5 + merges * 3.0
    }
}
```

- [ ] **Step 2: 实现置换表与 Expectimax 搜索**

在 `Engine.kt` 中继续添加：

```kotlin
    // 生成参数：[(值, 概率), ...]
    data class Spawn(val value: Int, val prob: Double)

    private val transTable = HashMap<String, Pair<Int, Double>>()

    private fun boardKey(board: Array<IntArray>): String {
        return board.joinToString("|") { row -> row.joinToString(",") }
    }

    fun getMove(
        board: Array<IntArray>,
        strategy: String = "standard",
        spawns: List<Spawn> = listOf(Spawn(1, 0.75), Spawn(2, 0.20), Spawn(3, 0.05))
    ): String {
        val empty = board.sumOf { r -> r.count { it == 0 } }
        transTable.clear()

        val depth = when (strategy) {
            "conservative" -> if (empty >= 6) 3 else 4
            "aggressive" -> when {
                empty >= 8 -> 5
                empty >= 5 -> 7
                empty >= 3 -> 9
                else -> 11
            }
            else -> when {
                empty >= 8 -> 3
                empty >= 5 -> 4
                empty >= 3 -> 6
                else -> 8
            }
        }

        val (_, bestDir) = expectimax(board, depth, true, spawns)
        return bestDir ?: "up"
    }

    private fun expectimax(
        board: Array<IntArray>,
        depth: Int,
        isPlayer: Boolean,
        spawns: List<Spawn>
    ): Pair<Double, String?> {
        val key = boardKey(board)
        transTable[key]?.let { (d, s) ->
            if (d >= depth) return Pair(s, null)
        }

        if (depth == 0) {
            val ev = evaluate(board)
            transTable[key] = Pair(0, ev)
            return Pair(ev, null)
        }

        if (isPlayer) {
            var bestScore = Double.NEGATIVE_INFINITY
            var bestDir: String? = null
            val ordered = orderMoves(board)
            for (d in ordered) {
                val (nb, gain, moved) = Board.move(board, d)
                if (!moved) continue
                val (cs, _) = expectimax(nb, depth - 1, false, spawns)
                val total = cs + gain
                if (total > bestScore) { bestScore = total; bestDir = d }
            }
            if (bestDir == null) {
                val ev = evaluate(board)
                transTable[key] = Pair(0, ev)
                return Pair(ev, null)
            }
            transTable[key] = Pair(depth, bestScore)
            return Pair(bestScore, bestDir)
        } else {
            val empty = mutableListOf<Pair<Int, Int>>()
            for (r in 0 until 4) for (c in 0 until 4) if (board[r][c] == 0) empty.add(r to c)
            if (empty.isEmpty()) {
                val ev = evaluate(board)
                transTable[key] = Pair(0, ev)
                return Pair(ev, null)
            }

            val sampled = if (empty.size > 8) empty.shuffled().take(8) else empty
            var total = 0.0
            for ((r, c) in sampled) {
                for (sp in spawns) {
                    val nb = board.map { it.clone() }.toTypedArray()
                    nb[r][c] = sp.value
                    val (s, _) = expectimax(nb, depth - 1, true, spawns)
                    total += sp.prob * s
                }
            }
            return Pair(total / sampled.size, null)
        }
    }

    private fun orderMoves(board: Array<IntArray>): List<String> {
        return Board.DIRECTIONS.map { d ->
            val (nb, gain, moved) = Board.move(board, d)
            if (moved) Pair(evaluate(nb) + gain, d) else Pair(Double.NEGATIVE_INFINITY, d)
        }.sortedByDescending { it.first }.map { it.second }
    }
}
```

- [ ] **Step 3: Commit**

```bash
git add android/app/src/main/java/com/game2048/ai/engine/Engine.kt
git commit -m "feat: add snake evaluation + expectimax engine"
```

---

## 第三阶段：AI 管道

### Task 4: 图像预处理 `ImagePreprocessor.kt`

**Files:**
- Create: `android/app/src/main/java/com/game2048/ai/pipeline/ImagePreprocessor.kt`
- Reference: `board_rect.json`, `board_splitter.py`, `infer.py`

- [ ] **Step 1: 实现透视矫正 + 4×4 切格**

```kotlin
package com.game2048.ai.pipeline

import android.graphics.Bitmap
import android.graphics.Rect

object ImagePreprocessor {
    // 棋盘坐标 (来自 board_rect.json)
    private val BOARD_RECT = Rect(114, 785, 1143, 1848)

    /** 从截图中裁出棋盘，切分为 4×4 格子 */
    fun splitCells(screenshot: Bitmap): Array<Array<Bitmap>> {
        // 裁出棋盘区域
        val board = Bitmap.createBitmap(
            screenshot,
            BOARD_RECT.left, BOARD_RECT.top,
            BOARD_RECT.width(), BOARD_RECT.height()
        )

        val cellW = board.width / 4
        val cellH = board.height / 4
        val margin = 0.05  // CELL_MARGIN_RATIO

        return Array(4) { r ->
            Array(4) { c ->
                val x = (c * cellW + cellW * margin).toInt()
                val y = (r * cellH + cellH * margin).toInt()
                val w = (cellW * (1 - 2 * margin)).toInt()
                val h = (cellH * (1 - 2 * margin)).toInt()
                Bitmap.createBitmap(board, x, y, w, h)
            }
        }
    }

    /** 根据坐标计算棋盘内的滑动方向起止点 */
    fun swipeCoordinates(direction: String): Pair<Pair<Float, Float>, Pair<Float, Float>> {
        val cx = BOARD_RECT.centerX().toFloat()
        val cy = BOARD_RECT.centerY().toFloat()
        val marginX = BOARD_RECT.width() * 0.15f
        val marginY = BOARD_RECT.height() * 0.15f
        val il = BOARD_RECT.left + marginX
        val ir = BOARD_RECT.right - marginX
        val it = BOARD_RECT.top + marginY
        val ib = BOARD_RECT.bottom - marginY

        return when (direction) {
            "up"    -> (cx to ib) to (cx to it)
            "down"  -> (cx to it) to (cx to ib)
            "left"  -> (ir to cy) to (il to cy)
            "right" -> (il to cy) to (ir to cy)
            else    -> (cx to ib) to (cx to it)
        }
    }
}
```

- [ ] **Step 2: Commit**

### Task 5: CNN 推理 `TileRecognizer.kt`

**Files:**
- Create: `android/app/src/main/java/com/game2048/ai/pipeline/TileRecognizer.kt`

- [ ] **Step 1: 实现 PyTorch Mobile 推理封装**

```kotlin
package com.game2048.ai.pipeline

import android.graphics.Bitmap
import android.os.Environment
import com.game2048.ai.App
import org.json.JSONObject
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.torchvision.TensorImageUtils
import java.io.File

class TileRecognizer {
    private val module: Module
    private val classes: List<String>

    init {
        // 优先加载外部模型，否则用内置
        val externalModel = File(Environment.getExternalStorageDirectory(), "2048/model.pt")
        val externalClasses = File(Environment.getExternalStorageDirectory(), "2048/classes.json")

        module = if (externalModel.exists()) {
            Module.load(externalModel.absolutePath)
        } else {
            Module.load(assetFilePath("model.pt"))
        }

        val jsonStr = if (externalClasses.exists()) {
            externalClasses.readText()
        } else {
            App.instance.assets.open("classes.json").bufferedReader().readText()
        }
        classes = JSONObject(jsonStr).getJSONArray("classes")
            .let { arr -> (0 until arr.length()).map { arr.getString(it) } }
    }

    fun predict(cell: Bitmap): Pair<Int, Float> {
        // 中心裁方 + 缩放到 64×64
        val size = minOf(cell.width, cell.height)
        val x = (cell.width - size) / 2
        val y = (cell.height - size) / 2
        val cropped = Bitmap.createBitmap(cell, x, y, size, size)
        val resized = Bitmap.createScaledBitmap(cropped, 64, 64, true)

        // 转 Tensor
        val tensor = TensorImageUtils.bitmapToFloat32Tensor(
            resized,
            floatArrayOf(0f, 0f, 0f),   // mean
            floatArrayOf(1f, 1f, 1f)    // std
        )

        val output = module.forward(IValue.from(tensor)).toTensor()
        val scores = output.dataAsFloatArray
        val bestIdx = scores.indices.maxByOrNull { scores[it] } ?: 0
        val conf = scores[bestIdx]

        val className = classes[bestIdx]
        val value = if (className == "empty") 0 else className.toIntOrNull() ?: 0
        return Pair(value, conf)
    }

    private fun assetFilePath(name: String): String {
        val f = File(App.instance.filesDir, name)
        App.instance.assets.open(name).use { src ->
            f.outputStream().use { dst -> src.copyTo(dst) }
        }
        return f.absolutePath
    }
}
```

- [ ] **Step 2: Commit**

---

## 第四阶段：无障碍服务

### Task 6: 无障碍服务 `GameService.kt`

**Files:**
- Create: `android/app/src/main/java/com/game2048/ai/service/GameService.kt`

- [ ] **Step 1: 实现截图 + 滑动 + 主循环**

```kotlin
package com.game2048.ai.service

import android.accessibilityservice.AccessibilityService
import android.accessibilityservice.GestureDescription
import android.graphics.Bitmap
import android.graphics.Path
import android.os.Handler
import android.os.Looper
import android.view.accessibility.AccessibilityEvent
import com.game2048.ai.engine.Board
import com.game2048.ai.engine.Engine
import com.game2048.ai.pipeline.ImagePreprocessor
import com.game2048.ai.pipeline.TileRecognizer

class GameService : AccessibilityService() {
    private val handler = Handler(Looper.getMainLooper())
    private val recognizer = TileRecognizer()
    private var running = false
    private var runnable: Runnable? = null

    // 配置（可从设置页修改）
    var strategy = "standard"
    var level = 1
    var postDelay = 660L  // ms

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {}

    override fun onInterrupt() { stop() }

    override fun onServiceConnected() {
        super.onServiceConnected()
        // 通知悬浮窗服务已就绪
    }

    fun start() {
        if (running) return
        running = true
        runnable = object : Runnable {
            override fun run() {
                if (!running) return
                step()
                handler.postDelayed(this, postDelay)
            }
        }
        handler.post(runnable!!)
    }

    fun stop() {
        running = false
        runnable = null
    }

    private fun step() {
        takeScreenshot(displayId, handler) { screenshot ->
            if (screenshot == null) return@takeScreenshot

            // 切分 → 识别
            val cells = ImagePreprocessor.splitCells(screenshot)
            val board = Array(4) { IntArray(4) }
            for (r in 0 until 4) {
                for (c in 0 until 4) {
                    val (value, _) = recognizer.predict(cells[r][c])
                    board[r][c] = value
                }
            }
            screenshot.recycle()

            // 决策
            val spawns = getSpawnsForLevel(level)
            val dir = Engine.getMove(board, strategy, spawns)

            // 滑动
            val (start, end) = ImagePreprocessor.swipeCoordinates(dir)
            gestureSwipe(start.first, start.second, end.first, end.second)
        }
    }

    private fun gestureSwipe(x1: Float, y1: Float, x2: Float, y2: Float) {
        val path = Path().apply {
            moveTo(x1, y1)
            lineTo(x2, y2)
        }
        val gesture = GestureDescription.Builder()
            .addStroke(GestureDescription.StrokeDescription(path, 0, 100))
            .build()
        dispatchGesture(gesture, null, null)
    }

    private fun getSpawnsForLevel(level: Int) = when (level) {
        1 -> listOf(Engine.Spawn(1, 0.75), Engine.Spawn(2, 0.20), Engine.Spawn(3, 0.05))
        2 -> listOf(Engine.Spawn(2, 0.75), Engine.Spawn(3, 0.20), Engine.Spawn(4, 0.05))
        else -> listOf(Engine.Spawn(3, 0.70), Engine.Spawn(4, 0.20), Engine.Spawn(5, 0.10))
    }
}
```

- [ ] **Step 2: Commit**

---

## 第五阶段：悬浮窗 UI

### Task 7: 悬浮窗 `FloatingButton.kt`

**Files:**
- Create: `android/app/src/main/java/com/game2048/ai/ui/FloatingButton.kt`

- [ ] **Step 1: 实现半透明圆点悬浮窗**

```kotlin
package com.game2048.ai.ui

import android.content.Context
import android.graphics.PixelFormat
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.WindowManager

class FloatingButton(
    private val ctx: Context,
    private val onToggle: (Boolean) -> Unit,
    private val onLongPress: () -> Unit
) {
    private val wm = ctx.getSystemService(Context.WINDOW_SERVICE) as WindowManager
    private val view = View(ctx).apply {
        setBackgroundResource(R.drawable.circle_gray)
    }
    private var isActive = false

    private val params = WindowManager.LayoutParams(
        80, 80,
        WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
        WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE,
        PixelFormat.TRANSLUCENT
    ).apply { gravity = Gravity.TOP or Gravity.START; x = 100; y = 300 }

    init {
        var startX = 0f; var startY = 0f
        var clickTime = 0L

        view.setOnTouchListener { _, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    startX = event.rawX; startY = event.rawY
                    clickTime = System.currentTimeMillis()
                    true
                }
                MotionEvent.ACTION_UP -> {
                    val dx = event.rawX - startX; val dy = event.rawY - startY
                    if (abs(dx) < 10 && abs(dy) < 10) {
                        if (System.currentTimeMillis() - clickTime > 2000) {
                            onLongPress()  // 长按 2 秒
                        } else {
                            isActive = !isActive
                            updateColor()
                            onToggle(isActive)  // 短按切换
                        }
                    }
                    true
                }
                MotionEvent.ACTION_MOVE -> {
                    params.x = event.rawX.toInt() - 40
                    params.y = event.rawY.toInt() - 40
                    wm.updateViewLayout(view, params)
                    true
                }
                else -> false
            }
        }
    }

    fun show() { wm.addView(view, params) }
    fun hide() { wm.removeView(view) }

    private fun updateColor() {
        val res = if (isActive) R.drawable.circle_green else R.drawable.circle_gray
        view.setBackgroundResource(res)
    }
}
```

需要同时创建 3 个 drawable 资源：
- `res/drawable/circle_gray.xml` — 灰色圆（`<shape android:shape="oval"><solid android:color="#80808080"/></shape>`）
- `res/drawable/circle_green.xml` — 绿色圆
- `res/drawable/circle_red.xml` — 红色圆

- [ ] **Step 2: Commit**

### Task 8: 设置面板 + MainActivity

**Files:**
- Create: `android/app/src/main/java/com/game2048/ai/ui/SettingsDialog.kt`
- Modify: `android/app/src/main/java/com/game2048/ai/MainActivity.kt`

- [ ] **Step 1: MainActivity 串联所有组件**

```kotlin
package com.game2048.ai

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.provider.Settings
import androidx.appcompat.app.AppCompatActivity
import com.game2048.ai.service.GameService
import com.game2048.ai.ui.FloatingButton

class MainActivity : AppCompatActivity() {
    private lateinit var floatingButton: FloatingButton
    private var service: GameService? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // 检查悬浮窗权限
        if (!Settings.canDrawOverlays(this)) {
            startActivity(Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                Uri.parse("package:$packageName")))
        }

        // 引导开启无障碍服务
        startActivity(Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS))

        floatingButton = FloatingButton(this,
            onToggle = { active ->
                if (active) service?.start() else service?.stop()
            },
            onLongPress = {
                SettingsDialog(this, service).show()
            }
        )
        floatingButton.show()
    }

    override fun onDestroy() {
        super.onDestroy()
        floatingButton.hide()
    }
}
```

- [ ] **Step 2: 设置对话框**

```kotlin
class SettingsDialog(ctx: Context, private val service: GameService?) : AlertDialog.Builder(ctx) {
    init {
        setTitle("设置")
        val items = arrayOf("等级 1/2/3", "策略 conservative/standard/aggressive", "滑动延迟")
        setItems(items) { _, which ->
            when (which) {
                0 -> showLevelPicker()
                1 -> showStrategyPicker()
                2 -> showDelayPicker()
            }
        }
    }
    // ... picker 实现
}
```

- [ ] **Step 3: Commit**

---

## 第六阶段：集成测试

### Task 9: 端到端测试

- [ ] **Step 1: 安装 APK 到手机**

```bash
./gradlew assembleDebug
adb install app/build/outputs/apk/debug/app-debug.apk
```

- [ ] **Step 2: 开启无障碍服务**

打开 2048 游戏 → 下拉通知栏 → 点击无障碍服务 → 开启 "2048 AI"

- [ ] **Step 3: 验证完整流程**

1. 圆点出现（灰色）→ 点击 → 变绿
2. 打开 2048 游戏 → 观察是否自动滑动
3. 长按圆点 → 设置面板弹出 → 切换等级/策略
4. 验证 `/sdcard/2048/model.pt` 模型替换生效

- [ ] **Step 4: Commit final**

```bash
git add -A
git commit -m "feat: complete Android APK with floating UI + accessibility service"
```

---

## 附件：drawable 资源

### `res/drawable/circle_gray.xml`
```xml
<shape xmlns:android="http://schemas.android.com/apk/res/android"
    android:shape="oval">
    <solid android:color="#80808080"/>
    <size android:width="40dp" android:height="40dp"/>
</shape>
```

### `res/drawable/circle_green.xml`
```xml
<shape xmlns:android="http://schemas.android.com/apk/res/android"
    android:shape="oval">
    <solid android:color="#8044CC44"/>
    <size android:width="40dp" android:height="40dp"/>
</shape>
```
