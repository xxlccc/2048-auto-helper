package com.game2048.ai.service

import android.accessibilityservice.AccessibilityService
import android.accessibilityservice.GestureDescription
import android.graphics.Bitmap
import android.graphics.Path
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.accessibility.AccessibilityEvent
import com.game2048.ai.engine.Engine
import com.game2048.ai.pipeline.ImagePreprocessor
import com.game2048.ai.pipeline.TileRecognizer

class GameService : AccessibilityService() {
    companion object {
        var instance: GameService? = null
    }

    private val handler = Handler(Looper.getMainLooper())
    private var running = false
    private var runnable: Runnable? = null

    var strategy = "standard"
    var level = 1
    var postDelay = 660L
    var swipeDelay = 100L

    private val recognizer by lazy { TileRecognizer() }

    override fun onServiceConnected() {
        super.onServiceConnected()
        instance = this
        Log.d("GameService", "无障碍服务已连接")
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {}

    override fun onInterrupt() {
        stop()
    }

    override fun onDestroy() {
        super.onDestroy()
        instance = null
        stop()
    }

    fun start() {
        if (running) return
        running = true
        Log.d("GameService", "自动模式启动")
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
        Log.d("GameService", "自动模式停止")
    }

    fun isRunning() = running

    private fun step() {
        try {
            takeScreenshot(displayId, handler) { screenshot ->
                if (screenshot == null || !running) return@takeScreenshot

                val cells = ImagePreprocessor.splitCells(screenshot)
                val board = Array(4) { IntArray(4) }
                for (r in 0 until 4)
                    for (c in 0 until 4) {
                        val (value, _) = recognizer.predict(cells[r][c])
                        board[r][c] = value
                        cells[r][c].recycle()
                    }
                screenshot.recycle()

                val spawns = getSpawns()
                val dir = Engine.getMove(board, strategy, spawns)

                val (start, end) = ImagePreprocessor.swipeCoordinates(dir)
                gestureSwipe(start.first, start.second, end.first, end.second)
            }
        } catch (e: Exception) {
            Log.e("GameService", "step error: ${e.message}")
        }
    }

    private fun gestureSwipe(x1: Float, y1: Float, x2: Float, y2: Float) {
        val path = Path().apply {
            moveTo(x1, y1)
            lineTo(x2, y2)
        }
        val gesture = GestureDescription.Builder()
            .addStroke(GestureDescription.StrokeDescription(path, 0, swipeDelay))
            .build()
        dispatchGesture(gesture, null, null)
    }

    private fun getSpawns() = when (level) {
        1 -> listOf(Engine.Spawn(1, 0.75), Engine.Spawn(2, 0.20), Engine.Spawn(3, 0.05))
        2 -> listOf(Engine.Spawn(2, 0.75), Engine.Spawn(3, 0.20), Engine.Spawn(4, 0.05))
        else -> listOf(Engine.Spawn(3, 0.70), Engine.Spawn(4, 0.20), Engine.Spawn(5, 0.10))
    }
}
