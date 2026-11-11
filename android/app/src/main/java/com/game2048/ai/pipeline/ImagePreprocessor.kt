package com.game2048.ai.pipeline

import android.graphics.Bitmap
import android.graphics.Rect

object ImagePreprocessor {
    // 棋盘坐标 (与 PC 端 board_rect.json 一致: [114, 785, 1143, 1848])
    private val BOARD_RECT = Rect(114, 785, 1143, 1848)

    fun splitCells(screenshot: Bitmap): Array<Array<Bitmap>> {
        val board = Bitmap.createBitmap(
            screenshot,
            BOARD_RECT.left, BOARD_RECT.top,
            BOARD_RECT.width(), BOARD_RECT.height()
        )

        val cellW = board.width / 4.0
        val cellH = board.height / 4.0
        val margin = 0.05  // CELL_MARGIN_RATIO = 5%

        return Array(4) { r ->
            Array(4) { c ->
                val x = (c * cellW + cellW * margin).toInt()
                val y = (r * cellH + cellH * margin).toInt()
                val w = (cellW * (1 - 2 * margin)).toInt()
                val h = (cellH * (1 - 2 * margin)).toInt()
                Bitmap.createBitmap(board, x.coerceAtLeast(0), y.coerceAtLeast(0),
                    w.coerceAtMost(board.width - x), h.coerceAtMost(board.height - y))
            }
        }
    }

    fun swipeCoordinates(direction: String): Pair<Pair<Float, Float>, Pair<Float, Float>> {
        val cx = BOARD_RECT.centerX().toFloat()
        val cy = BOARD_RECT.centerY().toFloat()
        val mx = BOARD_RECT.width() * 0.15f
        val my = BOARD_RECT.height() * 0.15f
        val il = BOARD_RECT.left + mx
        val ir = BOARD_RECT.right - mx
        val it = BOARD_RECT.top + my
        val ib = BOARD_RECT.bottom - my

        return when (direction) {
            "up"    -> Pair(cx to ib, cx to it)
            "down"  -> Pair(cx to it, cx to ib)
            "left"  -> Pair(ir to cy, il to cy)
            "right" -> Pair(il to cy, ir to cy)
            else    -> Pair(cx to ib, cx to it)
        }
    }
}
