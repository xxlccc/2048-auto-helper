package com.game2048.ai.ui

import android.content.Context
import android.graphics.PixelFormat
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.WindowManager
import kotlin.math.abs

class FloatingButton(
    private val ctx: Context,
    private val onToggle: (Boolean) -> Unit,
    private val onLongPress: () -> Unit
) {
    private val wm = ctx.getSystemService(Context.WINDOW_SERVICE) as WindowManager
    private var isActive = false

    private val view = View(ctx)
    private var bgGray: android.graphics.drawable.Drawable? = null
    private var bgGreen: android.graphics.drawable.Drawable? = null

    init {
        bgGray = ctx.resources.getDrawable(R.drawable.circle_gray, ctx.theme)
        bgGreen = ctx.resources.getDrawable(R.drawable.circle_green, ctx.theme)
        view.background = bgGray
    }

    private val params = WindowManager.LayoutParams(
        80, 80,
        WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
        WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE,
        PixelFormat.TRANSLUCENT
    ).apply {
        gravity = Gravity.TOP or Gravity.START
        x = 100
        y = 300
    }

    init {
        var startX = 0f
        var startY = 0f
        var downTime = 0L

        view.setOnTouchListener { _, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    startX = event.rawX
                    startY = event.rawY
                    downTime = System.currentTimeMillis()
                    true
                }
                MotionEvent.ACTION_UP -> {
                    val dx = abs(event.rawX - startX)
                    val dy = abs(event.rawY - startY)
                    if (dx < 10 && dy < 10) {
                        if (System.currentTimeMillis() - downTime > 2000) {
                            onLongPress()
                        } else {
                            isActive = !isActive
                            updateColor()
                            onToggle(isActive)
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

    fun show() {
        try { wm.addView(view, params) } catch (_: Exception) {}
    }

    fun hide() {
        try { wm.removeView(view) } catch (_: Exception) {}
    }

    fun setActive(active: Boolean) {
        isActive = active
        updateColor()
    }

    private fun updateColor() {
        view.background = if (isActive) bgGreen else bgGray
    }
}
