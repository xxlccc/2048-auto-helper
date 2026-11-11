package com.game2048.ai.ui

import android.app.AlertDialog
import android.content.Context
import android.widget.EditText
import android.widget.LinearLayout
import com.game2048.ai.service.GameService

class SettingsDialog(ctx: Context, private val service: GameService?) : AlertDialog.Builder(ctx) {

    init {
        setTitle("设置")

        val items = arrayOf(
            "等级 (当前: ${service?.level ?: 1})",
            "策略 (当前: ${service?.strategy ?: "standard"})",
            "滑动后延迟 (当前: ${service?.postDelay ?: 660}ms)"
        )

        setItems(items) { _, which ->
            when (which) {
                0 -> showLevelPicker(ctx)
                1 -> showStrategyPicker(ctx)
                2 -> showDelayInput(ctx)
            }
        }

        setNegativeButton("关闭", null)
    }

    private fun showLevelPicker(ctx: Context) {
        AlertDialog.Builder(ctx).apply {
            setTitle("选择等级")
            setItems(arrayOf("1 (75/20/5)", "2 (75/20/5)", "3 (70/20/10)")) { _, i ->
                service?.level = i + 1
            }
            show()
        }
    }

    private fun showStrategyPicker(ctx: Context) {
        AlertDialog.Builder(ctx).apply {
            setTitle("选择策略")
            setItems(arrayOf("conservative", "standard", "aggressive")) { _, i ->
                service?.strategy = listOf("conservative", "standard", "aggressive")[i]
            }
            show()
        }
    }

    private fun showDelayInput(ctx: Context) {
        val input = EditText(ctx).apply {
            hint = "毫秒"
            setText((service?.postDelay ?: 660).toString())
        }
        AlertDialog.Builder(ctx).apply {
            setTitle("滑动后延迟 (ms)")
            setView(input)
            setPositiveButton("确定") { _, _ ->
                input.text.toString().toLongOrNull()?.let {
                    if (it in 100..5000) service?.postDelay = it
                }
            }
            show()
        }
    }
}
