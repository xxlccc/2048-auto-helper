package com.game2048.ai

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.provider.Settings
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.game2048.ai.service.GameService
import com.game2048.ai.ui.FloatingButton
import com.game2048.ai.ui.SettingsDialog

class MainActivity : AppCompatActivity() {
    private lateinit var floatingButton: FloatingButton

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(android.R.layout.activity_list_item)

        // 引导开启悬浮窗权限
        if (!Settings.canDrawOverlays(this)) {
            Toast.makeText(this, "请允许悬浮窗权限", Toast.LENGTH_LONG).show()
            startActivity(Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                Uri.parse("package:$packageName")))
        }

        // 引导开启无障碍服务
        Toast.makeText(this,
            "请到 设置 → 无障碍 → 2048 AI 开启服务", Toast.LENGTH_LONG).show()

        floatingButton = FloatingButton(this,
            onToggle = { active ->
                if (active) {
                    GameService.instance?.start()
                } else {
                    GameService.instance?.stop()
                }
            },
            onLongPress = {
                SettingsDialog(this, GameService.instance).show()
            }
        )
        floatingButton.show()

        // 结束 Activity，留悬浮窗在桌面
        finish()
    }

    override fun onDestroy() {
        super.onDestroy()
        floatingButton.hide()
    }
}
