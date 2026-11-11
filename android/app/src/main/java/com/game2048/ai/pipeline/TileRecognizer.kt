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
        // 中心裁方 → 64×64
        val size = minOf(cell.width, cell.height)
        val x = (cell.width - size) / 2
        val y = (cell.height - size) / 2
        val cropped = Bitmap.createBitmap(cell, x, y, size, size)
        val resized = Bitmap.createScaledBitmap(cropped, 64, 64, true)

        // Bitmap → Tensor (RGB, 3×64×64)
        val tensor = TensorImageUtils.bitmapToFloat32Tensor(
            resized,
            floatArrayOf(0f, 0f, 0f),
            floatArrayOf(1f, 1f, 1f)
        )

        val output = module.forward(IValue.from(tensor)).toTensor()
        val scores = output.dataAsFloatArray
        var bestIdx = 0
        var bestScore = scores[0]
        for (i in 1 until scores.size) {
            if (scores[i] > bestScore) {
                bestScore = scores[i]
                bestIdx = i
            }
        }

        val className = classes[bestIdx]
        val value = if (className == "empty") 0 else className.toIntOrNull() ?: 0
        return Pair(value, bestScore)
    }

    private fun assetFilePath(name: String): String {
        val f = File(App.instance.filesDir, name)
        if (!f.exists()) {
            App.instance.assets.open(name).use { src ->
                f.outputStream().use { dst -> src.copyTo(dst) }
            }
        }
        return f.absolutePath
    }
}
