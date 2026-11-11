package com.game2048.ai.engine

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max

object Engine {

    data class Spawn(val value: Int, val prob: Double)

    // 蛇形权重 (与 PC 版 engine.py 完全一致)
    private val SNAKE_RAW = arrayOf(
        doubleArrayOf(10.0, 8.0, 7.0, 6.5),
        doubleArrayOf(0.5, 0.7, 1.0, 3.0),
        doubleArrayOf(-0.5, -1.5, -1.8, -2.0),
        doubleArrayOf(-3.8, -3.7, -3.5, -3.0)
    )
    private val SNAKE_BL = SNAKE_RAW.map { r -> r.map { exp(it) }.toDoubleArray() }.toTypedArray()
    private val SNAKE_BR = SNAKE_BL.map { it.reversedArray() }.toTypedArray()
    private val SNAKE_TL = SNAKE_BL.reversedArray()
    private val SNAKE_TR = SNAKE_TL.map { it.reversedArray() }.toTypedArray()
    private val ALL_SNAKES = arrayOf(SNAKE_BL, SNAKE_BR, SNAKE_TL, SNAKE_TR)

    private val transTable = HashMap<String, Pair<Int, Double>>()

    fun evaluate(board: Array<IntArray>): Double {
        val empty = Board.countEmpty(board).toDouble()

        var bestSnake = 0.0
        for (snake in ALL_SNAKES) {
            var s = 0.0
            for (r in 0 until 4)
                for (c in 0 until 4) {
                    val v = board[r][c]
                    if (v > 0) s += (1 shl v) * snake[r][c]
                }
            bestSnake = max(bestSnake, s)
        }

        var smooth = 0.0
        for (r in 0 until 4)
            for (c in 0 until 4) {
                val v = board[r][c]
                if (v == 0) continue
                if (c + 1 < 4 && board[r][c + 1] != 0)
                    smooth -= abs(v - board[r][c + 1]).toDouble()
                if (r + 1 < 4 && board[r + 1][c] != 0)
                    smooth -= abs(v - board[r + 1][c]).toDouble()
            }

        var merges = 0
        for (r in 0 until 4) {
            for (c in 0 until 3)
                if (board[r][c] != 0 && board[r][c] == board[r][c + 1]) merges++
        }
        for (r in 0 until 3)
            for (c in 0 until 4)
                if (board[r][c] != 0 && board[r][c] == board[r + 1][c]) merges++

        return bestSnake * 1.0 + empty * 5.0 + smooth * 0.5 + merges * 3.0
    }

    fun getMove(
        board: Array<IntArray>,
        strategy: String = "standard",
        spawns: List<Spawn> = listOf(Spawn(1, 0.75), Spawn(2, 0.20), Spawn(3, 0.05))
    ): String {
        val empty = Board.countEmpty(board)
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

    private fun boardKey(board: Array<IntArray>): String {
        val sb = StringBuilder()
        for (r in 0 until 4) {
            sb.append(board[r].joinToString(","))
            sb.append("|")
        }
        return sb.toString()
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
            for (d in orderMoves(board)) {
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
            for (r in 0 until 4)
                for (c in 0 until 4)
                    if (board[r][c] == 0) empty.add(Pair(r, c))

            if (empty.isEmpty()) {
                val ev = evaluate(board)
                transTable[key] = Pair(0, ev)
                return Pair(ev, null)
            }

            val sampled = if (empty.size > 8) empty.shuffled().take(8) else empty
            var total = 0.0
            for ((r, c) in sampled) {
                for (sp in spawns) {
                    val nb = Board.clone(board)
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
