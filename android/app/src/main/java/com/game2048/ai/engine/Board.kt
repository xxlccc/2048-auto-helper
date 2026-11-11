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
            "left" -> {
                val (r, s) = moveLeft(board)
                MoveResult(r, s, !boardsEqual(r, board))
            }
            "right" -> {
                val m = mirror(board)
                val (r, s) = moveLeft(m)
                MoveResult(mirror(r), s, !boardsEqual(r, board))
            }
            "up" -> {
                val t = transpose(board)
                val (r, s) = moveLeft(t)
                MoveResult(transpose(r), s, !boardsEqual(r, board))
            }
            else -> { // down
                val mt = mirror(transpose(board))
                val (r, s) = moveLeft(mt)
                MoveResult(transpose(mirror(r)), s, !boardsEqual(r, board))
            }
        }
    }

    fun mirror(board: Array<IntArray>): Array<IntArray> {
        return Array(SIZE) { r -> board[r].reversedArray() }
    }

    fun transpose(board: Array<IntArray>): Array<IntArray> {
        return Array(SIZE) { r -> IntArray(SIZE) { c -> board[c][r] } }
    }

    private fun moveLeft(board: Array<IntArray>): Pair<Array<IntArray>, Int> {
        val result = Array(SIZE) { IntArray(SIZE) }
        var totalScore = 0
        for (r in 0 until SIZE) {
            val (row, score) = slideRowLeft(board[r])
            result[r] = row
            totalScore += score
        }
        return Pair(result, totalScore)
    }

    private fun slideRowLeft(row: IntArray): Pair<IntArray, Int> {
        val nonZero = row.filter { it != 0 }
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
            if (!a[r].contentEquals(b[r])) return false
        }
        return true
    }

    fun clone(board: Array<IntArray>): Array<IntArray> {
        return Array(SIZE) { r -> board[r].clone() }
    }

    fun countEmpty(board: Array<IntArray>): Int {
        return board.sumOf { row -> row.count { it == 0 } }
    }
}
