package com.example.pocapp

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import kotlin.math.max

data class Detection(val box: RectF, val score: Float, val label: String)

class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private var dets: List<Detection> = emptyList()
    private var srcW = 0
    private var srcH = 0

    private val boxPaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 4f
        isAntiAlias = true
    }
    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 32f
        isAntiAlias = true
    }
    private val bgPaint = Paint().apply { color = Color.BLACK }

    fun updateDetections(d: List<Detection>, w: Int, h: Int) {
        dets = d
        srcW = w
        srcH = h
        postInvalidateOnAnimation()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (srcW == 0 || srcH == 0) return

        // PreviewView default scale type = FILL_CENTER:
        val scale = max(width / srcW.toFloat(), height / srcH.toFloat())
        val dx = (width  - srcW * scale) / 2f
        val dy = (height - srcH * scale) / 2f

        for (d in dets) {
            val l = dx + d.box.left   * scale
            val t = dy + d.box.top    * scale
            val r = dx + d.box.right  * scale
            val b = dy + d.box.bottom * scale
            canvas.drawRect(l, t, r, b, boxPaint)

            val label = "${d.label} ${(d.score * 100).toInt()}%"
            val pad = 6f
            val tw = textPaint.measureText(label)
            val th = textPaint.textSize
            canvas.drawRect(l, t - th - pad * 2, l + tw + pad * 2, t, bgPaint)
            canvas.drawText(label, l + pad, t - pad, textPaint)
        }
    }
}
