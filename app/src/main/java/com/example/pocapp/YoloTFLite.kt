package com.example.pocapp

import android.content.Context
import android.graphics.*
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

class YoloTFLite(
    context: Context,
    modelPath: String = "best_float16.tflite",
    private val inputW: Int = 320,
    private val inputH: Int = 320,
    numThreads: Int = 4
) {
    private val interpreter: Interpreter

    init {
        val opts = Interpreter.Options().apply {
            setNumThreads(numThreads)
            // setUseNNAPI(true) // optional
        }
        interpreter = Interpreter(loadModelFile(context, modelPath), opts)
    }

    private fun loadModelFile(context: Context, path: String): ByteBuffer {
        val fd = context.assets.openFd(path)
        FileInputStream(fd.fileDescriptor).use { fis ->
            val channel = fis.channel
            return channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.length)
        }
    }

    // Allocate once
    private val imgBuffer: ByteBuffer =
        ByteBuffer.allocateDirect(4 * inputW * inputH * 3).order(ByteOrder.nativeOrder())

    private val intValues = IntArray(inputW * inputH)

    /** Preprocess: Bitmap (ARGB_8888) -> NCHW float32 [0..1] */
    private fun preprocess(src: Bitmap): ByteBuffer {
        val resized = Bitmap.createScaledBitmap(src, inputW, inputH, true)
        imgBuffer.rewind()
        resized.getPixels(intValues, 0, inputW, 0, 0, inputW, inputH)
        var i = 0
        for (y in 0 until inputH) {
            for (x in 0 until inputW) {
                val p = intValues[i++]
                val r = (p shr 16 and 0xFF) / 255f
                val g = (p shr 8 and 0xFF) / 255f
                val b = (p and 0xFF) / 255f
                imgBuffer.putFloat(r)
                imgBuffer.putFloat(g)
                imgBuffer.putFloat(b)
            }
        }
        return imgBuffer
    }

    /** Postprocess: output [1, 2100, 5] -> list of boxes in model space (320x320) */
    private fun postprocess(raw: Array<Array<FloatArray>>, confThres: Float = 0.5f, iouThres: Float = 0.45f): List<Detection> {
        val preds = raw[0] // [2100][5] = cx, cy, w, h, conf
        val out = mutableListOf<Detection>()
        for (i in preds.indices) {
            val cx = preds[i][0]
            val cy = preds[i][1]
            val w = preds[i][2]
            val h = preds[i][3]
            val conf = preds[i][4]
            if (conf < confThres) continue

            // Heuristic: if values look normalized, scale to pixels
            val (pxW, pxH) = if (w <= 2f && h <= 2f) Pair(w * inputW, h * inputH) else Pair(w, h)
            val (pxCx, pxCy) = if (cx <= 2f && cy <= 2f) Pair(cx * inputW, cy * inputH) else Pair(cx, cy)

            val left = (pxCx - pxW / 2f).coerceIn(0f, inputW.toFloat())
            val top = (pxCy - pxH / 2f).coerceIn(0f, inputH.toFloat())
            val right = (pxCx + pxW / 2f).coerceIn(0f, inputW.toFloat())
            val bottom = (pxCy + pxH / 2f).coerceIn(0f, inputH.toFloat())

            out.add(Detection(RectF(left, top, right, bottom), conf, "red_square"))
        }
        return nms(out, iouThres)
    }

    private fun iou(a: RectF, b: RectF): Float {
        val inter = RectF()
        val has = inter.setIntersect(a, b)
        if (!has) return 0f
        val interArea = inter.width() * inter.height()
        val union = a.width() * a.height() + b.width() * b.height() - interArea
        return if (union <= 0f) 0f else (interArea / union)
    }

    private fun nms(dets: List<Detection>, iouThres: Float): List<Detection> {
        val sorted = dets.sortedByDescending { it.score }.toMutableList()
        val keep = mutableListOf<Detection>()
        while (sorted.isNotEmpty()) {
            val best = sorted.removeAt(0)
            keep.add(best)
            val it = sorted.iterator()
            while (it.hasNext()) {
                val d = it.next()
                if (iou(best.box, d.box) > iouThres) it.remove()
            }
        }
        return keep
    }

    fun detect(
        src: Bitmap,
        confThres: Float = 0.35f,   // start lower for debugging
        iouThres: Float = 0.45f
    ): List<Detection> {

        // 1) Shapes
        val inShape = interpreter.getInputTensor(0).shape() // [1,H,W,3]
        val H = inShape[1]
        val W = inShape[2]

        // 2) Preprocess (RGB float32 0..1, NHWC)
        val resized = if (src.width != W || src.height != H)
            Bitmap.createScaledBitmap(src, W, H, true) else src

        val input = ByteBuffer.allocateDirect(4 * W * H * 3).order(ByteOrder.nativeOrder())
        val pix = IntArray(W * H)
        resized.getPixels(pix, 0, W, 0, 0, W, H)
        var i = 0
        while (i < pix.size) {
            val p = pix[i++]
            input.putFloat(((p ushr 16) and 255) / 255f) // R
            input.putFloat(((p ushr 8) and 255) / 255f)  // G
            input.putFloat((p and 255) / 255f)          // B
        }
        input.rewind()

        // 3) Prepare output container
        val outShape = interpreter.getOutputTensor(0).shape() // either [1,5,N] or [1,N,5]
        val output: Array<Array<FloatArray>> =
            if (outShape.size == 3 && outShape[1] == 5)
                Array(1) { Array(5) { FloatArray(outShape[2]) } } // [1,5,N]
            else if (outShape.size == 3 && outShape[2] == 5)
                Array(1) { Array(outShape[1]) { FloatArray(5) } } // [1,N,5]
            else
                throw IllegalStateException("Unexpected YOLO output shape ${outShape.contentToString()}")

        // 4) Inference
        interpreter.run(input, output)

        // 5) Normalize to Nx5
        val N = if (outShape[1] == 5) outShape[2] else outShape[1]
        val preds = Array(N) { FloatArray(5) }
        if (outShape[1] == 5) {
            for (k in 0 until 5) for (j in 0 until N) preds[j][k] = output[0][k][j]
        } else {
            for (j in 0 until N) for (k in 0 until 5) preds[j][k] = output[0][j][k]
        }

        // 6) Decide coord format (xywh vs xyxy)
        var assumeXYWH = true
        run {
            var votesXYXY = 0
            val sample = minOf(N, 32)
            for (s in 0 until sample) {
                val a = preds[s]
                if (a[2] > a[0] && a[3] > a[1]) votesXYXY++
            }
            if (votesXYXY > sample / 2) assumeXYWH = false
        }

        // 7) Build detections (scale to ORIGINAL bitmap coords)
        val sx = src.width.toFloat() / W
        val sy = src.height.toFloat() / H
        val candidates = ArrayList<Detection>(64)

        for (j in 0 until N) {
            val a = preds[j]
            val score = a[4]
            if (score < confThres) continue

            // coords in model space
            val (x1m, y1m, x2m, y2m) = if (assumeXYWH) {
                val cx = a[0]; val cy = a[1]; val bw = a[2]; val bh = a[3]
                floatArrayOf(cx - bw / 2f, cy - bh / 2f, cx + bw / 2f, cy + bh / 2f)
            } else {
                floatArrayOf(a[0], a[1], a[2], a[3])
            }

            // if normalized (<= ~2), scale to model pixels first
            val Sx = if (x1m <= 2f && x2m <= 2f) W.toFloat() else 1f
            val Sy = if (y1m <= 2f && y2m <= 2f) H.toFloat() else 1f

            val x1 = (x1m * Sx * sx).coerceIn(0f, src.width.toFloat())
            val y1 = (y1m * Sy * sy).coerceIn(0f, src.height.toFloat())
            val x2 = (x2m * Sx * sx).coerceIn(0f, src.width.toFloat())
            val y2 = (y2m * Sy * sy).coerceIn(0f, src.height.toFloat())

            if (x2 > x1 && y2 > y1) {
                candidates += Detection(android.graphics.RectF(x1, y1, x2, y2), score, "red_square")
            }
        }

        // 8) NMS
        candidates.sortByDescending { it.score }
        val picked = ArrayList<Detection>(candidates.size)
        outer@ for (d in candidates) {
            for (p in picked) if (iouRect(d.box, p.box) > iouThres) continue@outer
            picked += d
        }
        return picked
    }



    /** IoU for RectF. Named 'iouRect' to avoid clashing with any existing 'iou'. */
    private fun iouRect(a: android.graphics.RectF, b: android.graphics.RectF): Float {
        val ix = maxOf(0f, minOf(a.right, b.right) - maxOf(a.left, b.left))
        val iy = maxOf(0f, minOf(a.bottom, b.bottom) - maxOf(a.top, b.top))
        val inter = ix * iy
        if (inter <= 0f) return 0f
        val union = a.width() * a.height() + b.width() * b.height() - inter
        return if (union <= 0f) 0f else inter / union
    }

}

