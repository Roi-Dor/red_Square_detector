package com.example.pocapp

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioManager
import android.media.ToneGenerator
import android.os.Bundle
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import android.graphics.Bitmap
import java.util.concurrent.Executors
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity

class MainActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var overlay: OverlayView
    private lateinit var yolo: YoloTFLite
    private val analyzerExecutor = Executors.newSingleThreadExecutor()
    private val beeper = ToneGenerator(AudioManager.STREAM_NOTIFICATION, 80)
    private var lastBeep = 0L

    private val reqPerm = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted -> if (granted) startCamera() else finish() }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        previewView = findViewById(R.id.previewView)
        overlay = findViewById(R.id.overlay)

        yolo = YoloTFLite(this, "best_float16.tflite", 320, 320)

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            reqPerm.launch(Manifest.permission.CAMERA)
        }
    }

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val provider = providerFuture.get()

            // 1) Ask Preview and ImageAnalysis to use the same rotation as the PreviewView
            val targetRotation = previewView.display.rotation

            val preview = Preview.Builder()
                .setTargetRotation(targetRotation)
                .build().also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }

            val analysis = ImageAnalysis.Builder()
                .setTargetRotation(targetRotation)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()

            analysis.setAnalyzer(analyzerExecutor) { image ->
                // Convert to Bitmap
                val bmp = imageProxyToBitmap(image)

                // 2) Rotate the bitmap to upright if needed (0, 90, 180, 270)
                val rot = image.imageInfo.rotationDegrees
                val upright = if (rot != 0) {
                    val m = android.graphics.Matrix().apply { postRotate(rot.toFloat()) }
                    Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
                } else bmp

                // 3) Run the model on the **upright** bitmap
                val dets = yolo.detect(upright, confThres = 0.8f)

                // 4) Tell the overlay the size of the **upright** source image
                overlay.updateDetections(dets, upright.width, upright.height)

                // (optional) beep…
                val now = System.currentTimeMillis()
                if (dets.isNotEmpty() && now - lastBeep > 500) {
                    beeper.startTone(ToneGenerator.TONE_PROP_BEEP, 120)
                    lastBeep = now
                }

                image.close()
            }

            provider.unbindAll()
            provider.bindToLifecycle(
                this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis
            )
        }, ContextCompat.getMainExecutor(this))
    }

    /** Convert ImageProxy (RGBA_8888) -> Bitmap */
    private fun imageProxyToBitmap(image: ImageProxy): Bitmap {
        val plane = image.planes[0]  // single plane in RGBA_8888
        val buffer = plane.buffer
        buffer.rewind()
        val bmp = Bitmap.createBitmap(image.width, image.height, Bitmap.Config.ARGB_8888)
        bmp.copyPixelsFromBuffer(buffer)
        return bmp
    }

    override fun onDestroy() {
        super.onDestroy()
        beeper.release()
        analyzerExecutor.shutdown()
    }
}
