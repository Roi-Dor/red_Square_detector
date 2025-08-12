# Road Sense – Android ADAS POC

A minimal on-device prototype that detects a **red square** in real time using **CameraX** and **TensorFlow Lite (YOLO)**, then overlays the detection on the live preview and plays a short beep. This POC is the foundation for a road-safety app that will later detect traffic signs, pedestrians, and hazards.

## Demo
_Coming soon (GIF/video)._

## Pipeline (how it works)
1. **CameraX** streams frames to a `PreviewView` and an `ImageAnalysis` analyzer using the same rotation.
2. The analyzer converts the `ImageProxy` to a **Bitmap** and rotates it **upright** (0/90/180/270).
3. The **YOLO TFLite** model (320×320) runs on the upright bitmap and outputs candidate boxes.
4. **Post-processing** converts model-space boxes to **original bitmap coordinates** and applies **NMS**.
5. **OverlayView** mirrors `PreviewView`’s scale (`FILL_CENTER`), maps source→view (`scale + dx/dy`), and draws boxes.
6. If any detection exists, a **beep** is played (throttled to once every 500 ms).

## Project structure
