Road Sense – Red Shapes POC (Android + YOLOv8/TFLite)

A lightweight Android proof-of-concept that uses the device camera to detect red geometric shapes in real time. It’s a stepping stone toward an ADAS-style app that will later detect traffic signs, hazards, pedestrians, and speed limits.

✨ Highlights

CameraX live preview + ImageAnalysis pipeline (RGBA_8888).

TensorFlow Lite YOLOv8 model (exported from Ultralytics).

Aspect-ratio correct preprocessing (letterbox) and accurate overlay mapping (PreviewView “FILL + center-crop” math).

Configurable thresholds, NMS, and tiny-box filtering to reduce false positives.

Helpful logs for shape/label debugging (YOLODBG, CAMDBG).

📁 Project structure
app/
 ├─ src/main/
 │   ├─ java/com/example/pocapp/
 │   │   ├─ MainActivity.kt          # Camera pipeline + analyzer
 │   │   ├─ YoloTFLite.kt            # TFLite wrapper + letterbox + parsing + NMS
 │   │   ├─ OverlayView.kt           # Draws boxes aligned with PreviewView
 │   │   └─ Detection.kt             # data class Detection(box, score, label)
 │   ├─ res/layout/
 │   │   └─ activity_main.xml        # PreviewView + OverlayView
 │   └─ assets/
 │       ├─ best_float16.tflite      # YOLOv8 model (copied by you)
 │       ├─ labels.txt               # One class name per line (edit order here)
 │       └─ test_square.jpg          # (optional) static self-test image
 ├─ AndroidManifest.xml              # Camera permission
 └─ build.gradle.kts                 # Android + TFLite dependencies

🧱 Requirements

Android Studio Giraffe+ (or newer) with Kotlin.

Android device with Camera2 support (most Android 8.0+ devices).

(Training/export) Python 3.10+ with ultralytics.

🚀 Build & Run

Open the project in Android Studio.

Put your model and labels in app/src/main/assets/:

best_float16.tflite (from Ultralytics export)

labels.txt (one class per line, in model class index order)

Ensure android.permission.CAMERA is in AndroidManifest.xml.

Run on a real device (USB or Wi-Fi).
The app will ask for camera permission on first launch.

Tip: You can drop a test_square.jpg into assets/ and the app will run a one-off self-test on startup (logged under CAMDBG).

⚙️ Configuration & Tuning

Open YoloTFLite.kt:

Input size: The app reads the size from the model automatically. If you exported with imgsz=416, that’s what it will use.

Box convention switch:

private const val ASSUME_TOPLEFT_XY = true


true → Treat YOLO head outputs as (x,y) = top-left (width/height as-is).

false → Treat as (cx,cy) = center.
This fixed the “boxes stuck to top edge” symptom.

Confidence / IoU / tiny box filter (defaults are conservative):

fun detect(src, confThres = 0.50f, iouThres = 0.45f)
private const val MIN_AREA_FRAC = 0.0015f // ~0.15% of image area


Increase confThres (e.g., 0.55–0.60) and/or MIN_AREA_FRAC to reduce random flickers.

Debug label IDs: Detections display as c<ID>:<name> so you can verify that your labels.txt ordering matches the model’s class indices. After you confirm, you can change the display to just name.

🧠 How the pipeline works

CameraX delivers frames to the analyzer in RGBA_8888.

We rotate the frame to upright (rotationDegrees) and convert to Bitmap.

Letterbox preprocessing keeps the original aspect ratio:

resize to fit the network size,

pad to full size (black),

normalize to 0..1 floats (NHWC RGB).

TFLite inference using XNNPACK (multi-threaded).

Output parsing: supports [1, C, N] and [1, N, C].

Applies sigmoid to objectness and class logits.

Handles both top-left and center box conventions via ASSUME_TOPLEFT_XY.

Converts to xyxy in input space, removes padding, then inverse scales back to the original upright bitmap coordinates.

Post-processing: confidence gating, tiny-box filter, class-agnostic NMS.

Overlay alignment: OverlayView uses PreviewView’s FILL + center-crop mapping:

scale = max(viewW/srcW, viewH/srcH)
dx = (viewW - srcW*scale)/2
dy = (viewH - srcH*scale)/2


This keeps boxes visually aligned regardless of phone/tablet aspect ratios.

🪵 Logging & Debugging

Model/labels:
YOLODBG I → model/labels loaded, tensor shapes.
YOLODBG D raw[i] → first few rows of head outputs to confirm conventions.

Per-frame:
CAMDBG D → input sizes, rotation, inference time.
YOLODBG D frame stats → candidate count, max objectness/class values, sample detections.

Common symptoms and fixes:

Symptom	Likely cause	Fix
Boxes hug top or bottom edges	Treating y center as top-left (or vice versa)	Flip ASSUME_TOPLEFT_XY
Boxes offset from objects	Overlay math not matching PreviewView	Use provided OverlayView.kt (FILL + center-crop)
Random small boxes	Low conf / noise	Raise confThres, raise MIN_AREA_FRAC
Wrong label text	labels.txt order doesn’t match model index	Use c<ID>: prefix to map IDs → reorder lines in labels.txt
Mapping labels correctly

Point the camera at a single shape.

Read the overlay: e.g., c2:red_triangle.

Ensure the 3rd line (index 2) in labels.txt is red_triangle.

Repeat for all classes until names match what you see.

🧪 Static self-test (optional)

Place test_square.jpg in assets/. On startup you’ll see in Logcat:

CAMDBG  I  SELFTEST static dets=...


Useful for smoke-testing the model on a known image without the camera.

🧬 Training & Export (Ultralytics)

Example using your dataset YAML (road_shapes.yaml):

# 1) Install
pip install ultralytics

# 2) Train
yolo train \
  model=yolov8n.pt \
  data=road_shapes.yaml \
  imgsz=416 \
  epochs=100 \
  name=y8n_416_redshapes

# 3) (optional) Validate
yolo val \
  model=runs/detect/y8n_416_redshapes/weights/best.pt \
  data=road_shapes.yaml \
  imgsz=416

# 4) Export to TFLite (float16)
yolo export \
  model=runs/detect/y8n_416_redshapes/weights/best.pt \
  format=tflite \
  imgsz=416 \
  half=True


Copy the exported best_float16.tflite into app/src/main/assets/.

INT8 (faster, smaller) – optional

You’ll need a small calibration set:

yolo export \
  model=.../best.pt \
  format=tflite \
  int8=True \
  imgsz=416 \
  data=road_shapes.yaml   # or a folder passed via --quantize-data


If you switch to INT8 and see accuracy drop, try per-channel quantization or larger imgsz.

⏱️ Performance tips

Use a smaller model (yolov8n) and/or smaller imgsz (e.g., 320) at export time.

Keep XNNPACK threads modest (2–4). More isn’t always faster on mobile.

If you don’t need full-resolution preview, you can scale the analyzer input to match the model size before conversion (we already letterbox to the model’s size internally).
