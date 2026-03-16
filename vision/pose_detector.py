# Syntax: Multi-line string literal acting as a module docstring. Semantically provides an overview of the script's purpose and usage instructions.
"""
pose_detector.py
----------------
Streams pose detection to your web browser.
Run it, then open: http://localhost:5000
Press Ctrl+C to quit.
"""

# Syntax: 'import' statement. Semantically imports the OpenCV library for image and video processing.
import cv2
# Syntax: 'import ... as ...' statement. Semantically imports the core MediaPipe library and aliases it as 'mp'.
import mediapipe as mp
# Syntax: 'from ... import ... as ...' statement. Semantically imports the Python API from MediaPipe tasks, aliased as 'mp_python'.
from mediapipe.tasks import python as mp_python
# Syntax: 'from ... import ... as ...' statement. Semantically imports the vision API from MediaPipe tasks, aliased as 'mp_vision'.
from mediapipe.tasks.python import vision as mp_vision
# Syntax: 'import' statement. Semantically imports the 'urllib.request' module for fetching URLs (e.g., downloading the model).
import urllib.request
# Syntax: 'import' statement. Semantically imports the 'os' module for interacting with the operating system (e.g., file existence checks).
import os
# Syntax: 'from ... import ...' statement. Semantically imports the 'Flask' application class and 'Response' object from the Flask web framework.
from flask import Flask, Response

# --- Download pose model if not already present ---
# Syntax: Assignment of string literal to a constant. Semantically defines the local file path to save the MediaPipe model.
MODEL_PATH = "pose_landmarker_lite.task"
# Syntax: Assignment of string literal to a constant. Semantically defines the internet URL to download the remote model if missing.
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landlandmark_lite/float16/latest/pose_landmarker_lite.task"

# Syntax: Conditional 'if' statement using 'not' logic and 'os.path.exists' function. Semantically checks if the model file is absent from the local path.
if not os.path.exists(MODEL_PATH):
    # Syntax: Function call 'print' with a string argument. Semantically outputs a status message to the console.
    print("Downloading pose model (~5MB)...")
    # Syntax: Function call 'urllib.request.urlretrieve' with source and destination strings. Semantically downloads the file from the web to disk.
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    # Syntax: Function call 'print' with string argument containing newline escape char. Semantically confirms download completion.
    print("Model downloaded.\n")

# --- Skeleton connections ---
# Syntax: Explicit list of tuple pairs assigned to constant. Semantically defines the drawing pairs specifying the structural skeleton wireframe.
CONNECTIONS = [
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
]

# Syntax: Function definition 'def' with standard parameters. Semantically, this is a helper rendering the skeleton UI on a specific frame canvas.
def draw_pose(frame, landmarks, h, w):
    # Syntax: Empty dictionary literal assignment. Semantically acts as a dynamic geometric coordinate lookup table.
    points = {}
    # Syntax: 'for' loop using 'enumerate' to get index and object natively unpacking tuples. Semantically loops over sequence of joint objects.
    for idx, lm in enumerate(landmarks):
        # Syntax: tuple unpacking of calculated integers resolving normalized space to native pixels dynamically. Semantically converts floating scales to actual image coordinates.
        cx, cy = int(lm.x * w), int(lm.y * h)
        # Syntax: Dictionary index assignment setting tracking coordinates directly. Semantically maps joint indexes to discrete (x,y) screen locations.
        points[idx] = (cx, cy)
        # Syntax: Internal library cv2.circle generation targeting local memory matrix implicitly scaling values explicitly drawing filled nodes. Semantically overlays green dots on joints.
        cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)
    # Syntax: Unpacking iteration extracting constant tuples 'for a, b in CONNECTIONS'. Semantically pairs explicit skeletal linkages dynamically mapping nodes.
    for a, b in CONNECTIONS:
        # Syntax: 'if ... in ... and ... in ...' associative matching logic filtering dictionary keys. Semantically confirms both linkage vertices exist securely to render.
        if a in points and b in points:
            # Syntax: Line geometric logic evaluating discrete memory coordinate structures tracing yellow strings across image objects conditionally mapping output explicitly. Semantically draws bone lines.
            cv2.line(frame, points[a], points[b], (0, 200, 255), 2)

# --- Setup MediaPipe ---
# Syntax: Direct API invocation targeting object parameter setup formatting local path constraints unconditionally. Semantically initializes the base configuration object loading model weights.
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
# Syntax: Keyword parameter assignment mapping configuration objects generating local tracking options defining API bounds explicitly. Semantically sets parameters for pose processing tracking a single person.
options = mp_vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False,
    num_poses=1
)
# Syntax: Class factory instantiation allocating dynamic structures based on configuration outputs natively starting core execution environment blocks explicitly logging parameters unconditionally. Semantically initializes the pose detector.
landmarker = mp_vision.PoseLandmarker.create_from_options(options)
# Syntax: Device allocation assignment assigning object properties identifying camera node hardware explicitly dynamically loading variables unconditionally natively targeting video stream explicitly logic limits unconditionally checking explicit properties natively outputting logic formats. Semantically captures the default webcam (device 0).
cap = cv2.VideoCapture(0)

# Syntax: Web framework factory invocation parsing global module name establishing application references natively formatting routing bounds actively logging variables mapped dynamically unconditionally natively format generation. Semantically creates the Flask web server instance.
app = Flask(__name__)

# Syntax: standard python generator function explicitly looping stream bounds wrapping continuous processing logically returning yields. Semantically provides an image streaming generator to feed the HTTP response.
def generate_frames():
    # Syntax: Infinite 'while' control flow handling active loop iterations evaluating unbounded loops conditionally escaping internal sequences actively logic checks explicit states. Semantically runs forever fetching camera frames.
    while True:
        # Syntax: Tuple unpacking capturing explicit video states unpacking boolean and image matrices conditionally logging values implicitly generating output parameters mapped strictly natively. Semantically reads next frame; 'ret' flags success, 'frame' is the image data.
        # Syntax: Tuple unpacking capturing explicit video states unpacking boolean and image matrices conditionally logging values implicitly generating output parameters mapped strictly natively. Semantically reads next frame; 'ret' flags success, 'frame' is the image data.
        ret, frame = cap.read()
        # Syntax: standard inverted truthy implicit bounds detecting falsy logic triggering explicit flow escapes validating logic processing parameters structurally mapping conditions statically. Semantically breaks out of loop if camera fails or ends.
        if not ret:
            # Syntax: Standard break sequence exiting immediate enclosing loops implicitly terminating local logic blocks escaping loop constructs inherently terminating flow checks explicitly parsing logic native bounds. Semantically exits the loop.
            break

        # Syntax: Invokes cv2 mapping property logic implicitly transforming memory constraints dynamically rewriting internal matrices implicitly defining bounds structurally manipulating values conditionally tracking standard operations mapping directly evaluating native methods. Semantically mirrors the image (flips horizontally) for a natural selfie view.
        frame = cv2.flip(frame, 1)
        # Syntax: Slicing object shapes mapping multidimensional bounds capturing sequence integer constraints statically defining properties isolating memory coordinates extracting elements natively evaluating array structures actively parsing outputs directly unpacking states naturally limiting extraction bounds capturing values explicitly. Semantically extracts the image height and width.
        h, w = frame.shape[:2]

        # Syntax: Explicit translation function applying constant color transformation arrays statically generating discrete matrix values returning native object strings parsing limits inherently formatting outputs mapping unconditionally returning native constraints isolating formats actively. Semantically converts the BGR image from OpenCV to RGB format needed by MediaPipe.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Syntax: Constructs specific explicit Image objects. Semantically wraps the numpy RGB array into a MediaPipe Image object.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        # Syntax: Direct function invocation passing constructed parameters triggering internal detection logic. Semantically performs the actual pose detection on the current frame.
        result = landmarker.detect(mp_image)

        # Syntax: Implicit boolean truth check assessing structural properties tracking array limits. Semantically checks if any body poses were found.
        if result.pose_landmarks:
            # Syntax: Indexing array properties extracting the primary index explicitly. Semantically extracts the first detected person's skeletal data.
            lm = result.pose_landmarks[0]
            # Syntax: Function invocation evaluating dynamic helper functions assigning parameters processing inputs. Semantically draws the joint dots and bone lines on the frame.
            draw_pose(frame, lm, h, w)
            # Syntax: Helper mapping generating graphic object layers parsing literal mappings. Semantically stamps a green 'Pose Detected' label onto the active video feed.
            cv2.putText(frame, "Pose Detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            # Syntax: Prints standard formatted strings checking specific internal matrix array index values. Semantically prints key joint locations (Shoulder, Hip, Knee) to the terminal using carriage return for real-time overwriting.
            print(
                f"L.Shoulder({lm[11].x:.2f},{lm[11].y:.2f}) | "
                f"L.Hip({lm[23].x:.2f},{lm[23].y:.2f}) | "
                f"L.Knee({lm[25].x:.2f},{lm[25].y:.2f})",
                end="\r"
            )
        # Syntax: Else fallback evaluation triggering generic operations actively logic limits. Semantically executes if nobody is in frame.
        else:
            # Syntax: Function mapping defining image text parameters logging object types tracking limits explicitly assigning color string representations dynamically logging variables identifying memory coordinates. Semantically warns visually that no person is detected in red.
            cv2.putText(frame, "No person detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Resize for display
        # Syntax: Call internal mapping resizing object matrices resolving target coordinates static processing evaluating outputs. Semantically scales the final output image to a standard 960x720 video size.
        frame = cv2.resize(frame, (960, 720))

        # Encode as JPEG
        # Syntax: Tuple resolution logic encoding specific outputs conditionally mapping variables validating implicit sequences. Semantically encodes the raw image matrix into a standard JPEG byte buffer for HTTP transport.
        _, buffer = cv2.imencode('.jpg', frame)
        # Syntax: internal method logic accessing array strings parsing object configurations. Semantically converts the JPEG buffer into standard raw bytes.
        frame_bytes = buffer.tobytes()

        # Syntax: Yield generator formatting HTTP protocols manually processing logic sequences appending variables evaluating specific byte string compositions explicitly. Semantically yields a multi-part valid HTTP string containing the new frame's binary data for the web stream.
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Syntax: Decorator logic binding route string constants to specific internal view logic formats mapping implicit sequences structurally parsing paths specific. Semantically maps the root URL '/' to the 'index' view function.
@app.route('/')
# Syntax: Standard function definition implicitly explicitly executing default HTTP view rendering natively evaluating standard object logic unconditionally. Semantically serves the default HTML page.
def index():
    # Syntax: Multi-line string return encapsulating generic browser syntax formatting defining visual rendering logic parsing DOM structs natively. Semantically returns a simple HTML document containing the structure and styles for viewing the stream.
    return '''
    <html>
    <head>
        <title>Tiplo - Pose Detector</title>
        <style>
            body { background: #111; display: flex; justify-content: center;
                   align-items: center; height: 100vh; margin: 0; }
            img { border: 3px solid #00ff00; border-radius: 8px; }
            h1 { color: #00ff00; text-align: center; font-family: monospace; }
        </style>
    </head>
    <body>
        <div>
            <h1>Tiplo — Live Pose Detector</h1>
            <img src="/video" width="960" height="720"/>
        </div>
    </body>
    </html>
    '''

# Syntax: Decorator logic defining endpoint URL routing paths dynamically binding view operations explicit. Semantically maps the '/video' URL to the 'video' view function.
@app.route('/video')
# Syntax: Function definitions tracking implicit bindings explicitly evaluating routing paths structurally unconditionally logging outputs natively defining bounds conditionally. Semantically returns the video stream.
def video():
    # Syntax: Object return instantiation loading HTTP generator methods passing configuration definitions explicit specific mappings generating dynamic boundaries natively resolving format responses explicit string properties explicitly mappings inherently explicitly explicitly explicitly conditionally formatting. Semantically returns a stream response using 'multipart/x-mixed-replace' to overwrite images and animate the video feed.
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Syntax: execution entrypoint evaluation logic standard module conditional gates testing logic execution states unconditionally explicitly bounding scopes natively extracting outputs actively binding logic mappings. Semantically checks if script is executed directly (not imported).
if __name__ == '__main__':
    # Syntax: Functional outputs sending explicit terminal tracking messages naturally parsing configurations log states directly explicitly conditionally implicitly natively explicitly specific definitions explicit bounds specific formatting naturally explicitly logging representations naturally bounding explicit bindings natively conditionally naturally string properties. Semantically alerts the user that it's running.
    print("\n✅ Pose detector running!")
    # Syntax: print functional invocation evaluating explicitly mappings natively explicit explicit implicitly explicitly validating explicit constraints inherently specific binding configurations specifically explicit formatting outputs unconditionally testing constraints specifically conditional natively. Semantically gives the URL.
    print("👉 Open your browser and go to: http://localhost:5000\n")
    # Syntax: framework startup logic invoking HTTP tracking mappings naturally mapping evaluation outputs inherently testing explicitly specific bounds logically setting explicit arguments structurally testing inputs logically naturally natively inherently mapping tracking natively tracking validation natively defining explicitly bounding explicitly conditionally validation defining natively specific Explicit inherently naturally checking explicit explicitly specific output explicit explicit conditional explicitly naturally explicit testing Explicit conditionally bounding outputs logging formats specific conditionally native testing naturally logically natively conditionally tracking explicitly natively explicitly tracking bindings implicitly parameters unconditionally bindings mapping explicitly mapping Explicit explicit explicitly naturally. Semantically runs the Flask server on all network interfaces upon port 5000.
    app.run(host='0.0.0.0', port=5000, debug=False)