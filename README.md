# CV Workshop Starter

Run your Teachable Machine model locally with real Python code and your webcam.

## Setup

1. Go back to [Teachable Machine](https://teachablemachine.withgoogle.com/train/image) with your trained model.
2. Click **Export Model** -> **Tensorflow** tab -> select **Keras** -> **Download my model**.
3. Unzip the download. You'll get two files: `keras_Model.h5` and `labels.txt`.
4. Drop both files into this folder, next to `inference.py`.
5. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
6. Run it:
   ```
   python inference.py
   ```
7. Press `ESC` in the window to quit.

That's it. No code editing required for the base version.

## Don't commit your model or labels

`keras_Model.h5` and `labels.txt` are in `.gitignore` on purpose. If your model was trained on your own face or a friend's, that's personal data. Don't push it to a public repo.

## Troubleshooting

- **`ModuleNotFoundError` for tensorflow/keras**: re-run `pip install -r requirements.txt`, and make sure you're using Python 3.9-3.11. Newer/older versions can have TensorFlow compatibility issues.
- **Webcam doesn't open**: close any other app using your camera (Zoom, Teams, another browser tab), then re-run.
- **Predictions look wrong / low confidence**: this usually means your training photos didn't have enough variety in lighting or angle. That's not a bug, it's the same overfitting problem covered in the workshop.

## Take it further

Once the base script works, here are a few directions to extend it:

- **Log predictions to a CSV** with a timestamp on every frame, so you can graph confidence over time.
- **Add pose estimation** with [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) alongside your classifier, and combine both outputs.
- **Wrap it in a Flask app** with a `/predict` route that takes an uploaded image and returns the prediction as JSON, so it's a real deployable API instead of a local script.
- **Swap the webcam loop for a folder of test images** and report accuracy across a whole batch instead of one live feed.

Pick one, build it, and you've got a real project for your resume or GitHub, not just a workshop screenshot.
