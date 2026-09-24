"""
Run your Teachable Machine model locally with your webcam.

Setup:
  1. Export your model from Teachable Machine: Export Model -> Tensorflow -> Keras -> Download my model
  2. Unzip it. You'll get two files: keras_Model.h5 and labels.txt
  3. Drop both files into this same folder, next to this script
  4. pip install -r requirements.txt
  5. python inference.py

Press ESC to quit.
"""

import os
import sys

import cv2
import numpy as np
from keras.models import load_model

MODEL_PATH = "keras_Model.h5"
LABELS_PATH = "labels.txt"


def load_trained_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
        print(f"Missing {MODEL_PATH} or {LABELS_PATH}.")
        print("Export your model from Teachable Machine (Tensorflow -> Keras)")
        print("and drop both files into this folder before running again.")
        sys.exit(1)

    model = load_model(MODEL_PATH, compile=False)
    with open(LABELS_PATH, "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    return model, class_names


def preprocess(frame):
    resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    array = np.asarray(resized, dtype=np.float32).reshape(1, 224, 224, 3)
    normalized = (array / 127.5) - 1
    return normalized


def main():
    np.set_printoptions(suppress=True)
    model, class_names = load_trained_model()

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Could not open webcam. Is another app using it?")
        sys.exit(1)

    print("Running. Press ESC in the window to quit.")

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        prediction = model.predict(preprocess(frame), verbose=0)
        index = int(np.argmax(prediction))
        label = class_names[index].split(" ", 1)[-1]  # strip the leading index number
        confidence = float(prediction[0][index]) * 100

        overlay_text = f"{label}  ({confidence:.1f}%)"
        cv2.putText(
            frame, overlay_text, (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA,
        )
        cv2.imshow("Your Model - press ESC to quit", frame)

        if cv2.waitKey(1) == 27:  # ESC
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
