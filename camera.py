import cv2
from flask import Response
import config
import cv2

camera = cv2.VideoCapture(config.CAMERA_INDEX)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

# Ouvrir la webcam (0 = webcam par défaut)
camera = cv2.VideoCapture(0)

def generate_frames():
    """
    Générateur de frames pour le streaming vidéo Flask
    """
    while True:
        success, frame = camera.read()
        if not success:
            break

        # Encode l’image en JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Streaming MJPEG
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def video_feed():
    """
    Route Flask utilisée dans app.py
    """
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )