import cv2
from PIL import Image
import numpy as np
import joblib
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch

class RealTimeRecognizer:
    def __init__(self, device='cpu'):
        self.device = device

        # Détecteur de visage
        self.mtcnn = MTCNN(image_size=160, margin=20, device=device)

        # Modèle FaceNet
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

        # Charger classifieur & label encoder
        self.classifier = joblib.load("models/classifier.pkl")
        self.label_encoder = joblib.load("models/label_encoder.pkl")

    def get_embedding(self, frame):
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        face = self.mtcnn(img)
        if face is None:
            return None
        face = face.unsqueeze(0).to(self.device)
        return self.facenet(face).detach().cpu().numpy().flatten()

    def recognize(self):
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("❌ Impossible d'ouvrir la caméra.")
            return

        print("🎥 Reconnaissance en temps réel démarrée. Appuie sur 'q' pour quitter.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            embedding = self.get_embedding(frame)

            confidence_threshold = 0.85


            if embedding is not None:
                proba = self.classifier.predict_proba([embedding])[0]
                pred = np.argmax(proba)
                confidence = proba[pred]

                if confidence < confidence_threshold:
                    name = "UNKNOWN"
                else:
                    name = self.label_encoder.inverse_transform([pred])[0]
                # Affichage
                cv2.putText(frame, f"{name} ({confidence*100:.1f}%)", 
                            (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 255, 0) if name != "UNKNOWN" else (0, 0, 255), 2)

            else:
                cv2.putText(frame, "No Face Detected", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Reconnaissance Faciale - HomeSafe", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    recognizer = RealTimeRecognizer(device='cpu')
    recognizer.recognize()