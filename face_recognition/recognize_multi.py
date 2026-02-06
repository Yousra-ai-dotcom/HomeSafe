# face_recognition/recognize_multi.py
import cv2
from PIL import Image
import numpy as np
import joblib
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch


# Cette classe gère la reconnaissance faciale de plusieurs personnes
# simultanément dans une même image vidéo
class MultiRecognizer:
    def __init__(self, device='cpu', min_confidence=0.6, threshold=0.85):
        self.device = device
        self.mtcnn = MTCNN(image_size=160, margin=20, min_face_size=20, device=device)
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device) # CNN FaceNet pré-entraîné pour transformer un visage en embedding
        self.classifier = joblib.load("models/classifier.pkl")
        self.label_encoder = joblib.load("models/label_encoder.pkl")
        self.min_confidence = min_confidence
        self.threshold = threshold  # seuil pour considérer confiance -> membre

    def recognize(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Impossible d'ouvrir la caméra.")
            return

        print("🎥 Reconnaissance multi-visages démarrée. Appuie sur 'q' pour quitter.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # get boxes + probs
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            boxes, probs = self.mtcnn.detect(img)

            if boxes is not None and probs is not None:
                for (box, p) in zip(boxes, probs): # Traitement individuel de chaque visage détecté
                    if p is None or p < self.min_confidence:
                        continue
                    x1, y1, x2, y2 = map(int, box)

                    # extraire le visage aligné en utilisant mtcnn (keep_all=False but crop single face)
                    # Ici on prend crop direct via PIL to keep it simple
                    face_crop = img.crop((x1, y1, x2, y2)).resize((160, 160))
                    face_tensor = torch.tensor(np.array(face_crop)).permute(2,0,1).float() / 255.0
                    # MTCNN usually performs whitening/normalization; to be safe, let facenet accept the tensor:
                    face_tensor = face_tensor.unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        embedding = self.facenet(face_tensor).cpu().numpy().flatten()

                    proba = self.classifier.predict_proba([embedding])[0]
                    pred = np.argmax(proba)
                    confidence = proba[pred]
                    # Prédiction de l’identité
                    # confidence = degré de certitude



                    if confidence < self.threshold:
                        name = "UNKNOWN"
                        color = (0, 0, 255)
                    else:
                        name = self.label_encoder.inverse_transform([pred])[0]
                        color = (0, 255, 0)

                    # dessiner
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{name} {confidence*100:.0f}%", (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                cv2.putText(frame, "No Face", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Reconnaissance Multi - HomeSafe", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    recognizer = MultiRecognizer(device='cpu', min_confidence=0.6, threshold=0.85)
    recognizer.recognize()