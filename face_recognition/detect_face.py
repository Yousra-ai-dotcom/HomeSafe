# face_recognition/detect_face.py
"""
Ce fichier sert à détecter les visages dans une image ou une frame vidéo.
Il NE reconnaît PAS l identité (ce n est pas encore du CNN de classification).
Il sert à localiser les visages avant de :
	•	extraire les embeddings (FaceNet)
	•	ou afficher des rectangles à l écran

 C est la première étape du pipeline de reconnaissance faciale.
"""

import cv2
from PIL import Image #utilisé car MTCNN attend des images PIL
import numpy as np
from facenet_pytorch import MTCNN # modèle CNN pré-entraîné pour la détection de visages (où est le visage (boîte),avec quelle probabilité c’est vraiment un visage)

# Cette classe encapsule toute la logique de détection de visages
# Elle sera utilisée dans d'autres fichiers (Streamlit, reconnaissance, etc.)

class FaceDetector:
    def __init__(self, device='cpu', min_confidence=0.5): # device : permet de choisir CPU ou GPU / min_confidence : seuil minimal pour accepter un visage détecté
        self.device = device
        self.mtcnn = MTCNN(
            image_size=160,
            margin=20, #  marge autour du visage (évite les crops trop serrés)
            min_face_size=20, # taille minimale du visage à détecter
            device=device
        )
        self.min_confidence = min_confidence

    def detect_faces(self, frame):
        """
        Retourne toutes les boîtes et probabilités au-dessus du seuil.
        boxes: ndarray (N,4) ou None
        probs: ndarray (N,) ou None
        """
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        try:
            boxes, probs = self.mtcnn.detect(img)
        except RuntimeError:
            # Handle case where MTCNN fails (e.g., empty tensor list)
            # Pour éviter un crash du programme, on intercepte l'erreur et on retourne simplement une liste vide
            return [], []

        if boxes is None or probs is None:
            return [], []

        # filtrer par probabilité
        kept = [(b, p) for b, p in zip(boxes, probs) if p is not None and p >= self.min_confidence]
        # On filtre les visages détectés en fonction du seuil de confiance  Cela permet d'éliminer les faux positifs
        if not kept:
            return [], []

        boxes_kept, probs_kept = zip(*kept)
        return np.array(boxes_kept), np.array(probs_kept)


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    detector = FaceDetector(min_confidence=0.6)

    print("🎥 Test detect_face.py — Appuie sur 'q' pour quitter")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes, probs = detector.detect_faces(frame)

        if len(boxes) > 0:
            for (x1, y1, x2, y2), p in zip(boxes, probs):
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"{p:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        else:
            cv2.putText(frame, "No Face", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Face Detector Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()