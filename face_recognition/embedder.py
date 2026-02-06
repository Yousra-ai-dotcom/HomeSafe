import os
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1 # InceptionResnetV1 : CNN FaceNet pré-entraîné pour l'extraction d'embeddings
import torch
import cv2

class FaceEmbedder:
    # Cette classe encapsule toute la logique d'extraction des embeddings faciaux /Elle est utilisée aussi bien pour l'entraînement que pour la reconnaissance en temps réel
    def __init__(self, device='cpu'):
        self.device = device
        
        # Détecteur de visages
        self.mtcnn = MTCNN(image_size=160, margin=20, min_face_size=40, device=device)
        
        # Modèle FaceNet pré-entraîné # Chargement du modèle FaceNet pré-entraîné sur VGGFace2 / eval() : mode évaluation (pas d'entraînement)
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    def get_embedding(self, image_path): #pour générer les embeddings du dataset
        img = Image.open(image_path)
        face = self.mtcnn(img)
        if face is None:
            print(f"❌ Aucun visage détecté dans {image_path}")
            return None
        
        face = face.unsqueeze(0).to(self.device) # ajout d'une dimension batch et envoi vers le device
        embedding = self.model(face).detach().cpu().numpy().flatten() # passage du visage dans le CNN FaceNet

        return embedding

    def get_embedding_from_frame(self, face_bgr):
        """
        Utilisé pour le temps réel (Streamlit / webcam).
        Prend une image OpenCV (numpy array BGR) et retourne l'embedding FaceNet.
        """
        # Conversion BGR (OpenCV) -> RGB
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(face_rgb)

        # Détection du visage avec MTCNN
        try:
            face = self.mtcnn(img)
        except RuntimeError:
            # Handle case where MTCNN fails (e.g., empty tensor list)
            return None
        if face is None:
            return None

        face = face.unsqueeze(0).to(self.device)

        #extraction de l'embedding sans calcul de gradient (mode inférence)
        with torch.no_grad():
            embedding = self.model(face)

        return embedding.detach().cpu().numpy().flatten()

    def process_dataset(self, dataset_path="dataset"):
        embeddings = []
        labels = []

        print("🔍 Génération des embeddings...")

        for person_name in os.listdir(dataset_path):
            person_folder = os.path.join(dataset_path, person_name)

            if not os.path.isdir(person_folder):
                continue

            for img_name in os.listdir(person_folder):
                img_path = os.path.join(person_folder, img_name)
                embedding = self.get_embedding(img_path) # Extraction de l'embedding via FaceNet

                if embedding is not None:
                    embeddings.append(embedding)
                    labels.append(person_name)

        embeddings = np.array(embeddings)
        labels = np.array(labels)

        os.makedirs("embeddings", exist_ok=True)
        np.save("embeddings/members_embeddings.npy", embeddings)
        np.save("embeddings/labels.npy", labels)

        print("✔ Embeddings générés et sauvegardés !")
        print(f"Total images utilisées : {len(labels)}")

        return embeddings, labels


if __name__ == "__main__":
    embedder = FaceEmbedder(device='cpu')
    embedder.process_dataset()