import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder # LabelEncoder : conversion des labels texte en valeurs numériques
from sklearn.model_selection import train_test_split
import joblib # sauvegarde et chargement des modèles
import os

def train_classifier():
    print("📥 Chargement des embeddings...")
    embeddings = np.load("embeddings/members_embeddings.npy") #Les embeddings représentent les caractéristiques faciales extraites par le CNN
    labels = np.load("embeddings/labels.npy")

    print(f"Total embeddings: {len(labels)}")

    print("🔤 Encodage des labels...") 
    # Conversion des noms (strings) en classes numériques
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    print("📊 Division train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels_encoded, test_size=0.2, random_state=42
    )

    print("🤖 Entraînement du classifieur SVM...")#est bien adapté aux embeddings de FaceNet car ils sont déjà séparables dans l’espace des caractéristiques. 
    # Création d'un SVM linéaire
    # probability=True permet d'obtenir une probabilité de confiance
    # Entraînement du classifieur sur les embeddings
    classifier = SVC(kernel='linear', probability=True)
    classifier.fit(X_train, y_train)

    print("📈 Évaluation du modèle...")
    accuracy = classifier.score(X_test, y_test)
    print(f"✔ Accuracy du classifieur : {accuracy * 100:.2f}%")

    print("💾 Sauvegarde du modèle...")
    os.makedirs("models", exist_ok=True)
    joblib.dump(classifier, "models/classifier.pkl")
    joblib.dump(label_encoder, "models/label_encoder.pkl")

    print("🎉 Classifieur entraîné et sauvegardé !")

if __name__ == "__main__":
    train_classifier()