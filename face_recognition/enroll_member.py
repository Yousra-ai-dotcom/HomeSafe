import cv2
import os

def enroll_member(member_name, num_images=20):
    # Créer le dossier du membre
    member_path = os.path.join("dataset", member_name)
    os.makedirs(member_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    # Lire 10 images pour "réveiller" la caméra
    # Lecture de quelques frames pour permettre à la caméra de s'initialiser (évite les images noires ou mal exposées)
    for i in range(10):
        cap.read()

    if not cap.isOpened():
        print("❌ Erreur : impossible d'ouvrir la caméra.")
        return

    print(f"📸 Capture en cours pour : {member_name}")
    print("Appuie sur 'q' pour quitter.")

    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Afficher la vidéo
        cv2.imshow("Enregistrement du membre", frame)

        # Sauvegarder les images
        img_path = os.path.join(member_path, f"{member_name}_{count}.jpg")
        cv2.imwrite(img_path, frame)
        count += 1

        if count >= num_images:
            print(f"✔ {num_images} images enregistrées pour {member_name}")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    name = input("Entrez le nom du membre : ")
    enroll_member(name)