import cv2
import numpy as np
import os

def checkDataset(directory="dataset/"):
    if os.path.exists(directory) and len(os.listdir(directory)) != 0:
        return True
    return False

def organizeDataset(path="dataset/"):
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = []
    ids = []
    
    # Iterasi melalui setiap folder dalam dataset
    for person_folder in os.listdir(path):
        person_path = os.path.join(path, person_folder)
        if os.path.isdir(person_path):
            # Ambil ID dari folder (misalnya Person_1_Nama)
            person_id = int(person_folder.split('_')[1])
            
            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
                face = faceCascade.detectMultiScale(img)
                
                if len(face) == 0:
                    print(f"No face detected in file: {img_name}")
                    continue

                for (x, y, w, h) in face:
                    faces.append(img[y:y+h, x:x+w])
                    ids.append(person_id)

    return faces, np.array(ids)

if not checkDataset():
    print("Dataset not found")
else:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # Latih model pengenalan wajah
    print("Training faces...")
    faces, ids = organizeDataset()

    if len(faces) == 0 or len(ids) == 0:
        print("No faces found in the dataset for training.")
    else:
        recognizer.train(faces, ids)
        print("Training finished!")

        # Simpan model
        recognizer.write("face-model.yml")
        print("Model saved as 'face-model.yml'")