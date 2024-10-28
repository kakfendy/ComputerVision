import cv2
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face-model.yml")  # Memuat model yang telah dilatih
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
font = cv2.FONT_HERSHEY_COMPLEX

# Membuat daftar nama dari folder dataset
names = ['None']  # Mulai dengan nama None untuk ID 0
dataset_path = "dataset/"
for person_folder in os.listdir(dataset_path):
    if os.path.isdir(os.path.join(dataset_path, person_folder)):
        names.append(person_folder.split('_')[2])  # Ambil nama dari folder (misalnya Person_1_Nama)

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        if confidence < 100:
            name = names[id]  # Mengambil nama berdasarkan ID
            confidence_text = f"{round(100 - confidence)}%"
        else:
            name = "unknown"
            confidence_text = f"{round(100 - confidence)}%"

        cv2.putText(frame, str(name), (x + 5, y - 5), font, 1, (255, 0, 0), 1)
        cv2.putText(frame, str(confidence_text), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()