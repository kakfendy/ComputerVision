import cv2
import os

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)
dataset_path = "dataset/"

if not os.path.exists(dataset_path):
    os.mkdir(dataset_path)

# Tambahkan input untuk ID orang
person_id = input("Enter person ID: ")  # Mengambil input ID dari pengguna
person_name = input("Enter person name: ")  # Nama juga bisa diambil untuk pengorganisasian

# Membuat folder untuk orang tertentu berdasarkan ID
person_path = os.path.join(dataset_path, f"Person_{person_id}_{person_name}")
if not os.path.exists(person_path):
    os.mkdir(person_path)

count = 0  # Menghitung jumlah gambar per orang

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1
        # Simpan gambar di folder sesuai ID orang
        cv2.imwrite(os.path.join(person_path, f"Person_{person_id}_{count}.jpg"), gray[y:y + h, x:x + w])

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) == ord('q'):
        break
    elif count == 30:  # Berhenti setelah 30 foto diambil
        break

cap.release()
cv2.destroyAllWindows()