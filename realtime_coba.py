from ultralytics import YOLO
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import google.generativeai as genai
import mediapipe as mp 

# Konfigurasi API Gemini
genai.configure(api_key="API_KEY")

# Muat YOLO pretrained model
yolo_model = YOLO('yolov5s.pt') 

# Muat model klasifikasi Anda
classification_model = load_model('my_model.h5')

# Baca label dari file label.txt
with open('/Users/saidilhalim/Documents/object_detection_v1/labels.txt', 'r') as f:
    CLASS_LABELS = [line.strip() for line in f.readlines()]

# Setup MediaPipe Hand Detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

def generate_text_from_labels(detected_labels):
    """Fungsi untuk menghasilkan teks berdasarkan label yang terdeteksi"""
    if not detected_labels:
        return "Tidak ada objek yang terdeteksi."
    
    prompt = " ".join(detected_labels)  # Gabungkan label menjadi satu kalimat
    
    try:
        # Gunakan model generative untuk menghasilkan kalimat
        model = genai.GenerativeModel('gemini-1.5-flash')  # Gunakan model Gemini (atau model lain)
        response = model.generate_content(f"Buat kalimat berdasarkan kata-kata berikut: {prompt}")
        return response.text  # Mengembalikan teks hasil generasi
    except Exception as e:
        return f"Error dalam menghasilkan teks: {e}"

def classify_objects(image, bboxes):
    object_classes = []
    detected_labels = []  # Untuk menyimpan label yang terdeteksi
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = map(int, bbox[:4])  # Koordinat bounding box
        cropped_image = image[ymin:ymax, xmin:xmax]  # Potong gambar
        
        if cropped_image.size == 0:
            continue
        
        # Resize gambar agar sesuai dengan input model klasifikasi
        resized_image = cv2.resize(cropped_image, (150, 150))  # Sesuaikan ukuran input model Anda
        resized_image = np.expand_dims(resized_image, axis=0) / 255.0  # Normalisasi
        
        # Klasifikasi menggunakan model
        prediction = classification_model.predict(resized_image)
        class_index = np.argmax(prediction)  # Indeks kelas dengan probabilitas tertinggi
        confidence = prediction[0][class_index]  # Confidence score
        class_label = CLASS_LABELS[class_index]  # Nama label berdasarkan indeks
        
        object_classes.append((class_label, confidence, bbox))  # Simpan label, confidence, dan bounding box
        detected_labels.append(class_label)  # Simpan label untuk text generation
    
    return object_classes, detected_labels  # Kembalikan juga label yang terdeteksi

def detect_hands(frame):
    """Deteksi tangan menggunakan MediaPipe"""
    # Convert frame ke RGB karena MediaPipe membutuhkan input dalam format RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    
    hand_bboxes = []
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Mendapatkan bounding box dari landmark tangan
            min_x = min([landmark.x for landmark in hand_landmarks.landmark])
            min_y = min([landmark.y for landmark in hand_landmarks.landmark])
            max_x = max([landmark.x for landmark in hand_landmarks.landmark])
            max_y = max([landmark.y for landmark in hand_landmarks.landmark])
            
            # Normalisasi koordinat
            height, width, _ = frame.shape
            min_x = int(min_x * width)
            min_y = int(min_y * height)
            max_x = int(max_x * width)
            max_y = int(max_y * height)
            
            hand_bboxes.append([min_x, min_y, max_x, max_y])
    
    return hand_bboxes

def process_frame(frame):
    # Deteksi objek menggunakan YOLO
    results = yolo_model(frame)
    detections = results[0]  # Hasil deteksi
    
    # Filter hanya untuk label "person"
    person_detections = [
        det for det in detections if det.boxes.cls.cpu().numpy()[0] == 0
    ]  # Label "person" memiliki indeks 0 dalam COCO dataset
    
    # Ekstrak bounding box untuk deteksi objek
    bboxes = [det.boxes.xyxy.cpu().numpy()[0] for det in person_detections]
    
    # Deteksi tangan
    hand_bboxes = detect_hands(frame)
    
    # Klasifikasi objek dalam bounding box
    object_classes, detected_labels = classify_objects(frame, bboxes)
    
    # Hasilkan teks berdasarkan label yang terdeteksi
    generated_text = generate_text_from_labels(detected_labels)
    print("Generated Text:", generated_text)  # Tampilkan teks yang dihasilkan
    
    # Visualisasi hasil
    for obj_class, confidence, bbox in object_classes:
        xmin, ymin, xmax, ymax = map(int, bbox[:4])
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)  # Bounding box merah
        label_text = f"{obj_class} ({confidence:.2f})"  # Format teks dengan confidence
        cv2.putText(frame, label_text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Visualisasi deteksi tangan
    for bbox in hand_bboxes:
        xmin, ymin, xmax, ymax = map(int, bbox)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # Bounding box hijau untuk tangan
    
    return frame

def main():
    # Inisialisasi kamera
    cap = cv2.VideoCapture(0)  # Gunakan indeks kamera 0 (ubah jika menggunakan kamera eksternal)
    
    if not cap.isOpened():
        print("Tidak dapat mengakses kamera.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame dari kamera.")
            break
        
        # Proses frame untuk deteksi dan klasifikasi
        processed_frame = process_frame(frame)
        
        # Tampilkan frame yang telah diproses
        cv2.imshow('Realtime Detection and Classification', processed_frame)
        
        # Tekan 'q' untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Lepaskan kamera dan tutup semua jendela
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
