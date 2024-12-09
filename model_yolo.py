# Import libraries
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import google.generativeai as genai

# Konfigurasi Gemini API
genai.configure(api_key="API_KEY")

# Muat model YOLO dan klasifikasi
yolo_model = YOLO('yolov5s.pt') 
classification_model = load_model('my_model.h5') 

# Muat label dari file
with open('/Users/saidilhalim/Documents/object_detection_v1/labels.txt', 'r') as f:
    CLASS_LABELS = [line.strip() for line in f.readlines()]

# Fungsi deteksi objek
def detect_objects(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Gambar tidak ditemukan: {image_path}")
    
    results = yolo_model(image)
    bboxes = results[0].boxes.xyxy.cpu().numpy()  # xmin, ymin, xmax, ymax
    confidences = results[0].boxes.conf.cpu().numpy()  # Confidence scores
    return bboxes, confidences, image

# Fungsi klasifikasi objek
def classify_objects(image, bboxes):
    detected_labels = []
    accuracies = []
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = map(int, bbox[:4])
        cropped_image = image[ymin:ymax, xmin:xmax]
        if cropped_image.size == 0:
            continue
        
        resized_image = cv2.resize(cropped_image, (150, 150)) 
        resized_image = np.expand_dims(resized_image, axis=0) / 255.0 
        
        prediction = classification_model.predict(resized_image)
        class_index = np.argmax(prediction)
        detected_labels.append(CLASS_LABELS[class_index])
        accuracies.append(prediction[0][class_index])  
    return detected_labels, accuracies

# Fungsi untuk menghasilkan teks menggunakan Gemini API
def generate_sentence_gemini(detected_labels):
    if not detected_labels:
        return "Tidak ada handsign yang terdeteksi."
    
    prompt = " ".join(detected_labels)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    try:
        response = model.generate_content(f"Buat kalimat berdasarkan kata-kata berikut: {prompt}")
        return response.text  # Ambil teks respons
    except Exception as e:
        return f"Error saat menghasilkan teks: {e}"

# Fungsi untuk memvisualisasikan hasil
def visualize_results(image_path, bboxes, detected_labels, accuracies):
    image = cv2.imread(image_path)
    for bbox, label, accuracy in zip(bboxes, detected_labels, accuracies):
        xmin, ymin, xmax, ymax = map(int, bbox[:4])
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        cv2.putText(image, f"{label} ({accuracy:.2f})", (xmin, ymin - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    cv2.imshow('Detection and Classification', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    image_paths = [ 
        '/Users/saidilhalim/Documents/object_detection_v1/image/halo.jpg',
        '/Users/saidilhalim/Documents/object_detection_v1/image/siapa.jpg'
    ]
    
    try:
        all_detected_labels = []
        for image_path in image_paths:
            # Deteksi dan klasifikasi
            bboxes, confidences, image = detect_objects(image_path)
            detected_labels, accuracies = classify_objects(image, bboxes)
            
            # Hasilkan teks menggunakan Gemini
            generated_text = generate_sentence_gemini(detected_labels)
            
            # Tampilkan hasil
            print(f"Hasil untuk {image_path}:")
            print("Deteksi:", detected_labels)
            print("Akurasi:", accuracies)
            print("Teks yang Dihasilkan:", generated_text)
            
            # Tambahkan hasil ke daftar semua label
            all_detected_labels.extend(detected_labels)
            
            # Visualisasi hasil
            visualize_results(image_path, bboxes, detected_labels, accuracies)
        
        # Gabungkan semua label untuk teks akhir
        final_text = generate_sentence_gemini(all_detected_labels)
        print("\nTeks Gabungan untuk Semua Gambar:", final_text)
    
    except Exception as e:
        print(f"Error: {e}")
