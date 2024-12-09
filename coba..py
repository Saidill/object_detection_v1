import tensorflow as tf

# Load the model using SavedModel format
model = tf.saved_model.load('yolov5s_tf')

# If you want to call the model to make predictions, you need to access the inference function.
infer = model.signatures['serving_default']

# Cek informasi tentang model
print(model)

import cv2
import numpy as np

# Fungsi untuk memuat gambar dan melakukan praproses
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (640, 640))  # Ukuran input model
    image = np.expand_dims(image, axis=0)  # Tambahkan batch dimension
    image = image / 255.0  # Normalisasi ke rentang [0, 1]
    return image

# Gambar input
image_path = '/Users/saidilhalim/Documents/object_detection_v1/image/siapa.jpg'

# Proses gambar
image = preprocess_image(image_path)

# Jalankan inferensi
input_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
output = infer(input_tensor)

# Tampilkan hasil prediksi
print(output)
