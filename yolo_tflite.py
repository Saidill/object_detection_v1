import tensorflow as tf

# Tentukan path ke model SavedModel Anda
saved_model_path = 'yolov5s_tf'  # Ganti dengan path model Anda

# Mengonversi model ke format TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)

# Pilih optimasi (opsional)
# Misalnya, menggunakan optimasi untuk ukuran model lebih kecil
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Lakukan konversi
tflite_model = converter.convert()

# Simpan model TFLite ke file
with open('yolov5s_model.tflite', 'wb') as f:
    f.write(tflite_model)
