import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import layers, models

print("1. Data load ho raha hai...")

# Pandas se CSV files padhna
train_df = pd.read_csv('mitbih_train.csv', header=None)
test_df = pd.read_csv('mitbih_test.csv', header=None)

print(f"Train Data Shape: {train_df.shape}")
print(f"Test Data Shape: {test_df.shape}")

# Last column (187th index) Target Label hai
X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values

X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values

# Multi-class ko Binary (0: Normal, 1: Abnormal) me convert karna
y_train = (y_train > 0).astype(int)
y_test = (y_test > 0).astype(int)

# Check Class Distribution
normal_count = np.sum(y_train == 0)
abnormal_count = np.sum(y_train == 1)

print(f"Normal Samples (0): {normal_count}")
print(f"Abnormal Samples (1): {abnormal_count}")

# 1. Class Weight calculate karna (Abnormal signal ko importance dene ke liye)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"\nClass Weights calculated: {class_weight_dict}")

# 2. Reshape Input for 1D-CNN (Samples, Steps, Channels)
X_train_cnn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_cnn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# 3. Build Lightweight 1D-CNN Model Architecture
model = models.Sequential([
    layers.Conv1D(filters=16, kernel_size=5, activation='relu', input_shape=(187, 1)),
    layers.MaxPooling1D(pool_size=2),
    layers.Conv1D(filters=32, kernel_size=5, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid') # Binary Output
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Recall(name='recall')])

print("\n2. Model Structure Ready! Training shuru ho rahi hai...")

# 4. Train Model with Class Weights
history = model.fit(
    X_train_cnn, y_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_test_cnn, y_test),
    class_weight=class_weight_dict
)
# ==========================================
# 5. Model Conversion to TFLite & Quantization
# ==========================================
print("\n3. Model ko Int8 Quantize karke TFLite me badla ja raha hai...")

# Representative Dataset Generator (Quantization ke liye zaruri)
def representative_data_gen():
    for input_value in tf.data.Dataset.from_tensor_slices(X_train_cnn).batch(1).take(100):
        yield [tf.cast(input_value, tf.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model_quant = converter.convert()

# TFLite model file save karein
with open('ecg_model_quant.tflite', 'wb') as f:
    f.write(tflite_model_quant)

print("TFLite Quantized Model saved: ecg_model_quant.tflite")

# ==========================================
# 6. Convert TFLite to C-Header (.h) File
# ==========================================
print("\n4. C-Header file (model_data.h) generate ho rahi hai...")

hex_lines = [f'0x{b:02x}' for b in tflite_model_quant]
c_array = ", ".join(hex_lines)

header_content = f"""#ifndef MODEL_DATA_H
#define MODEL_DATA_H

const unsigned char g_model[] = {{
  {c_array}
}};
const unsigned int g_model_len = {len(tflite_model_quant)};

#endif // MODEL_DATA_H
"""

with open('model_data.h', 'w') as f:
    f.write(header_content)

print("Success! 'model_data.h' file successfully created!")
# ==========================================
# ==========================================
# ==========================================
# 7. Test Quantized TFLite Model on Full Dataset (Shuffled)
# ==========================================
print("\n5. Quantized TFLite Model ko Shuffled Test Data par verify kiya ja raha hai...")

interpreter = tf.lite.Interpreter(model_path="ecg_model_quant.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Shuffled indices taaki Normal aur Abnormal dono mix milein
np.random.seed(42)
test_indices = np.random.choice(len(X_test_cnn), size=2000, replace=False)

y_pred_tflite = []
y_true_tflite = []

input_scale, input_zero_point = input_details[0]['quantization']
output_scale, output_zero_point = output_details[0]['quantization']

for idx in test_indices:
    sample = X_test_cnn[idx:idx+1]
    
    # Quantize input signal float -> int8
    sample_quant = np.round(sample / input_scale + input_zero_point).astype(np.int8)
    
    interpreter.set_tensor(input_details[0]['index'], sample_quant)
    interpreter.invoke()
    
    output = interpreter.get_tensor(output_details[0]['index'])
    
    # Dequantize output int8 -> float score
    raw_val = int(output[0][0])
    score = (raw_val - output_zero_point) * output_scale
    
    # Threshold 0.40
    pred_label = 1 if score > 0.40 else 0
    
    y_pred_tflite.append(pred_label)
    y_true_tflite.append(y_test[idx])

from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

acc = accuracy_score(y_true_tflite, y_pred_tflite)
rec = recall_score(y_true_tflite, y_pred_tflite)
cm = confusion_matrix(y_true_tflite, y_pred_tflite)

print(f"\n--- TFLite Quantized Model Verification Result ---")
print(f"Quantized Model Accuracy: {acc * 100:.2f}%")
print(f"Quantized Model Recall  : {rec * 100:.2f}%")
print(f"Confusion Matrix:\n{cm}")
print("--------------------------------------------------")