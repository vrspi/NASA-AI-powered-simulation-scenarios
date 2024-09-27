# Import necessary libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Load and preprocess data
data = pd.read_csv('neo_data.csv', sep=',')
print("Columns in data:")
print(data.columns.tolist())

# Check for missing values
print("Missing values per column:")
print(data.isnull().sum())

# List of numerical features to be used
numerical_features = [
    'H', 'G', 'diameter', 'albedo', 'rot_per', 'e', 'a', 'q',
    'i', 'om', 'w', 'ma', 'n', 'per', 'moid'
]

# Ensure all features exist in the data
missing_features = [f for f in numerical_features if f not in data.columns]
if missing_features:
    print(f"Missing numerical features: {missing_features}")
    # Remove missing features from the list
    numerical_features = [f for f in numerical_features if f in data.columns]

# Fill missing numerical values with the mean
data[numerical_features] = data[numerical_features].fillna(data[numerical_features].mean())

# Target variable
if 'class' in data.columns:
    target = 'class'
else:
    raise ValueError("Target variable 'class' not found in data.")

# Drop rows with missing target values
data = data.dropna(subset=[target])

# Encode the target variable
le_class = LabelEncoder()
data['class_encoded'] = le_class.fit_transform(data[target])

# Initialize the scaler
scaler = StandardScaler()

# Scale numerical features
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Prepare features and target
X = data[numerical_features]
y = data['class_encoded']

# Number of classes
num_classes = len(le_class.classes_)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Convert labels to categorical format
y_train_categorical = to_categorical(y_train, num_classes)
y_test_categorical = to_categorical(y_test, num_classes)

# Define the model architecture
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Model summary
model.summary()

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_classification_model.keras', save_best_only=True)

# Fit the model
history = model.fit(
    X_train, y_train_categorical,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping, model_checkpoint]
)

# Load the best model
model.load_weights('best_classification_model.keras')

# Evaluate on test data
test_loss, test_accuracy = model.evaluate(X_test, y_test_categorical)
print(f'Test Accuracy: {test_accuracy:.2f}')

# Predict classes
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

# Classification report
print(classification_report(y_test, y_pred, target_names=le_class.classes_))

# Plot accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Convert and save model for TensorFlow.js
import tensorflowjs as tfjs
tfjs.converters.save_keras_model(model, 'NEOsClassification_Model')

# Zip the model directory
import subprocess
subprocess.run(['zip', '-r', 'NEOsClassification_Model.zip', 'NEOsClassification_Model'], check=True)

# Save scaler parameters and class labels
import json

scaler_params = {
    'mean': scaler.mean_.tolist(),
    'scale': scaler.scale_.tolist()
}
with open('scaler_params.json', 'w') as f:
    json.dump(scaler_params, f)

class_labels = le_class.classes_.tolist()
with open('class_labels.json', 'w') as f:
    json.dump(class_labels, f)