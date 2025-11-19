#To compile this code the interpreter version should be python3 3.10.11 as this is required to import the tenser flow
#All libraries will need to be installed using -python3 pip install
#import necessary Libraries 
#------------------------
#Pandas for handling CSV data 
import pandas as pd
#numpy for xxx (Fill this)
import numpy as np
# for xxx (Fill this)
import glob
# for xxx (Fill this)
import os
#matplot lib for plotting 
import matplotlib.pyplot as plt
#sklearn for data processing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
#seaborn for data visualisation
import seaborn as sns
#tensorflow for deeplearning
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers, models 


def load_labeled_csvs(basePath):
    data = []
    # Folders: Looks inside the 'car', 'bus', and 'train' folders, assings numerical label as neural networks cant process text
    classMap = {
        "bus": 0,
        "car": 1,
        "train": 2
    }

    for transportMode, label in classMap.items():
        folder = os.path.join(basePath, transportMode, "*.csv")
        files = glob.glob(folder)

        # Debug print, we had issues reading the files
        print(f"Searching in: {folder}  |  Found: {len(files)} files")

        for f in files:
            df = pd.read_csv(f)
            df["label"] = label
            data.append(df)

    if not data: # Debug print, we had issues reading the files

        raise ValueError("No CSV files found. Check basePath and folder names (bus/car/train).")
    return pd.concat(data, ignore_index=True)

# Defining paths to datasets
basePath = r"MCJASM-Group-Data\MCJASM-Group-Data"

df = load_labeled_csvs(basePath)
print("Loaded dataset shape:", df.shape)

#Tensorflow only works on Python 3.10
feature_columns = [
    "longitude",
    "latitude",
    "altitude",
    "horizontal accuracy",
    "vertical accuracy",
    "speed",
    "speed accuracy",
    "course",
    "course accuracy"
]

# X = input data
# y = labels (bus/car/train)
X = df[feature_columns].fillna(0) 
y = df["label"]

# Train and test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Feature scaling
# Neural networks work best when features are standardised
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Tenserflow model build
model = models.Sequential([
    layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    # Output layer: 3 classes (bus, car, train)
    layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  #Data labels are integers (0/1/2)
    metrics=['accuracy']
)
model.summary()

# Train the model (15 cycles)
history = model.fit(
    X_train,
    y_train,
    epochs=15,
    batch_size=32,
    validation_split=0.2
)

# Plot the training accuracy and loss values
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label="Train Accuracy")
plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.show()

# Confusion Matrix

y_pred = np.argmax(model.predict(X_test), axis=1)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Bus", "Car", "Train"],
    yticklabels=["Bus", "Car", "Train"]
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

print("\nClassification Report:\n")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Bus", "Car", "Train"]
))
print("\nAll tasks completed successfully!")