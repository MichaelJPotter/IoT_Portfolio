# ============================================================
# IMPORT NECESSARY LIBRARIES
# ============================================================

# Pandas for handling CSV data
import pandas as pd

# NumPy for numerical operations (arrays, math, calculations)
import numpy as np

# Glob for searching for files inside folders (pattern matching *.csv)
import glob

# OS for working with file paths and directories
import os

# sklearn for data splitting, scaling, and evaluation metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

# Seaborn for visualisation (heatmaps)
import seaborn as sns

# TensorFlow for deep learning and neural networks
import tensorflow as tf
from tensorflow.keras import layers, models

# Matplotlib for accuracy/loss plots
import matplotlib.pyplot as plt


# ============================================================
# FUNCTION — LOAD AND LABEL CSV FILES
# ============================================================

def load_labeled_csvs(basePath):
    """
    This function:
    - Looks inside the 'Car', 'Bus', and 'Train' folders
    - Loads all CSV files inside each folder
    - Adds a numeric 'label' column (0,1,2)
    - Combines them into one large dataset for training
    """

    data = []

    # Folder names mapped to labels (Neural Networks require numbers, not text)
    classMap = {
        "Bus": 0,
        "Car": 1,
        "Train": 2
    }

    # Loop through each folder
    for transportMode, label in classMap.items():

        # Build the folder path and find all CSV files inside it
        folder = os.path.join(basePath, transportMode, "*.csv")
        files = glob.glob(folder)

        # Load each CSV file and assign label
        for f in files:
            df = pd.read_csv(f)
            df["label"] = label
            data.append(df)
            print("Loaded:", f)

    # Combine every CSV into one DataFrame
    return pd.concat(data, ignore_index=True)


# ============================================================
# LOAD DATASET
# ============================================================

basePath = "IoT_Portfolio/MCJASM-Group-Data/MCJASM-Group-Data"
df = load_labeled_csvs(basePath)

print("Loaded dataset shape:", df.shape)


# ============================================================
# SELECT FEATURES FOR TRAINING
# ============================================================

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

# X = features, y = label column
X = df[feature_columns].fillna(0)
y = df["label"]


# ============================================================
# TRAIN/TEST SPLIT
# ============================================================

# Split into 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # keeps class balance even
)


# ============================================================
# FEATURE SCALING
# ============================================================

# Neural nets train better when features are scaled
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ============================================================
# BUILD THE NEURAL NETWORK MODEL
# ============================================================

model = models.Sequential([

    # First hidden layer: 32 neurons
    layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),

    # Second hidden layer
    layers.Dense(32, activation='relu'),

    # Output layer (3 outputs: Bus, Car, Train)
    layers.Dense(3, activation='softmax')
])

# Compile model: sparse loss = labels are integers (0/1/2)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


# ============================================================
# TRAIN THE MODEL (15 EPOCHS)
# ============================================================

history = model.fit(
    X_train,
    y_train,
    epochs=15,          # As requested
    batch_size=32,
    validation_split=0.2
)


# ============================================================
# PLOT TRAINING ACCURACY + LOSS
# ============================================================

plt.figure(figsize=(12, 5))

# Accuracy curve
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Model Accuracy Over Time")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Loss curve
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss Over Time")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()


# ============================================================
# CONFUSION MATRIX
# ============================================================

# Convert softmax probabilities → class numbers
y_pred = np.argmax(model.predict(X_test), axis=1)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Bus", "Car", "Train"],
    yticklabels=["Bus", "Car", "Train"]
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:\n")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Bus", "Car", "Train"]
))
