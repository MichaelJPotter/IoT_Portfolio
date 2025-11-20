#To compile this code the interpreter version should be python3 3.10.11 as this is required to import the tenser flow
#All libraries will need to be installed using -python3 pip install

#--- Importing libraries --- 
#Import necessary Libraries 
#For handling CSV data 
import pandas as pd
import numpy as np
#For file manipulation
import glob
import os
#For data processing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
#For data visualisation
import seaborn as sns
import matplotlib.pyplot as plt
#For deep learning 
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers, models 

#--- Loading CSV data ---
def load_labeled_csvs(basePath):
    #The CSV's are stored in 3 folders, each with an associated transport type 
    #Each transport type is assinged numerical label as neural networks cant process text (0:Bus, 1:Car, 2:Train) 
    data = []
    classMap = {
        "bus": 0,
        "car": 1,
        "train": 2
    }

    #Finds the CSV files from the folders
    for transportMode, label in classMap.items():
        #Finds files in provided path ending in ".csv"
        folder = os.path.join(basePath, transportMode, "*.csv")
        files = glob.glob(folder)
        #Appends the CSV data to the dataframe 
        for f in files:
            df = pd.read_csv(f)
            df["label"] = label
            data.append(df)
    
    #Exception handling 
    if not data: 
        raise ValueError("No CSV files found")

    return pd.concat(data, ignore_index=True)

#Defining path to datasets
basePath = r"MCJASM-Group-Data\MCJASM-Group-Data"

#Loading CSV data into a data-frame
df = load_labeled_csvs(basePath)

#--- Training/Testing ---
#defining features of the dataset
feature_columns = ["longitude","latitude","altitude","horizontal accuracy","vertical accuracy","speed","speed accuracy","course","course accuracy"]

# X is the data of the dataset and all NaN values are set to 0 (as deep learning only takes integers)
X = df[feature_columns].fillna(0)
# y is the labels of the transport type (bus/car/train) 
y = df["label"]

<<<<<<< HEAD
#setting the training to testing split (we decided on 8:2)
=======
# Train and test split
>>>>>>> 65a0f9952e310d540ef377b4cadda5dbcdb5c10c
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2, #20% of data is reserved for testing
    random_state=42,
    stratify=y
)

<<<<<<< HEAD
# Standardising features of the data to better fit the model
=======
# Feature scaling
# Neural networks work best when features are standardised
>>>>>>> 65a0f9952e310d540ef377b4cadda5dbcdb5c10c
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

<<<<<<< HEAD
#Building the tensorflow model
=======
# Tenserflow model build
>>>>>>> 65a0f9952e310d540ef377b4cadda5dbcdb5c10c
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

<<<<<<< HEAD
#Training the tensorflow model
=======
# Train the model (15 cycles)
>>>>>>> 65a0f9952e310d540ef377b4cadda5dbcdb5c10c
history = model.fit(
    X_train,
    y_train,
    epochs=15, #We decided on 15 epochs but can be tweaked to prevent overfitting
    batch_size=32,
    validation_split=0.2
)

<<<<<<< HEAD
#--- Data Visualisation ---
=======
# Plot the training accuracy and loss values
>>>>>>> 65a0f9952e310d540ef377b4cadda5dbcdb5c10c
plt.figure(figsize=(12, 5))

# Plotting accuracy 
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label="Train Accuracy")
plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Plotting loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
<<<<<<< HEAD

#Displaying Accuracy & Loss
plt.tight_layout()
plt.show()

#Creating confusion matrix
=======
plt.tight_layout()
plt.show()

# Confusion Matrix

>>>>>>> 65a0f9952e310d540ef377b4cadda5dbcdb5c10c
y_pred = np.argmax(model.predict(X_test), axis=1)
cm = confusion_matrix(y_test, y_pred)
#Plotting confusion matrix heatmap
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

#Displaying confusion matrix
plt.show()

<<<<<<< HEAD
#Classification report
=======
>>>>>>> 65a0f9952e310d540ef377b4cadda5dbcdb5c10c
print("\nClassification Report:\n")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Bus", "Car", "Train"]
))
print("\nAll tasks completed successfully!")