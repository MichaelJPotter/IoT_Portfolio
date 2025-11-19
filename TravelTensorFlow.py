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
#import matplotlib.pyplot as plt
#sklearn for data processing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
#seaborn for data visualisation
import seaborn as sns
#tensorflow for deeplearning
import tensorflow as tf
from tensorflow.keras import layers, models

def load_labeled_csvs(basePath):
    """
    This function:
    - 
    - Loads all CSV files inside each folder
    - Adds a 'label' column based on the folder name
    - Combines all CSVs into one big dataset
    """

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

    if not data:        # Debug print, we had issues reading the files

        raise ValueError("No CSV files found. Check basePath and folder names (bus/car/train).")

    return pd.concat(data, ignore_index=True)


# Used absolute path so there's no confusion about working directory
basePath = r"C:\Users\kikit\OneDrive - Bath Spa University\gitHub\IoT_Portfolio\MCJASM-Group-Data\MCJASM-Group-Data"

df = load_labeled_csvs(basePath)
print("Loaded dataset shape:", df.shape)

