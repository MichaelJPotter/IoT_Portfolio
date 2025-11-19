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
    - Looks inside the 'car', 'bus', and 'train' folders
    - Loads all CSV files inside each folder
    - Adds a 'label' column based on the folder name
    - Combines all CSVs into one big dataset
    """

    data = []

    # Map folder names to numeric labels for machine learning as Neural Networks cant handle texts 
    classMap = {
        "Bus": 0,
        "Car": 1,
        "Train": 2
    }

    # Loop through each folder
    for transportMode, label in classMap.items():
        folder = os.path.join(basePath, transportMode , "*.csv")  # Path to CSVs
        files = glob.glob(folder)  # Get all files in that folder

        # Load each CSV in the folder
        for f in files:
            df = pd.read_csv(f)
            df["label"] = label   # Assign label based on folder
            data.append(df)

    # Combine all CSVs into one DataFrame
    return pd.concat(data, ignore_index=True)

basePath = "IoT_Portfolio/MCJASM-Group-Data.zip/MCJASM-Group-Data"

df = load_labeled_csvs(basePath)

print("Loaded dataset shape:", df.shape)

