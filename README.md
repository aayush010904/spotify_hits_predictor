# Spotify Hits Predictor

## Introduction
Music streaming platforms like Spotify have transformed the way we discover and listen to music. By analyzing key audio features of songs, this project aims to predict whether a song is likely to become a hit. The dataset used spans multiple decades (1960s-2010s) and includes various musical attributes.

## Project Structure
- **Data Import & Preprocessing**: Loads multiple datasets and combines them into a single dataframe.
- **Exploratory Data Analysis (EDA)**: Visualizations to understand key audio features influencing hit songs.
- **Feature Engineering & Selection**: Identifies significant attributes for training models.
- **Dimensionality Reduction (PCA)**: Reduces redundancy in data for better model performance.
- **Machine Learning Models**: Trains and evaluates models including:
  - Support Vector Machine (SVM)
  - Random Forest Classifier
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
- **Hyperparameter Tuning**: Uses RandomizedSearchCV to optimize model performance.

## Libraries Used
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_score, recall_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")
```

## Dataset
The dataset consists of hit and non-hit songs from multiple decades with features such as:
- **Danceability**
- **Energy**
- **Loudness**
- **Valence**
- **Tempo**
- **Speechiness**
- **Acousticness**
- **Instrumentalness**
- **Liveness**
- **Chorus hit sections**

## Exploratory Data Analysis
- **Top Artists with Most Hits**: A bar plot visualizing the most successful artists.
- **Feature Correlation Heatmap**: Identifies relationships between audio features and hit probability.
- **Hit vs Non-Hit Distribution**: A grouped bar graph showing the distribution of songs.

## Machine Learning Models & Results
Models were trained on the dataset, and their performances were evaluated using accuracy and F1-score:

| Model               | Accuracy  | 
|--------------------|----------|
| SVM               | 73.72%    |
| Random Forest     | 73.47%    |
| Logistic Regression | 69.64%    |
| KNN               | 71.76%    | 

To further improve the accuracy, the subset of data (2000s to 2010s) was used, keeping evolving music trends in mind.

After further preprocessing and retraining, the following metrics were obtained:

| Model               | Accuracy  | 
|--------------------|----------|
| SVM               | 82.56%    |
| Random Forest     | 82.48%    |
| Logistic Regression | 80.93%    |
| KNN               | 79.30%    | 

After choosing a temporal dataset, we can clearly see the jump in best accuracy, from 73.72% to 82.56%. To further improve this, hyperparameter tuning of SVM was done.


## Hyperparameter Tuning
- **Support Vector Machine (SVM)** was optimized using `RandomizedSearchCV`.
- Best parameters:
```python
{'C': 6.1848, 'gamma': 'auto', 'kernel': 'rbf'}
```
- Final retrained model achieved **82.80% accuracy**.

## Conclusion
This project showcases the power of machine learning in predicting hit songs. Future enhancements could include:
- **Lyrical Sentiment Analysis**
- **Social Media Engagement Metrics**
- **Deep Learning Models for better predictions**

## References
- *SpotHitPy: A Study For ML-Based Song Hit Prediction Using Spotify* ([ResearchGate](https://www.researchgate.net/publication/367280936_SpotHitPy_A_Study_For_ML-Based_Song_Hit_Prediction_Using_Spotify))

