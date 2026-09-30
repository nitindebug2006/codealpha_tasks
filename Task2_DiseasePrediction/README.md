# CodeAlpha Disease Prediction

## Project Overview

This project predicts the possibility of heart disease using patient medical data and machine learning classification algorithms.

## Objective

The objective is to apply classification techniques to structured medical data and predict whether heart disease is present or not.

## Dataset

The project uses the UCI Heart Disease dataset.

The dataset contains patient information such as:

- Age
- Sex
- Chest pain type
- Resting blood pressure
- Cholesterol
- Fasting blood sugar
- Resting ECG
- Maximum heart rate
- Exercise induced angina
- ST depression
- Slope
- Number of major vessels
- Thalassemia

## Machine Learning Models

The following classification algorithms were implemented:

1. Logistic Regression
2. Support Vector Machine (SVM)
3. Random Forest
4. XGBoost

## Model Results

| Model | Accuracy |
|---|---:|
| Logistic Regression | 83.33% |
| SVM | 85.00% |
| Random Forest | 85.00% |
| XGBoost | 81.67% |

## Project Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Preprocessing
   ↓
Train/Test Split
   ↓
Machine Learning Models
   ↓
Model Evaluation
   ↓
Disease Prediction
   ↓
Streamlit Web Application