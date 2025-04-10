# SGD-Yield-Predictor

## Overview

SGD-Yield-Predictor is a machine learning project aimed at predicting crop yields using the Stochastic Gradient Descent (SGD) regression algorithm. The model is trained on agricultural datasets and deployed using Streamlit, providing an interactive interface for users to input data and obtain yield predictions.

## Features

- **Data Preprocessing**: Cleans and prepares agricultural datasets for modeling.
- **Model Training**: Utilizes SGD regression for predictive modeling.
- **Interactive Interface**: Deployed with Streamlit for user-friendly interactions.

## Installation

To set up the project locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Atharvasaraiya/SGD-Yield-Predictor.git
   
##  Features

- Data preprocessing and feature engineering using Scikit-learn
- Polynomial regression with SGD
- Interactive Streamlit web application
- Real-time prediction based on user input
- Ready-to-use trained model and preprocessor files

## Project Structure

   - Crops Prediction.ipynb: Jupyter Notebook containing data analysis and model training code.

   - App.py: Main script for the Streamlit application.

   - Requirements.txt: Lists all Python dependencies required for the project.

   - Model.pkl, sc.pkl, pf.pkl, features.pkl: Serialized model and preprocessing objects.

## TechStack

- Python
- Scikit-learn
- Pandas & NumPy
- Matplotlib & Seaborn
- Streamlit

## Input Format Example

Area: Albania
Crop: Soybeans
Rainfall: 1485.0 mm
Pesticides: 121.00 tonnes
Avg. Temperature: 16.37°C

## Model Info

Model: SGDRegressor with PolynomialFeatures(degree=2)
Target: Crop Yield (hg/ha_yield)
Performance: Measured using R² score





 
  



