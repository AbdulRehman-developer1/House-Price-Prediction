# 🏠 House Price Prediction

## Overview
An AI-powered Streamlit web app that predicts house prices using a trained Gradient Boosting Regressor. Users enter property details (area, bedrooms, location score, condition, etc.) and get an instant price prediction with a premium, glassmorphism-styled interface.

## Tech Stack
- Python
- Streamlit
- scikit-learn (Gradient Boosting Regressor)
- Pandas / NumPy
- Joblib

## Project Structure
```
House_Price_Prediction/
├── app.py                  # Streamlit application
├── requirements.txt        # Dependencies
├── models/artifact.pkl     # Trained model + Encoder + Scaler
├── data set/*.csv          # Training data
├── plots/*.png             # EDA visualizations
└── house-price-prediction-notebook.ipynb   # Training notebook
```

## Run Commands
```bash
# Create & activate virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Author
**Abdul Rehman**
