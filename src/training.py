import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

class Trainner():
  def __init__(self, data_path_file="data/data-ligth.csv"):
    self.data_path_file = data_path_file
    self.models = {
      'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
      'gradient_boosting': GradientBoostingRegressor(random_state=42),
      'linear_regression': LinearRegression()
    }
    self.best_model = None
    self.best_model_name = None
    self.best_score = float('-inf')

  # Load data
  def load_data(self):
    if not self.data_path_file.endswith(".csv"):
      raise ValueError("\nData file must be a CSV file")
    if not os.path.exists(self.data_path_file):
      raise FileNotFoundError("\nData file not found")
    self.data = pd.read_csv(self.data_path_file)
    print("\nData loaded")
    print(self.data.head())

  # Clean data
  def clean_data(self):
    self.data.info()
    # colonne conservé: neighbourhood, latitude, longitude, room_type, price, minimum_nights
    colomns_used = ['neighbourhood', 'latitude', 'longitude', 'room_type', 'price', 'minimum_nights']
    
    # retirer les colonnes non utilisées
    self.data = self.data[colomns_used]
      
    # retirer les lignes avec le price manquant
    self.data = self.data.dropna(subset=['price'])
    self.data.info()

  # Normalize data
  def normalize_data(self):
    pass

  # Visualize data
  def visualize_data(self):
    pass

  # Split data
  def split_data(self):
    pass

  # Train models
  def train_models(self):
    pass

  # Evaluate models
  def evaluate_models(self):
    pass

  # Save models
  def save_models(self, model_path_folder="models/"):
    pass
  
  def start(self):
    self.load_data()
    self.clean_data()
    self.normalize_data()
    self.visualize_data()
    self.split_data()
    self.train_models()
    self.evaluate_models()
    self.save_models()
  
# Run training
if __name__ == '__main__':
  trainer = Trainner()
  trainer.start()