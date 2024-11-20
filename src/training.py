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

    # Identifier les valeurs aberrantes pour le prix
    price_q1 = self.data['price'].quantile(0.25)
    price_q3 = self.data['price'].quantile(0.75)
    price_iqr = price_q3 - price_q1
    price_lower_bound = price_q1 - 1.5 * price_iqr
    price_upper_bound = price_q3 + 1.5 * price_iqr
    
     # Supprimer les valeurs aberrantes pour le prix
    self.data = self.data[(self.data['price'] >= price_lower_bound) & (self.data['price'] <= price_upper_bound)]
    
    # Identifier les valeurs aberrantes pour le nombre de nuits minimum
    min_nights_q1 = self.data['minimum_nights'].quantile(0.25)
    min_nights_q3 = self.data['minimum_nights'].quantile(0.75)
    min_nights_iqr = min_nights_q3 - min_nights_q1
    min_nights_lower_bound = min_nights_q1 - 1.5 * min_nights_iqr
    min_nights_upper_bound = min_nights_q3 + 1.5 * min_nights_iqr
    
    # Supprimer les valeurs aberrantes pour le nombre de nuits minimum
    self.data = self.data[(self.data['minimum_nights'] >= min_nights_lower_bound) & (self.data['minimum_nights'] <= min_nights_upper_bound)]
    
    self.data.info()

  # Normalize data
  def normalize_data(self):
    # Encoder les variables catégorielles
    label_encoder = LabelEncoder()
    self.data['neighbourhood'] = label_encoder.fit_transform(self.data['neighbourhood'])
    self.data['room_type'] = label_encoder.fit_transform(self.data['room_type'])
    self.data.info()
    
    # Standardiser les variables numériques
    scaler = StandardScaler()
    self.data[['price', 'minimum_nights']] = scaler.fit_transform(self.data[['price', 'minimum_nights']])

    # Vérifier les statistiques descriptives des variables normalisées
    print("\nStatistiques descriptives après normalisation:")
    print(self.data.describe())

  # Visualize data
  def visualize_data(self):
    # Visualiser la distribution des prix
    plt.figure(figsize=(8, 6))
    sns.histplot(data=self.data, x="price", kde=True)
    plt.title("Distribution des prix")
    plt.savefig("plots/price_distribution.png")
    plt.close()

    # Visualiser la distribution du nombre de nuits minimum
    plt.figure(figsize=(8, 6))
    sns.histplot(data=self.data, x="minimum_nights", kde=True)
    plt.title("Distribution du nombre de nuits minimum")
    plt.savefig("plots/minimum_nights_distribution.png")
    plt.close()
    
    # Visualiser la relation entre le prix et le type de logement
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=self.data, x="room_type", y="price")
    plt.title("Prix moyen par type de logement")
    plt.xlabel("Type de logement")
    plt.ylabel("Prix")
    plt.savefig("plots/price_by_room_type.png")
    plt.close()

    # Visualiser la relation entre le prix et la localisation (quartier)
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=self.data, x="longitude", y="latitude", hue="price")
    plt.title("Localisation des annonces et prix")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.savefig("plots/price_by_location.png")
    plt.close()
    
    # Calculer la matrice de corrélation
    corr_matrix = self.data.corr()

    # Visualiser la matrice de corrélation
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="YlOrRd")
    plt.title("Matrice de corrélation")
    plt.savefig("plots/correlation_matrix.png")
    plt.close()

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