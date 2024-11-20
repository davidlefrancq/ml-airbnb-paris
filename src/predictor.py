import os
import joblib
import pandas as pd
import logging
from typing import Dict

class Predictor:
  def __init__(self, models_dir='models'):
    self.models_dir = models_dir
    self.logger = self._setup_logger()
    self._load_resources()

  def _setup_logger(self):
    logger = logging.getLogger('MLPredictor')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

  def _load_resources(self):
    try:
      # Chargement des encodeurs
      self.encoders = {
        'room_type': joblib.load(os.path.join(self.models_dir, 'encoder_room_type.pkl'))
      }
      
      # Chargement des scalers
      self.scalers = {
        'accommodates': joblib.load(os.path.join(self.models_dir, 'scaler_accommodates.pkl')),
        'bathrooms': joblib.load(os.path.join(self.models_dir, 'scaler_bathrooms.pkl')),
        'beds': joblib.load(os.path.join(self.models_dir, 'scaler_beds.pkl')),
        'price': joblib.load(os.path.join(self.models_dir, 'scaler_price.pkl')),
        'latitude': joblib.load(os.path.join(self.models_dir, 'scaler_latitude.pkl')),
        'longitude': joblib.load(os.path.join(self.models_dir, 'scaler_longitude.pkl')),
        'minimum_nights': joblib.load(os.path.join(self.models_dir, 'scaler_minimum_nights.pkl')),
        'zipcode': joblib.load(os.path.join(self.models_dir, 'scaler_zipcode.pkl'))
      }
      
      # Chargement de tous les modèles
      self.models = {
        'random_forest': joblib.load(os.path.join(self.models_dir, 'random_forest.pkl')),
        'gradient_boosting': joblib.load(os.path.join(self.models_dir, 'gradient_boosting.pkl')),
        'svr': joblib.load(os.path.join(self.models_dir, 'svr.pkl')),
        'linear_regression': joblib.load(os.path.join(self.models_dir, 'linear_regression.pkl'))
      }
      
      self.logger.info("Resources loaded successfully")
    except Exception as e:
      self.logger.error(f"Error loading resources: {str(e)}")
      raise

  def get_possible_values(self):
    return {
      'room_type': self.encoders['room_type'].classes_.tolist()
    }

  def predict(self, data: dict) -> Dict[str, float]:
    """
    Prédire le prix d'un logement Airbnb avec tous les modèles.
    
    Args:
        data: dict avec les clés suivantes:
            - accommodates: int
            - bathrooms: float
            - beds: int
            - latitude: float
            - longitude: float
            - room_type: str
            - minimum_nights: int
            - zipcode: int
    
    Returns:
        Dict[str, float]: prix prédit par chaque modèle
    """
    try:
      # Conversion en DataFrame
      df = pd.DataFrame([data])
      
      # Encodage des variables catégorielles
      df['room_type'] = self.encoders['room_type'].transform(df['room_type'])
      
      # Normalisation des variables numériques
      if 'accommodates' in df.columns: df['accommodates'] = self.scalers['accommodates'].transform(df[['accommodates']])
      if 'bathrooms' in df.columns: df['bathrooms'] = self.scalers['bathrooms'].transform(df[['bathrooms']])
      if 'beds' in df.columns: df['beds'] = self.scalers['beds'].transform(df[['beds']])
      if 'latitude' in df.columns: df['latitude'] = self.scalers['latitude'].transform(df[['latitude']])
      if 'longitude' in df.columns: df['longitude'] = self.scalers['longitude'].transform(df[['longitude']])
      if 'minimum_nights' in df.columns: df['minimum_nights'] = self.scalers['minimum_nights'].transform(df[['minimum_nights']])
      if 'zipcode' in df.columns: df['zipcode'] = self.scalers['zipcode'].transform(df[['zipcode']])
      
      # Prédictions avec tous les modèles
      predictions = {}
      for name, model in self.models.items():
        price_scaled = model.predict(df)[0]
        price = self.scalers['price'].inverse_transform([[price_scaled]])[0][0]
        predictions[name] = round(price, 2)
      return predictions
          
    except Exception as e:
      self.logger.error(f"Error making prediction: {str(e)}")
      raise