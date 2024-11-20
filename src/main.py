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
                'neighbourhood': joblib.load(os.path.join(self.models_dir, 'encoder_neighbourhood.pkl')),
                'room_type': joblib.load(os.path.join(self.models_dir, 'encoder_room_type.pkl'))
            }
            
            # Chargement des scalers
            self.scalers = {
                'price': joblib.load(os.path.join(self.models_dir, 'scaler_price.pkl')),
                'latitude': joblib.load(os.path.join(self.models_dir, 'scaler_latitude.pkl')),
                'longitude': joblib.load(os.path.join(self.models_dir, 'scaler_longitude.pkl')),
                'minimum_nights': joblib.load(os.path.join(self.models_dir, 'scaler_minimum_nights.pkl'))
            }
            
            # Chargement de tous les modèles
            self.models = {
                'random_forest': joblib.load(os.path.join(self.models_dir, 'random_forest.pkl')),
                'gradient_boosting': joblib.load(os.path.join(self.models_dir, 'gradient_boosting.pkl')),
                'linear_regression': joblib.load(os.path.join(self.models_dir, 'linear_regression.pkl'))
            }
            
            self.logger.info("Resources loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading resources: {str(e)}")
            raise

    def get_possible_values(self):
        return {
            'neighbourhood': self.encoders['neighbourhood'].classes_.tolist(),
            'room_type': self.encoders['room_type'].classes_.tolist()
        }

    def predict(self, data: dict) -> Dict[str, float]:
        """
        Prédire le prix d'un logement Airbnb avec tous les modèles.
        
        Args:
            data: dict avec les clés suivantes:
                - neighbourhood: str
                - latitude: float
                - longitude: float
                - room_type: str
                - minimum_nights: int
        
        Returns:
            Dict[str, float]: prix prédit par chaque modèle
        """
        try:
            # Conversion en DataFrame
            df = pd.DataFrame([data])
            
            # Encodage des variables catégorielles
            df['neighbourhood'] = self.encoders['neighbourhood'].transform(df['neighbourhood'])
            df['room_type'] = self.encoders['room_type'].transform(df['room_type'])
            
            # Normalisation des variables numériques
            df['latitude'] = self.scalers['latitude'].transform(df[['latitude']])
            df['longitude'] = self.scalers['longitude'].transform(df[['longitude']])
            df['minimum_nights'] = self.scalers['minimum_nights'].transform(df[['minimum_nights']])
            
            # Prédictions avec tous les modèles
            predictions = {}
            for name, model in self.models.items():
                price_scaled = model.predict(df)[0]
                price = self.scalers['price'].inverse_transform([[price_scaled]])[0][0]
                predictions[name] = round(price, 2)
                self.logger.info(f"Prediction with {name}: {price:.2f}€")
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error making prediction: {str(e)}")
            raise

# Exemple d'utilisation
if __name__ == '__main__':
    predictor = Predictor()
    
    # print("\nValeurs possibles :")
    # print(predictor.get_possible_values())
        
    # Exemples
    sample_data_list = [
      {
        'neighbourhood': 'Louvre',
        'latitude': 48.856614,
        'longitude': 2.352222,
        'room_type': 'Entire home/apt',
        'minimum_nights': 2
      },
      {
        'neighbourhood': 'Hôtel-de-Ville',
        'latitude': 48.85751,
        'longitude': 2.35511,
        'room_type': 'Entire home/apt',
        'minimum_nights': 3
      },
      {
        'neighbourhood': 'Entrepôt',
        'latitude': 48.87151,
        'longitude': 2.37219,
        'room_type': 'Entire home/apt',
        'minimum_nights': 4
      },

      {
        'neighbourhood': 'Entrepôt',
        'latitude': 48.86954,
        'longitude': 2.36702,
        'room_type': 'Entire home/apt',
        'minimum_nights': 30
      },

      {
        'neighbourhood': 'Gobelins',
        'latitude': 48.83593,
        'longitude': 2.35108,
        'room_type': 'Entire home/apt',
        'minimum_nights': 2
      }
    ]
    
    # Predictions
    for sample_data in sample_data_list:
      print("\n")
      print(sample_data)
      predictions = predictor.predict(sample_data)
      print("Prix prédits:")
      for model_name, price in predictions.items():
        print(f"{model_name}: {price}€")