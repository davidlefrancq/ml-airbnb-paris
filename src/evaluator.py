import joblib
import pandas as pd
import numpy as np
import logging
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from predictor import Predictor

class ModelEvaluator:
    def __init__(self, models_dir='models'):
        self.predictor = Predictor(models_dir)
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        logger = logging.getLogger('MLEvaluator')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def evaluate(self, test_data_path: str):
      self.test_data = pd.read_csv(test_data_path)
      nb = len(self.test_data)
      self.logger.info(f"Number of elements: {nb}")
      
      columns_used = ['latitude', 'longitude', 'zipcode', 'accommodates', 'room_type', 'beds', 'price', 'minimum_nights']
      self.test_data = self.test_data[columns_used]
      
      self.test_data = self.test_data.dropna(subset=['price'])
      
      if not np.issubdtype(self.test_data['price'].dtype, np.number):
          self.test_data['price'] = self.test_data['price'].str.replace(r'[^\d.]', '', regex=True).astype(float)
      if not np.issubdtype(self.test_data['zipcode'].dtype, np.number):
          self.test_data['zipcode'] = self.test_data['zipcode'].str.replace(r'[^\d]', '', regex=True).astype(float)

      for column in ['latitude', 'longitude', 'zipcode', 'accommodates', 'beds', 'price', 'minimum_nights']:
          Q1 = self.test_data[column].quantile(0.25)
          Q3 = self.test_data[column].quantile(0.75)
          IQR = Q3 - Q1
          self.test_data = self.test_data[(self.test_data[column] >= Q1 - 1.5 * IQR) & (self.test_data[column] <= Q3 + 1.5 * IQR)]
      
      # Reset index
      self.test_data = self.test_data.reset_index(drop=True)
      
      y_true = self.test_data['price']
      
      self.logger.info(f"Number of elements: {len(self.test_data)}")
      
      results = {}
      nb_rows = len(self.test_data)
      for i, row in enumerate(self.test_data.itertuples()):
        self.logger.info(f"Processing row {i+1}/{nb_rows}")
        sample = {
          'latitude': row.latitude,
          'longitude': row.longitude,
          'zipcode': row.zipcode,
          'accommodates': row.accommodates,
          'room_type': row.room_type,
          'beds': row.beds,
          'minimum_nights': row.minimum_nights
        }
        # self.logger.info(f"{sample}")
        predictions = self.predictor.predict(sample)
        
        for model_name, pred_price in predictions.items():
          if model_name not in results:
              results[model_name] = []
          results[model_name].append(pred_price)
      
      metrics = {}
      for model_name, predictions in results.items():
          metrics[model_name] = {
              'RMSE': np.sqrt(mean_squared_error(y_true, predictions)),
              'MAE': mean_absolute_error(y_true, predictions),
              'R2': r2_score(y_true, predictions)
          }
          
      return metrics

if __name__ == '__main__':
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate('data/paris_airbnb_test.csv')
    
    print("\nEvaluation Results:")
    for model, scores in metrics.items():
        print(f"\n{model}:")
        for metric, value in scores.items():
            print(f"{metric}: {value:.4f}")